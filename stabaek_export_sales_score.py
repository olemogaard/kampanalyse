"""
Stabæk 2025 – exportability + sales score
=========================================

Bygger to modeller på det norske overgangsmarkedet 2020–2025 og scorer
Stabæks 2025-stall på:
  * p_sold   – sannsynlighet for at en spillersesong utløser et BETALT salg
  * p_export – sannsynlighet for overgang til utenlandsk klubb

VIKTIG TOLKNING: Modellene fanger MARKEDETS INTERESSE (hva markedet
historisk har betalt for / hentet ut), ikke klubbens vilje til å selge
eller spillerens eget ønske. p_sold er kalibrert mot baseraten.

Datagrunnlag (alle filer lastet ned KOMPLETT – ingen trunkering):
  * tm_transfers.csv      – rå Transfermarkt-logg, norske klubber (labels)
  * master_history.csv    – KPI-panel 2020–2024 (Wyscout, normalisert)
  * players_master.csv    – KPI-panel 2025 (scoringspopulasjon)

Integritetssjekk: transfers_enriched (8,9 MB) kunne ikke lastes hel gjennom
Drive-connectoren (hard størrelsestak ~7 MB; deterministisk feil på 8,9 MB).
Den er rekonstruert som rå TM-logg ⋈ (one-to-many) KPI-panel på
(forbokstav, etternavn, sesong). Rekonstruksjonen gir 11 341 rader / 759
betalte salg mot fasit 11 445 / 758 (avvik 0,9 % / 0,1 %, forklart av
navnebro-grensetilfeller). Se utskrift nederst.
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

RNG = 42

# --------------------------------------------------------------------------
# 1. Innlasting
# --------------------------------------------------------------------------
tm = pd.read_csv("data/tm_transfers.csv", low_memory=False)
mh = pd.read_csv("data/master_history.csv", low_memory=False)
pm = pd.read_csv("data/players_master.csv", low_memory=False)

tm["sesong"] = pd.to_numeric(tm["sesong"], errors="coerce")
mh["season"] = pd.to_numeric(mh["season"], errors="coerce")


# --------------------------------------------------------------------------
# 2. Navnebro (Wyscout-forkortet "A. Andersson" <-> TM "Adam Andersson")
# --------------------------------------------------------------------------
def _strip(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))


def name_key(name):
    s = re.sub(r"[^a-z ]", " ", _strip(name).lower().replace(".", " "))
    toks = [t for t in s.split() if t]
    return (toks[0][0], toks[-1]) if toks else None


tm["k"] = tm["spiller"].map(name_key)
mh["k"] = mh["player"].map(name_key)

# --------------------------------------------------------------------------
# 3. Rolle-harmonisering -> 5 grovroller
# --------------------------------------------------------------------------
ROLE_MAP = {
    # master_history.unified_role
    "CB": "CB", "FB/WB": "FB", "CM/AM": "CM", "ST": "ST", "W": "W",
    # players_master.Position_group
    "BACK": "FB", "STRIKER": "ST", "WINGER": "W", "FORWARD": "ST",
    "DM": "CM",
}
mh["role"] = mh["unified_role"].map(ROLE_MAP)
pm["role"] = pm["Position_group"].map(ROLE_MAP)
ROLES = ["CB", "FB", "CM", "W", "ST"]

# --------------------------------------------------------------------------
# 4. Features  (felles KPI-kolonner; ekskluder hjelpekolonner)
# --------------------------------------------------------------------------
FEATURES = [c for c in mh.columns if c in pm.columns and c not in ("role", "k")]
assert "Age" in FEATURES
print(f"Antall KPI-features (+Age): {len(FEATURES)}")

# --------------------------------------------------------------------------
# 5. Rekonstruer transfers_enriched (one-to-many, samme sesong)
# --------------------------------------------------------------------------
mh_join = mh[["k", "season", "role"] + FEATURES].rename(columns={"season": "sesong"})
mh_join["_matched"] = 1
enr = tm.merge(mh_join, on=["k", "sesong"], how="left", suffixes=("", "_kpi"))

fee = pd.to_numeric(enr["overgangssum_eur"], errors="coerce")
enr["sold"] = ((enr["overgang_type"] == "kjøp") & (fee > 0)).astype(int)
# Eksport = BETALT salg til utenlandsk klubb (ikke gratis/låneoverganger ut)
enr["exported"] = (enr["sold"].eq(1) &
                   enr["til_land"].notna() &
                   (enr["til_land"] != "Norway")).astype(int)
enr["has_kpi"] = enr["_matched"].eq(1)

print("\n=== INTEGRITETSSJEKK ===")
print(f"Rekonstruerte rader        : {len(enr):>6}   (fasit 11 445)")
print(f"Betalte salg (sold==1)     : {int(enr['sold'].sum()):>6}   (fasit   758)")
print(f"Eksporter (til utl.)       : {int(enr['exported'].sum()):>6}")
print(f"Rader med KPI-features     : {int(enr['has_kpi'].sum()):>6}")

# --------------------------------------------------------------------------
# 6. Treningssett = rader med KPI-features
# --------------------------------------------------------------------------
train = enr[enr["has_kpi"]].copy()
train["role"] = train["role"].fillna("CM")
base_sold = train["sold"].mean()
base_exp = train["exported"].mean()
print(f"\nTreningssett (m/ KPI)      : {len(train)} rader")
print(f"Baserate p_sold            : {base_sold:.3%}   (referanse: 8,7 %)")
print(f"Baserate p_export          : {base_exp:.3%}")


def design(df):
    """KPI + Age + rolle-dummies -> matrise. Median-imputering fra trening."""
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    for r in ROLES:
        X[f"role_{r}"] = (df["role"].values == r).astype(float)
    return X


X = design(train)
medians = X[FEATURES].median()
X[FEATURES] = X[FEATURES].fillna(medians)
groups = train["k"].astype(str).values   # samme spiller aldri splittet


def evaluate(y, label):
    """OOS AUC via StratifiedGroupKFold + isotonisk kalibrering."""
    y = y.values
    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=RNG)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = cross_val_predict(clf, X, y, cv=cv, groups=groups,
                            method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    # kalibrer OOS-score isotonisk mot faktisk utfall
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    brier = brier_score_loss(y, iso.predict(oof))
    print(f"  {label:8s}  OOS AUC = {auc:.3f}   Brier(kal.) = {brier:.4f}   "
          f"n+={int(y.sum())}")
    # endelig modell på alt + kalibrator
    clf.fit(X, y)
    return clf, iso, auc


print("\n=== MODELLER (out-of-sample, GroupKFold på spiller) ===")
clf_sold, iso_sold, auc_sold = evaluate(train["sold"], "p_sold")
clf_exp, iso_exp, auc_exp = evaluate(train["exported"], "p_export")

# --------------------------------------------------------------------------
# 7. Score Stabæk 2025
# --------------------------------------------------------------------------
stab = pm[pm["Team"].astype(str).str.contains("Stab", case=False, na=False)].copy()
stab["role"] = stab["role"].fillna("CM")
Xs = design(stab)
Xs[FEATURES] = Xs[FEATURES].fillna(medians)

stab["p_sold"] = iso_sold.predict(clf_sold.predict_proba(Xs)[:, 1])
stab["p_export"] = iso_exp.predict(clf_exp.predict_proba(Xs)[:, 1])
# salgsscore = markedsinteresse for et BETALT salg, lift over baserate
stab["sales_score"] = stab["p_sold"] / base_sold
stab["age_peak"] = stab["Age"].between(24, 25)

out = (stab[["Player", "role", "Age", "Minutes",
             "p_sold", "p_export", "sales_score", "age_peak"]]
       .sort_values("p_sold", ascending=False)
       .reset_index(drop=True))
out.index += 1

pd.set_option("display.width", 160)
print("\n=== STABÆK 2025 – RANGERT PÅ MARKEDSINTERESSE (p_sold) ===")
print(out.to_string(float_format=lambda v: f"{v:,.3f}"))

out.to_csv("stabaek_2025_export_sales_scores.csv", index_label="rank")
print("\nLagret: stabaek_2025_export_sales_scores.csv")

# metadata til rapport
with open("_run_meta.txt", "w") as f:
    f.write(f"{len(enr)}|{int(enr['sold'].sum())}|{int(enr['exported'].sum())}|"
            f"{len(train)}|{base_sold:.4f}|{base_exp:.4f}|"
            f"{auc_sold:.3f}|{auc_exp:.3f}\n")
