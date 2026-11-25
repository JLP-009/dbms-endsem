# DBMS Lab End-Sem — Mobile Game Telemetry Analysis
**NIT Warangal · 2026**

## How to run
```bash
pip install -r requirements.txt
python analysis.py > output.txt 2>&1
```
Place the 5 data CSVs inside a folder called `dbms assignment/` in the same directory before running.

## What it covers
| Section | Topic |
|---|---|
| §0 | Data quality audit (duplicates, linked devices) |
| §1a | Daily health — DAU, revenue, ARPDAU by platform |
| §1b | Retention cohort table D1/D7/D14/D30 |
| §1c | Spender behaviour — concentration, time-to-first-purchase, repeat gap |
| §1d | Rewarded ad engagement & cap saturation |
| §2 | Deep-dive: ARPDAU decline = Simpson's Paradox (mix-shift, not quality drop) |
| §3 | Acquisition channel comparison with chi-squared & Mann-Whitney tests |
| §4 | Churn prediction — logistic regression, ROC-AUC 0.685, threshold economics |

## Stack
DuckDB · pandas · scipy · scikit-learn
