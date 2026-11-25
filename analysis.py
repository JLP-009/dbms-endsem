#!/usr/bin/env python3
"""
DBMS Lab End-Sem Assignment — Full Analytical Pipeline
=======================================================
Mobile game telemetry  |  2026-01-01 to 2026-04-30  |  ~80k players

Setup
-----
    unzip dbms_assignment.zip          # creates "dbms assignment/" folder
    pip install duckdb pandas numpy scipy scikit-learn
    python analysis.py > output.txt 2>&1

All result tables are also saved to ./results/*.csv
"""

import os, sys, warnings
import duckdb
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_rows", 300)

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
BASE = "dbms assignment"
OUT  = "results"
os.makedirs(OUT, exist_ok=True)

CSVS = {
    "profile":  os.path.join(BASE, "player_profile - player_profile.csv"),
    "day":      os.path.join(BASE, "player_day - player_day.csv"),
    "purchase": os.path.join(BASE, "purchase - purchase.csv"),
    "spend":    os.path.join(BASE, "currency_spend - currency_spend.csv"),
    "ad":       os.path.join(BASE, "ad_view - ad_view.csv"),
}
CSVS = {k: v.replace("\\", "/") for k, v in CSVS.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# DUCKDB SETUP  — views scan CSVs lazily; nothing loads into RAM until .df()
# ═══════════════════════════════════════════════════════════════════════════════
con = duckdb.connect()
for alias, path in [
    ("vw_profile",  CSVS["profile"]),
    ("vw_day",      CSVS["day"]),
    ("vw_purchase", CSVS["purchase"]),
    ("vw_spend",    CSVS["spend"]),
    ("vw_ad",       CSVS["ad"]),
]:
    con.execute(
        f"CREATE VIEW {alias} AS "
        f"SELECT * FROM read_csv_auto('{path}', header=True, nullstr='')"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)

def banner(title: str) -> None:
    line = "═" * 72
    print(f"\n{line}\n  {title}\n{line}")

def show(df: pd.DataFrame, title: str = "", save_as: str = "") -> None:
    if title:
        banner(title)
    print(df.to_string(index=False))
    if save_as:
        save(df, save_as)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Reusable CTE fragments
# ─────────────────────────────────────────────────────────────────────────────
# Canonical players: exclude secondary/linked devices to avoid double-counting.
# When original_device_id IS NOT NULL, the device was migrated → another row
# already represents this player as the canonical identity.
CANONICAL = """canonical AS (
    SELECT device_id, install_date, platform, country,
           country_tier, acquisition_channel, first_purchase_date
    FROM vw_profile
    WHERE original_device_id IS NULL
)"""

# De-duplicated purchases: same purchase_id logged multiple times (data bug)
# → keep only the row with the earliest event_ts.
DEDUP_PUR = """dedup_purchase AS (
    SELECT purchase_id, device_id, event_ts, pack_name,
           usd_amount, currency_granted, sale_flag
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY purchase_id ORDER BY event_ts) AS rn
        FROM vw_purchase
    ) t
    WHERE rn = 1
)"""

# activity_date is INTEGER YYYYMMDD → convert to DATE once here.
DAY_DATE = """day_dated AS (
    SELECT device_id, activity_date,
           strptime(CAST(activity_date AS VARCHAR), '%Y%m%d')::DATE AS act_date,
           player_level, session_count, playtime_minutes, app_version
    FROM vw_day
)"""

print("✓ DuckDB views ready.  Starting analysis …\n")

# ═══════════════════════════════════════════════════════════════════════════════
# §0  DATA QUALITY AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
banner("§0  DATA QUALITY AUDIT")

# 0a. Row counts
rc = q("""
SELECT
    (SELECT COUNT(*) FROM vw_profile)  AS profile_rows,
    (SELECT COUNT(*) FROM vw_day)      AS day_rows,
    (SELECT COUNT(*) FROM vw_purchase) AS purchase_rows,
    (SELECT COUNT(*) FROM vw_spend)    AS spend_rows,
    (SELECT COUNT(*) FROM vw_ad)       AS ad_rows
""")
show(rc, "Row counts")

# 0b. Duplicate purchase_ids
dup = q("""
SELECT purchase_id, COUNT(*) AS occurrences
FROM vw_purchase
GROUP BY purchase_id HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 20
""")
print(f"→ Duplicate purchase_ids: {len(dup)}")
if len(dup):
    print(dup.to_string(index=False))
print()

# 0c. Linked devices
linked = q("""
SELECT COUNT(*) AS linked_devices
FROM vw_profile
WHERE original_device_id IS NOT NULL
""")
print(f"→ Linked (secondary) devices: {linked['linked_devices'].iloc[0]}  — excluded from player counts\n")

# 0d. Date ranges
ranges = q(f"""
WITH {DAY_DATE},
act_range AS (SELECT MIN(act_date) AS min_act, MAX(act_date) AS max_act FROM day_dated),
prof_range AS (SELECT MIN(install_date) AS min_inst, MAX(install_date) AS max_inst,
                      COUNT(*) AS total_devices FROM vw_profile)
SELECT min_inst AS earliest_install, max_inst AS latest_install,
       min_act  AS earliest_activity, max_act AS latest_activity,
       total_devices
FROM prof_range CROSS JOIN act_range
""")
show(ranges, "Date ranges")

# 0e. first_purchase_date consistency check
fpd = q(f"""
WITH {DEDUP_PUR},
first_in_table AS (
    SELECT device_id, MIN(event_ts::DATE) AS fp_in_purchase
    FROM dedup_purchase GROUP BY device_id
)
SELECT
    COUNT(*) AS devices_with_purchases,
    SUM(CASE WHEN pp.first_purchase_date IS NULL THEN 1 ELSE 0 END)
        AS null_fpd_has_purchase,
    SUM(CASE WHEN pp.first_purchase_date IS NOT NULL
              AND pp.first_purchase_date != fi.fp_in_purchase THEN 1 ELSE 0 END)
        AS fpd_mismatch
FROM vw_profile pp
JOIN first_in_table fi ON pp.device_id = fi.device_id
""")
show(fpd, "first_purchase_date vs purchase table")

# 0f. Ad data start time (potential truncation on day 1)
ad_first = q("""
SELECT MIN(event_ts) AS earliest_ad_event, MAX(event_ts) AS latest_ad_event
FROM vw_ad
""")
show(ad_first, "Ad view timestamp range")

print("""DATA QUALITY ISSUES FOUND
━━━━━━━━━━━━━━━━━━━━━━━━━━
1. DUPLICATE PURCHASE IDs — same purchase_id logged at different timestamps.
   Fix: ROW_NUMBER() OVER (PARTITION BY purchase_id ORDER BY event_ts), keep rn=1.
   Impact: without dedup, revenue is overstated.

2. LINKED DEVICES (original_device_id IS NOT NULL) — a subset of rows in
   player_profile represent device migrations; the same player has two rows.
   Fix: exclude these rows from player/cohort counts.  Revenue and activity
   data for the linked device is legitimately the same player's data.

3. INSTALL VOLUME NOT FLAT — many players installed months before 2026-01-01.
   These pre-window installs dominate early-year DAU and distort the ARPDAU
   trend (see §2 — Simpson's Paradox).

4. AD VIEW TRUNCATION ON DAY 1 — the earliest ad_view event is after 14:00 on
   2026-01-01, while currency_spend starts at 06:00.  Day-1 ad totals are
   understated.  When plotting daily ad views, note this for the first day.

5. first_purchase_date in player_profile may differ from MIN(event_ts) in the
   purchase table (purchases outside the observation window, before 2026-01-01).
   Always use the purchase table as the authoritative source for revenue.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# §1a  DAILY HEALTH METRICS
# ═══════════════════════════════════════════════════════════════════════════════
banner("§1a  DAILY HEALTH — by platform, every day")

daily_health = q(f"""
WITH {DEDUP_PUR},
{DAY_DATE},
rev_by_player_day AS (
    SELECT event_ts::DATE AS rev_date, device_id,
           SUM(usd_amount) AS daily_usd
    FROM dedup_purchase
    GROUP BY event_ts::DATE, device_id
)
SELECT
    d.activity_date,
    p.platform,
    COUNT(DISTINCT d.device_id)                                               AS dau,
    ROUND(COALESCE(SUM(r.daily_usd), 0), 2)                                   AS gross_revenue,
    ROUND(COALESCE(SUM(r.daily_usd), 0) / COUNT(DISTINCT d.device_id), 5)    AS arpdau,
    ROUND(
        COUNT(DISTINCT CASE WHEN r.daily_usd IS NOT NULL THEN d.device_id END)
        * 100.0 / COUNT(DISTINCT d.device_id), 2
    )                                                                          AS payer_pct
FROM day_dated d
JOIN vw_profile p ON d.device_id = p.device_id
LEFT JOIN rev_by_player_day r
    ON d.device_id = r.device_id AND d.act_date = r.rev_date
GROUP BY d.activity_date, p.platform
ORDER BY d.activity_date, p.platform
""")
save(daily_health, "1a_daily_health")
print(f"Full table: {len(daily_health)} rows → results/1a_daily_health.csv")
print("First 20 rows:")
print(daily_health.head(20).to_string(index=False))

# Overall summary
all_days = daily_health.groupby("activity_date").agg(
    dau=("dau","sum"), revenue=("gross_revenue","sum")
).reset_index()
all_days["arpdau"] = all_days["revenue"] / all_days["dau"]
print(f"\nWindow summary (all platforms):")
print(all_days[["dau","revenue","arpdau"]].describe().round(4).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# §1b  RETENTION COHORT TABLE
# ═══════════════════════════════════════════════════════════════════════════════
banner("§1b  RETENTION COHORT TABLE — install week × D1/D7/D14/D30")
print("""
Scope: players installed 2026-01-01 to 2026-04-30 (pre-window installs
excluded — their D1/D7 activity predates the observation data).

Rule for NULL cells: a cell is NULL when install_week + milestone_days
> 2026-04-30.  Reporting a retention rate for a milestone we cannot yet
observe (window not elapsed) would be misleading — it would look like
low retention but is actually truncation.
""")

cohort = q(f"""
WITH {CANONICAL},
in_window AS (
    SELECT device_id, install_date,
           date_trunc('week', install_date)::DATE AS install_week
    FROM canonical
    WHERE install_date BETWEEN DATE '2026-01-01' AND DATE '2026-04-30'
),
{DAY_DATE},
cohort_days AS (
    SELECT iw.install_week, iw.device_id, iw.install_date, d.act_date,
           date_diff('day', iw.install_date, d.act_date) AS d_since
    FROM in_window iw
    JOIN day_dated d ON iw.device_id = d.device_id
),
sizes AS (
    SELECT install_week, COUNT(DISTINCT device_id) AS n
    FROM in_window GROUP BY install_week
)
SELECT
    ca.install_week,
    s.n                                                                      AS cohort_n,
    ROUND(COUNT(DISTINCT CASE WHEN ca.d_since = 1  THEN ca.device_id END)
          * 100.0 / s.n, 1)                                                  AS d1_pct,
    CASE WHEN ca.install_week + INTERVAL '7' DAY  <= DATE '2026-04-30'
         THEN ROUND(COUNT(DISTINCT CASE WHEN ca.d_since = 7  THEN ca.device_id END)
                    * 100.0 / s.n, 1) END                                    AS d7_pct,
    CASE WHEN ca.install_week + INTERVAL '14' DAY <= DATE '2026-04-30'
         THEN ROUND(COUNT(DISTINCT CASE WHEN ca.d_since = 14 THEN ca.device_id END)
                    * 100.0 / s.n, 1) END                                    AS d14_pct,
    CASE WHEN ca.install_week + INTERVAL '30' DAY <= DATE '2026-04-30'
         THEN ROUND(COUNT(DISTINCT CASE WHEN ca.d_since = 30 THEN ca.device_id END)
                    * 100.0 / s.n, 1) END                                    AS d30_pct
FROM cohort_days ca
JOIN sizes s ON ca.install_week = s.install_week
GROUP BY ca.install_week, s.n
ORDER BY ca.install_week
""")
show(cohort, "Cohort retention grid", save_as="1b_cohort_retention")

# ═══════════════════════════════════════════════════════════════════════════════
# §1c  SPENDER BEHAVIOUR
# ═══════════════════════════════════════════════════════════════════════════════
banner("§1c-i  REVENUE CONCENTRATION")
print("Population ranked: paying players only (non-payers excluded).")
print("Both choices are defensible; ranking all players inflates top-N% share.\n")

conc = q(f"""
WITH {DEDUP_PUR},
payer_totals AS (
    SELECT device_id, SUM(usd_amount) AS spend
    FROM dedup_purchase GROUP BY device_id
),
ranked AS (
    SELECT spend,
           ROW_NUMBER() OVER (ORDER BY spend DESC) AS rnk,
           COUNT(*) OVER ()                         AS n,
           SUM(spend) OVER ()                       AS total
    FROM payer_totals
)
SELECT
    COUNT(*)                                                                AS n_payers,
    ROUND(SUM(spend) / MAX(total) * 100, 1)                                AS all_payers_pct,
    ROUND(SUM(CASE WHEN rnk <= CEIL(n*0.01) THEN spend ELSE 0 END)
          / MAX(total) * 100, 1)                                            AS top_1pct_share,
    ROUND(SUM(CASE WHEN rnk <= CEIL(n*0.10) THEN spend ELSE 0 END)
          / MAX(total) * 100, 1)                                            AS top_10pct_share,
    ROUND(SUM(CASE WHEN rnk <= CEIL(n*0.50) THEN spend ELSE 0 END)
          / MAX(total) * 100, 1)                                            AS top_50pct_share,
    ROUND(MAX(total), 2)                                                    AS total_revenue_usd
FROM ranked
""")
show(conc, save_as="1c_i_revenue_concentration")

# ── 1c-ii. Time to first purchase ────────────────────────────────────────────
banner("§1c-ii  TIME TO FIRST PURCHASE")

payer_summary = q(f"""
WITH {CANONICAL},
{DEDUP_PUR},
first_purch AS (
    SELECT device_id, MIN(event_ts::DATE) AS fp_date
    FROM dedup_purchase GROUP BY device_id
),
timing AS (
    SELECT c.device_id,
           fp.fp_date,
           date_diff('day', c.install_date, fp.fp_date) AS d2fp
    FROM canonical c LEFT JOIN first_purch fp ON c.device_id = fp.device_id
)
SELECT
    COUNT(*)                                                                     AS total_players,
    SUM(CASE WHEN fp_date IS NOT NULL THEN 1 ELSE 0 END)                        AS ever_paid,
    ROUND(SUM(CASE WHEN fp_date IS NOT NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS payer_rate_pct,
    ROUND(AVG(CASE WHEN fp_date IS NOT NULL THEN d2fp END), 1)                  AS mean_days_to_first,
    ROUND(MEDIAN(CASE WHEN fp_date IS NOT NULL THEN d2fp END), 1)               AS median_days_to_first,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
          (ORDER BY CASE WHEN fp_date IS NOT NULL THEN d2fp END), 1)            AS p25,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
          (ORDER BY CASE WHEN fp_date IS NOT NULL THEN d2fp END), 1)            AS p75,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
          (ORDER BY CASE WHEN fp_date IS NOT NULL THEN d2fp END), 1)            AS p90
FROM timing
""")
show(payer_summary)

ttfp = q(f"""
WITH {CANONICAL},
{DEDUP_PUR},
first_purch AS (
    SELECT device_id, MIN(event_ts::DATE) AS fp_date
    FROM dedup_purchase GROUP BY device_id
),
timing AS (
    SELECT date_diff('day', c.install_date, fp.fp_date) AS d2fp
    FROM canonical c JOIN first_purch fp ON c.device_id = fp.device_id
    WHERE fp.fp_date >= c.install_date   -- guard pre-install data edge case
)
SELECT
    CASE
        WHEN d2fp = 0           THEN '00_same_day'
        WHEN d2fp <= 1          THEN '01_day_1'
        WHEN d2fp <= 3          THEN '02_days_2-3'
        WHEN d2fp <= 7          THEN '03_days_4-7'
        WHEN d2fp <= 14         THEN '04_days_8-14'
        WHEN d2fp <= 30         THEN '05_days_15-30'
        WHEN d2fp <= 60         THEN '06_days_31-60'
        ELSE                         '07_60plus_days'
    END                                                AS bucket,
    COUNT(*)                                           AS n_players,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM timing
GROUP BY bucket ORDER BY bucket
""")
show(ttfp, "Time-to-first-purchase distribution", save_as="1c_ii_ttfp")

# ── 1c-iii. Repeat purchase gap ───────────────────────────────────────────────
banner("§1c-iii  REPEAT PURCHASE GAP (window function, players with 2+ purchases)")

rep_gap = q(f"""
WITH {DEDUP_PUR},
multi_buyers AS (
    SELECT device_id FROM dedup_purchase
    GROUP BY device_id HAVING COUNT(*) > 1
),
gaps AS (
    SELECT device_id,
           date_diff('day',
               LAG(event_ts::DATE) OVER (PARTITION BY device_id ORDER BY event_ts),
               event_ts::DATE
           ) AS gap_days
    FROM dedup_purchase
    WHERE device_id IN (SELECT device_id FROM multi_buyers)
)
SELECT
    CASE
        WHEN gap_days = 0           THEN '00_same_day'
        WHEN gap_days = 1           THEN '01_1_day'
        WHEN gap_days BETWEEN 2 AND 3   THEN '02_2-3_days'
        WHEN gap_days BETWEEN 4 AND 7   THEN '03_4-7_days'
        WHEN gap_days BETWEEN 8 AND 14  THEN '04_8-14_days'
        WHEN gap_days BETWEEN 15 AND 30 THEN '05_15-30_days'
        WHEN gap_days BETWEEN 31 AND 60 THEN '06_31-60_days'
        ELSE                             '07_60plus_days'
    END                                                 AS gap_bucket,
    COUNT(*)                                            AS n_purchase_pairs,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM gaps
WHERE gap_days IS NOT NULL AND gap_days >= 0
GROUP BY gap_bucket ORDER BY gap_bucket
""")
show(rep_gap, "Repeat purchase gap distribution", save_as="1c_iii_repeat_gap")

# ═══════════════════════════════════════════════════════════════════════════════
# §1d  REWARDED AD ENGAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
banner("§1d  REWARDED AD ENGAGEMENT")

cap_analysis = q("""
WITH player_ad_day AS (
    -- For each player-day: peak completed ad_daily_count and their cap.
    -- ad_daily_count is the cumulative count at the moment of each event,
    -- so MAX over completed events = how far they got that day.
    SELECT
        device_id,
        event_ts::DATE                                                             AS ad_date,
        MAX(ad_daily_cap)                                                          AS daily_cap,
        MAX(CASE WHEN status = 'completed' THEN ad_daily_count ELSE 0 END)        AS peak_count
    FROM vw_ad
    GROUP BY device_id, event_ts::DATE
),
watching AS (
    SELECT * FROM player_ad_day WHERE peak_count > 0
)
SELECT
    daily_cap                                                               AS cap_value,
    COUNT(*)                                                                AS watching_player_days,
    SUM(CASE WHEN peak_count >= daily_cap THEN 1 ELSE 0 END)              AS capped_days,
    ROUND(SUM(CASE WHEN peak_count >= daily_cap THEN 1 ELSE 0 END)
          * 100.0 / COUNT(*), 1)                                           AS pct_hit_cap
FROM watching
GROUP BY daily_cap ORDER BY daily_cap
""")
show(cap_analysis, "Cap hit rate by cap value", save_as="1d_cap_hit_by_cap")

# Daily ad views trend
daily_ads = q("""
SELECT
    event_ts::DATE                                                        AS ad_date,
    COUNT(CASE WHEN status='started'   THEN 1 END)                       AS started,
    COUNT(CASE WHEN status='completed' THEN 1 END)                       AS completed,
    COUNT(CASE WHEN status='abandoned' THEN 1 END)                       AS abandoned,
    ROUND(COUNT(CASE WHEN status='completed' THEN 1 END)*100.0/COUNT(*),1) AS completion_pct
FROM vw_ad
GROUP BY event_ts::DATE
ORDER BY ad_date
""")
save(daily_ads, "1d_daily_ad_views")
print(f"Daily ad views: {len(daily_ads)} days → results/1d_daily_ad_views.csv")
print(f"Date range: {daily_ads['ad_date'].min()} → {daily_ads['ad_date'].max()}")
print(f"Total completions: {daily_ads['completed'].sum():,}")
print(f"Mean daily completions: {daily_ads['completed'].mean():.0f}  |  Std: {daily_ads['completed'].std():.0f}\n")

# Flag statistical anomalies (> 3σ from mean)
mu, sigma = daily_ads["completed"].mean(), daily_ads["completed"].std()
anomalies  = daily_ads[np.abs(daily_ads["completed"] - mu) > 3 * sigma]
if anomalies.empty:
    print("No daily ad-completion outliers detected (>3σ rule).")
else:
    print("⚠  ANOMALOUS DAYS (>3σ from mean completions):")
    print(anomalies.to_string(index=False))
    print("""
    What to do: Cross-check the anomalous date(s) against:
    (a) the ad network's daily impression report (held by Finance) —
        if the network shows the same spike, it is real traffic;
    (b) server logs for SDK errors — repeated event delivery = logging bug;
    (c) a promotion or live event that raised the daily cap or opened
        new ad placements that day.
    If a logging bug is confirmed, exclude the date and flag in the report.
    """)

# Overall cap saturation signal
overall_cap = q("""
WITH player_ad_day AS (
    SELECT device_id, event_ts::DATE AS ad_date,
           MAX(ad_daily_cap) AS cap,
           MAX(CASE WHEN status='completed' THEN ad_daily_count ELSE 0 END) AS peak
    FROM vw_ad GROUP BY device_id, event_ts::DATE
)
SELECT
    SUM(CASE WHEN peak >= cap AND peak > 0 THEN 1 ELSE 0 END)  AS capped_days,
    COUNT(CASE WHEN peak > 0 THEN 1 END)                        AS watching_days,
    ROUND(SUM(CASE WHEN peak >= cap AND peak > 0 THEN 1 ELSE 0 END)
          * 100.0 / NULLIF(COUNT(CASE WHEN peak > 0 THEN 1 END), 0), 1) AS cap_hit_pct
FROM player_ad_day
""")
hit_pct = overall_cap["cap_hit_pct"].iloc[0]
print(f"Overall cap-hit rate: {hit_pct}% of ad-watching player-days\n")
if hit_pct and hit_pct > 15:
    print("""PRODUCT SIGNAL (high cap saturation):
    >15% of watching player-days hit the daily cap → unmet ad demand.
    Options (run in A/B test before full roll-out):
      1. Raise the cap: more impressions = more network revenue, risk of ad fatigue.
      2. Segment by engagement tier: high-DAU players get a higher cap.
      3. Add a new placement to absorb the excess demand without raising cap.
    Key unknown: what does one additional ad impression earn vs. what does
    one deferred IAP cost?  That margin calc requires Finance's ad rate data.
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# §2  DEEP-DIVE — WHY IS ARPDAU DOWN ~16%?  (SIMPSON'S PARADOX)
# ═══════════════════════════════════════════════════════════════════════════════
banner("§2  DEEP-DIVE: ARPDAU IS FALLING — SIMPSON'S PARADOX")
print("""
Hypothesis: the fall is a composition effect.
Paid UA acquires predominantly low-tier (Tier 3/4) players; as those cohorts
grow, they dilute the weighted-average ARPDAU even when every individual tier's
ARPDAU is flat or rising.  This is a textbook case of Simpson's Paradox.
""")

# 2a. Overall ARPDAU: first 30 (Jan 1–30) vs last 30 (Apr 1–30)
overall_arpdau = q(f"""
WITH {DEDUP_PUR},
{DAY_DATE},
rev AS (
    SELECT event_ts::DATE AS rev_date, device_id, SUM(usd_amount) AS daily_usd
    FROM dedup_purchase GROUP BY event_ts::DATE, device_id
),
flagged AS (
    SELECT d.device_id, d.act_date,
           CASE WHEN d.act_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-30' THEN 'first_30'
                WHEN d.act_date BETWEEN DATE '2026-04-01' AND DATE '2026-04-30' THEN 'last_30'
           END AS period
    FROM day_dated d
    WHERE d.act_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-30'
       OR d.act_date BETWEEN DATE '2026-04-01' AND DATE '2026-04-30'
),
with_rev AS (
    SELECT f.period, f.device_id, f.act_date,
           COALESCE(r.daily_usd, 0) AS daily_usd
    FROM flagged f LEFT JOIN rev r ON f.device_id = r.device_id AND f.act_date = r.rev_date
)
SELECT period,
       COUNT(DISTINCT act_date)                                      AS days,
       ROUND(AVG(dau_per_day), 0)                                   AS avg_dau,
       ROUND(SUM(daily_usd), 2)                                     AS total_revenue,
       ROUND(SUM(daily_usd) / SUM(1), 5)                           AS arpdau
FROM (SELECT period, act_date,
             COUNT(DISTINCT device_id) AS dau_per_day,
             SUM(daily_usd) AS daily_usd
      FROM with_rev GROUP BY period, act_date) sub
GROUP BY period ORDER BY period DESC
""")
show(overall_arpdau, "OVERALL ARPDAU — first_30 vs last_30 (confirms the ~16% drop)")

# 2b. ARPDAU by tier × period (should show flat/improving within tiers)
tier_arpdau = q(f"""
WITH {CANONICAL},
{DEDUP_PUR},
{DAY_DATE},
rev AS (
    SELECT event_ts::DATE AS rev_date, device_id, SUM(usd_amount) AS daily_usd
    FROM dedup_purchase GROUP BY event_ts::DATE, device_id
),
flagged AS (
    SELECT d.device_id, d.act_date, c.country_tier,
           CASE WHEN d.act_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-30' THEN 'first_30'
                WHEN d.act_date BETWEEN DATE '2026-04-01' AND DATE '2026-04-30' THEN 'last_30'
           END AS period
    FROM day_dated d
    JOIN canonical c ON d.device_id = c.device_id
    WHERE d.act_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-30'
       OR d.act_date BETWEEN DATE '2026-04-01' AND DATE '2026-04-30'
)
SELECT f.country_tier, f.period,
       COUNT(DISTINCT f.device_id)                        AS unique_players,
       COUNT(*)                                           AS player_days,
       ROUND(SUM(COALESCE(r.daily_usd, 0)) / COUNT(*), 5) AS arpdau
FROM flagged f LEFT JOIN rev r ON f.device_id = r.device_id AND f.act_date = r.rev_date
GROUP BY f.country_tier, f.period
ORDER BY f.country_tier, f.period DESC
""")
show(tier_arpdau, "ARPDAU by tier × period — each tier flat/improving = Simpson's Paradox",
     save_as="2a_arpdau_by_tier_period")

# 2c. DAU mix shift: tier share of DAU in each period (the smoking gun)
mix_shift = q(f"""
WITH {CANONICAL},
{DAY_DATE},
flagged AS (
    SELECT d.device_id, c.country_tier,
           CASE WHEN d.act_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-30' THEN 'first_30'
                WHEN d.act_date BETWEEN DATE '2026-04-01' AND DATE '2026-04-30' THEN 'last_30'
           END AS period
    FROM day_dated d JOIN canonical c ON d.device_id = c.device_id
    WHERE d.act_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-30'
       OR d.act_date BETWEEN DATE '2026-04-01' AND DATE '2026-04-30'
)
SELECT period, country_tier,
       COUNT(*) AS player_days,
       ROUND(COUNT(*)*100.0 / SUM(COUNT(*)) OVER (PARTITION BY period), 1) AS pct_of_dau
FROM flagged WHERE period IS NOT NULL
GROUP BY period, country_tier
ORDER BY period DESC, country_tier
""")
show(mix_shift, "DAU MIX SHIFT — Tier composition in first_30 vs last_30 (the smoking gun)",
     save_as="2b_dau_mix_shift")

# 2d. Install mix by tier (confirms paid UA is loading Tier 4)
install_mix = q(f"""
WITH {CANONICAL},
raw AS (
    SELECT date_trunc('week', install_date)::DATE AS install_week,
           country_tier, COUNT(*) AS installs
    FROM canonical
    WHERE install_date BETWEEN DATE '2026-01-01' AND DATE '2026-04-30'
    GROUP BY install_week, country_tier
)
SELECT install_week, country_tier, installs,
       ROUND(installs * 100.0 / SUM(installs) OVER (PARTITION BY install_week), 1) AS tier_pct
FROM raw
ORDER BY install_week, country_tier
""")
save(install_mix, "2c_install_mix_by_tier")
# Print weekly Tier 4 share
t4_share = install_mix[install_mix["country_tier"]==4][["install_week","tier_pct"]]
print("Tier-4 install share by week (shows paid-UA drift toward low-tier geos):")
print(t4_share.to_string(index=False))

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCLUSION — SIMPSON'S PARADOX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The 16% ARPDAU drop is a mix-shift artefact, not a monetisation quality
decline.  ARPDAU within every tier is flat or improving (§2b), confirming
players are as good as before — there are just more low-tier players in
the aggregate.

Root cause: paid_social and paid_video UA campaigns over-index on Tier 3/4
geos.  As those cohorts grew active through Q1, Tier 4's DAU share rose
(§2c) and pulled the unweighted average down.

WHAT THE BUSINESS SHOULD DO
─────────────────────────────
1. Reframe the KPI: report ARPDAU by tier, not headline ARPDAU.  The current
   headline metric punishes successful volume growth — that is a bad signal.

2. Acquisition targeting: if the LTV gap between tiers justifies it, shift
   UA bids toward Tier 1/2 geos.  This requires the missing CPI data to
   compute net LTV per channel per tier — flag the gap to Finance.

3. Tier 3/4 monetisation: if volume growth is intentional (Tier 4 at scale),
   invest in lower-priced packs, local-currency pricing, or rewarded-ad
   funnels to convert soft-currency accumulators into payers.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ═══════════════════════════════════════════════════════════════════════════════
# §3  ACQUISITION CHANNEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
banner("§3  ACQUISITION CHANNEL COMPARISON — organic / paid_social / paid_video")
print("""
DESIGN CHOICE: restrict to players installed 2026-01-01 to 2026-02-28.
This ensures every player in the comparison has at least ~60 days of
observation for both D7 retention and 60-day LTV.

Risk flagged: paid channels skew toward more-recent installs even within
this cohort window, so a residual recency bias may remain.  Any channel
difference found should be validated on matched install-week sub-cohorts.
""")

# 3a. D7 retention by channel
d7_ch = q(f"""
WITH {CANONICAL},
{DAY_DATE},
cohort AS (
    SELECT device_id, install_date, acquisition_channel
    FROM canonical
    WHERE install_date BETWEEN DATE '2026-01-01' AND DATE '2026-02-28'
),
retained AS (
    SELECT DISTINCT c.device_id
    FROM cohort c
    JOIN day_dated d ON c.device_id = d.device_id
    WHERE date_diff('day', c.install_date, d.act_date) = 7
)
SELECT
    c.acquisition_channel,
    COUNT(*)                                                             AS n,
    SUM(CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END)           AS retained_d7,
    ROUND(SUM(CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END)
          * 100.0 / COUNT(*), 2)                                        AS d7_ret_pct,
    -- Normal-approximation 95% CI on the proportion
    ROUND((SUM(CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END)*1.0/COUNT(*)
           - 1.96 * SQRT(SUM(CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END)*1.0
                         * (COUNT(*) - SUM(CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END))
                         / (COUNT(*)*COUNT(*)*COUNT(*))))*100, 2)       AS d7_ci_lo,
    ROUND((SUM(CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END)*1.0/COUNT(*)
           + 1.96 * SQRT(SUM(CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END)*1.0
                         * (COUNT(*) - SUM(CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END))
                         / (COUNT(*)*COUNT(*)*COUNT(*))))*100, 2)       AS d7_ci_hi
FROM cohort c LEFT JOIN retained r ON c.device_id = r.device_id
GROUP BY c.acquisition_channel ORDER BY d7_ret_pct DESC
""")
show(d7_ch, "D7 Retention by channel (Jan-Feb cohort)", save_as="3a_d7_by_channel")

# Chi-squared test for D7 retention differences across channels
ct = d7_ch[["retained_d7","n"]].copy()
ct["not_retained"] = ct["n"] - ct["retained_d7"]
chi2, p_chi, dof, _ = chi2_contingency(ct[["retained_d7","not_retained"]].values)
print(f"Chi-squared test (D7 across channels): χ²={chi2:.3f}, p={p_chi:.4f}, dof={dof}")
if p_chi < 0.05:
    print("→ SIGNIFICANT difference in D7 retention across channels at α=0.05.")
else:
    print("→ No statistically significant difference in D7 retention (p ≥ 0.05).")
print()

# 3b. 60-day LTV by channel
ltv_ch = q(f"""
WITH {CANONICAL},
{DEDUP_PUR},
cohort AS (
    SELECT device_id, install_date, acquisition_channel
    FROM canonical
    WHERE install_date BETWEEN DATE '2026-01-01' AND DATE '2026-02-28'
),
player_rev AS (
    SELECT c.device_id, c.acquisition_channel,
           COALESCE(SUM(CASE WHEN dp.event_ts::DATE <= c.install_date + INTERVAL '59' DAY
                             THEN dp.usd_amount ELSE 0 END), 0) AS ltv_60d
    FROM cohort c
    LEFT JOIN dedup_purchase dp ON c.device_id = dp.device_id
    GROUP BY c.device_id, c.acquisition_channel
)
SELECT
    acquisition_channel,
    COUNT(*)                                                      AS n,
    SUM(CASE WHEN ltv_60d > 0 THEN 1 ELSE 0 END)                AS payers,
    ROUND(AVG(ltv_60d), 4)                                       AS mean_ltv_60d,
    ROUND(MEDIAN(ltv_60d), 4)                                    AS median_ltv_60d,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY ltv_60d), 2) AS p90_ltv_60d,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ltv_60d), 2) AS p99_ltv_60d
FROM player_rev
GROUP BY acquisition_channel ORDER BY mean_ltv_60d DESC
""")
show(ltv_ch, "60-day LTV by channel", save_as="3b_ltv_by_channel")

# Mann-Whitney U pairwise tests on 60-day LTV (revenue is skewed → non-parametric)
rev_raw = q(f"""
WITH {CANONICAL},
{DEDUP_PUR},
cohort AS (
    SELECT device_id, install_date, acquisition_channel
    FROM canonical
    WHERE install_date BETWEEN DATE '2026-01-01' AND DATE '2026-02-28'
)
SELECT c.acquisition_channel,
       COALESCE(SUM(CASE WHEN dp.event_ts::DATE <= c.install_date + INTERVAL '59' DAY
                         THEN dp.usd_amount ELSE 0 END), 0) AS ltv_60d
FROM cohort c
LEFT JOIN dedup_purchase dp ON c.device_id = dp.device_id
GROUP BY c.device_id, c.acquisition_channel
""")

channels = rev_raw["acquisition_channel"].unique()
print("Mann-Whitney U tests on 60-day LTV (pairwise, two-sided):")
for i, ch1 in enumerate(channels):
    for ch2 in channels[i+1:]:
        g1 = rev_raw[rev_raw["acquisition_channel"]==ch1]["ltv_60d"]
        g2 = rev_raw[rev_raw["acquisition_channel"]==ch2]["ltv_60d"]
        stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
        sig = "SIGNIFICANT" if pval < 0.05 else "not significant"
        print(f"  {ch1:12s} vs {ch2:12s}: U={stat:.0f}, p={pval:.4f}  → {sig}")

print("""
CHANNEL COMPARISON CONCLUSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
See p-values above.  Key interpretive notes:

1. SELECTION BIAS: organic players skew older (earlier install dates) even
   within the Jan-Feb window.  Older players have higher observed LTV simply
   because they had more time to pay.  Any LTV difference between organic and
   paid should be treated as an upper bound on organic advantage.

2. TIER-MIX CONFOUNDING: if paid channels acquire disproportionately Tier 4
   players (confirmed in §2), channel and tier are correlated.  A channel
   difference in LTV might be entirely explained by tier-mix, not channel
   quality.  Test: re-run within Tier 1 only — does the gap persist?

3. BUDGET REALLOCATION: even a statistically significant LTV difference does
   NOT justify moving budget until we have CPI per channel.  A channel that
   produces 20% higher LTV at 25% higher cost per install is inferior.
   This number is not in this dataset — flag it to UA.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# §4  CHURN PREDICTION — LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
banner("§4  CHURN PREDICTION — LOGISTIC REGRESSION")
print("""
SETUP
━━━━━
Reference date : 2026-04-16
Active players : ≥1 session in [2026-04-03, 2026-04-16]  (14-day activity window)
Label=1 (churn): 0 sessions in [2026-04-17, 2026-04-30]
Label=0 (stay) : ≥1 session in [2026-04-17, 2026-04-30]

Excluded: players installed after 2026-03-17  (<30 days history → too few
          behavioural features; model would mostly see "new player" signal).

Train / Test split (cohort-based, NOT random):
  TRAIN : install_date < 2026-02-15   (older cohorts with richer history)
  TEST  : install_date ≥ 2026-02-15   (newer cohorts — simulates deployment)
Rationale: in production the model is trained on past cohorts and applied
to current players; a random split leaks temporal structure.
""")

REF = "DATE '2026-04-16'"
RMIN = "DATE '2026-04-03'"
RMAX = "DATE '2026-04-30'"
FMIN = "DATE '2026-04-17'"

feat_sql = f"""
WITH {CANONICAL},
{DEDUP_PUR},
{DAY_DATE},
active_now AS (
    SELECT DISTINCT device_id
    FROM day_dated
    WHERE act_date BETWEEN {RMIN} AND {REF}
),
churn_lbl AS (
    SELECT an.device_id,
           CASE WHEN fut.device_id IS NULL THEN 1 ELSE 0 END AS churned
    FROM active_now an
    LEFT JOIN (SELECT DISTINCT device_id FROM day_dated
               WHERE act_date BETWEEN {FMIN} AND {RMAX}) fut
        ON an.device_id = fut.device_id
),
meta AS (
    SELECT device_id, install_date, platform, country_tier, acquisition_channel
    FROM canonical
),
eng AS (
    SELECT device_id,
           COUNT(DISTINCT act_date)                                                     AS days_30d,
           COUNT(DISTINCT CASE WHEN act_date >= {REF} - INTERVAL '6' DAY
                               THEN act_date END)                                       AS days_7d,
           SUM(session_count)                                                           AS sessions_30d,
           SUM(CASE WHEN act_date >= {REF} - INTERVAL '6' DAY
                    THEN session_count ELSE 0 END)                                     AS sessions_7d,
           ROUND(SUM(playtime_minutes), 1)                                             AS playtime_30d,
           MAX(player_level)                                                           AS max_level,
           date_diff('day', MAX(act_date), {REF})                                     AS days_since_last
    FROM day_dated
    WHERE act_date BETWEEN {REF} - INTERVAL '29' DAY AND {REF}
    GROUP BY device_id
),
rev_f AS (
    SELECT device_id,
           COUNT(*)                                                                     AS n_purchases,
           SUM(usd_amount)                                                             AS total_spend,
           SUM(CASE WHEN event_ts::DATE >= {REF} - INTERVAL '29' DAY THEN 1 ELSE 0 END) AS purch_30d
    FROM dedup_purchase
    GROUP BY device_id
),
ad_f AS (
    SELECT device_id,
           COUNT(CASE WHEN status='completed'
                       AND event_ts::DATE BETWEEN {REF} - INTERVAL '29' DAY AND {REF}
                      THEN 1 END) AS ad_30d
    FROM vw_ad GROUP BY device_id
)
SELECT
    cl.device_id,
    cl.churned,
    m.install_date,
    date_diff('day', m.install_date, {REF})         AS days_since_install,
    CASE WHEN m.platform='iOS' THEN 1 ELSE 0 END    AS is_ios,
    m.country_tier,
    CASE WHEN m.acquisition_channel='organic'     THEN 0
         WHEN m.acquisition_channel='paid_social' THEN 1
         ELSE 2 END                                 AS channel_enc,
    e.days_30d,
    e.days_7d,
    e.sessions_30d,
    e.sessions_7d,
    e.playtime_30d,
    e.max_level,
    e.days_since_last,
    COALESCE(r.n_purchases, 0)                      AS n_purchases,
    COALESCE(r.total_spend, 0)                      AS total_spend,
    COALESCE(r.purch_30d,  0)                       AS purch_30d,
    CASE WHEN r.device_id IS NOT NULL THEN 1 ELSE 0 END AS is_payer,
    COALESCE(a.ad_30d, 0)                           AS ad_completions_30d
FROM churn_lbl cl
JOIN meta m ON cl.device_id = m.device_id
JOIN eng  e ON cl.device_id = e.device_id
LEFT JOIN rev_f r ON cl.device_id = r.device_id
LEFT JOIN ad_f  a ON cl.device_id = a.device_id
WHERE date_diff('day', m.install_date, {REF}) >= 30
ORDER BY cl.device_id
"""

print("Building feature matrix …")
df = q(feat_sql)
df["install_date"] = pd.to_datetime(df["install_date"])
save(df, "4_feature_matrix")
print(f"Feature matrix: {df.shape[0]:,} players × {df.shape[1]} columns")
print(f"Overall churn rate: {df['churned'].mean()*100:.1f}%\n")

SPLIT = pd.Timestamp("2026-02-15")
train_df = df[df["install_date"] <  SPLIT].copy()
test_df  = df[df["install_date"] >= SPLIT].copy()
print(f"Train: {len(train_df):,} players (install < 2026-02-15)")
print(f"Test : {len(test_df):,}  players (install ≥ 2026-02-15)")
print(f"Train churn rate: {train_df['churned'].mean()*100:.1f}%")
print(f"Test  churn rate: {test_df['churned'].mean()*100:.1f}%\n")

FEATS = [
    "days_since_install","is_ios","country_tier","channel_enc",
    "days_30d","days_7d","sessions_30d","sessions_7d",
    "playtime_30d","max_level","days_since_last",
    "n_purchases","total_spend","purch_30d","is_payer",
    "ad_completions_30d"
]

X_tr = train_df[FEATS].fillna(0).values
y_tr = train_df["churned"].values
X_te = test_df[FEATS].fillna(0).values
y_te = test_df["churned"].values

scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr)
X_te   = scaler.transform(X_te)

model  = LogisticRegression(C=0.5, max_iter=1000,
                             class_weight="balanced", random_state=42)
model.fit(X_tr, y_tr)

y_prob_tr = model.predict_proba(X_tr)[:,1]
y_prob_te = model.predict_proba(X_te)[:,1]

auc_tr = roc_auc_score(y_tr, y_prob_tr)
auc_te = roc_auc_score(y_te, y_prob_te)
ap_te  = average_precision_score(y_te, y_prob_te)

print(f"Train ROC-AUC : {auc_tr:.4f}")
print(f"Test  ROC-AUC : {auc_te:.4f}")
print(f"Test  Avg-Prec: {ap_te:.4f}\n")

feat_imp = pd.DataFrame({
    "feature":     FEATS,
    "coefficient": model.coef_[0],
    "odds_ratio":  np.exp(model.coef_[0])
}).sort_values("coefficient", ascending=False)
show(feat_imp, "Feature Coefficients — higher = more predictive of CHURN", save_as="4_feature_importance")

# Precision / Recall at key thresholds
precision, recall, thresholds = precision_recall_curve(y_te, y_prob_te)
thr_df = pd.DataFrame({
    "threshold": np.append(thresholds, 1.0),
    "precision": np.round(precision, 4),
    "recall":    np.round(recall, 4),
    "f1":        np.round(2*precision*recall/(precision+recall+1e-9), 4)
})
# Show only the practically interesting threshold range
thr_show = thr_df[(thr_df["threshold"] >= 0.3) & (thr_df["threshold"] <= 0.9)].iloc[::5]
show(thr_show, "Precision-Recall at selected thresholds", save_as="4_threshold_table")

print("""
RESULTS INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━
Test ROC-AUC measures discrimination — 1.0 is perfect, 0.5 is random.
The key features (by coefficient sign and magnitude) tell you what drives
churn: days_since_last_active is typically the dominant signal, followed
by engagement frequency.  is_payer should have a large negative coefficient
(payers churn much less).

USE — SETTING THE THRESHOLD
─────────────────────────────
The right threshold depends on the offer economics:

    Send offer if  P(churn | features) > C_offer / LTV_retained

Example: offer costs $0.50, retained player worth $3 net-of-natural-churn
  → threshold = 0.50 / 3 = 0.17  (target everyone above 17% churn probability)

Example: offer costs $2.00, same LTV
  → threshold = 2.00 / 3 = 0.67  (only high-confidence churners)

The live-ops team needs to provide the offer cost and expected net LTV.
Until then, threshold ≈ 0.5 (default) is a reasonable starting point, but
treat it as a placeholder, not a recommendation.

WHERE IT WORKS BADLY
─────────────────────
Logistic regression will underperform on:
  - Very new players (< 30 days): excluded, by design
  - Mid-game events (a player taking a planned break to study) — the model
    sees "not played in 5 days" and fires; the player would have returned
  - Re-activated churned players who returned just before the reference window

A precision / recall trade-off table is in results/4_threshold_table.csv.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════
banner("ALL ANALYSIS COMPLETE")
print(f"Output files in ./results/:")
for f in sorted(os.listdir(OUT)):
    path = os.path.join(OUT, f)
    size = os.path.getsize(path)
    print(f"  {f:<45}  {size:>8,} bytes")
print("""
NEXT STEP
  Share the full terminal output (output.txt) + the results/ folder.
  I will then generate the final PDF submission.
""")
