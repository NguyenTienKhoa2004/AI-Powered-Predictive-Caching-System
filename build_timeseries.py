"""
build_timeseries.py  (RAM-efficient version)
=============================================
Gop 72 file *_clean.csv thanh 1 bang time series:

    title                    | 2026-01-01 00:00 | 2026-01-01 01:00 | ...
    Albert_Einstein          |      500         |      423         | ...
    Python_(programming_...) |      200         |      180         | ...

Chay 2 luot (pass) de khong bao gio load qua 1 file vao RAM cung luc:
  Pass 1: Dem tong views moi bai viet -> chon top-N pho bien nhat
  Pass 2: Doc lai tung file, chi giu top-N -> ghi vao pivot matrix

Output:
  processed/timeseries_matrix.csv  -- wide format (title x 72 gio)
  processed/timeseries_long.csv    -- long format (title, timestamp, views)
"""

import gc
import heapq
import pandas as pd
import re
import sys
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================
PROCESSED_DIR = Path(r"C:\Users\as\OneDrive\Desktop\Thesis\processed")
OUTPUT_DIR    = Path(r"C:\Users\as\OneDrive\Desktop\Thesis\processed")

TOP_N    = 5_000   # Giu lai bao nhieu bai viet pho bien nhat
FNAME_RE = re.compile(r'pageviews-(\d{8})-(\d{6})_clean\.csv')


# =============================================================================
# HELPERS
# =============================================================================
def fname_to_timestamp(fp: Path) -> pd.Timestamp:
    """Chuyen ten file -> Timestamp. VD: pageviews-20260101-000000 -> 2026-01-01 00:00"""
    m = FNAME_RE.match(fp.name)
    d, t = m.group(1), m.group(2)
    return pd.Timestamp(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:]),
                        hour=int(t[:2]), minute=int(t[2:4]))


def get_files(processed_dir: Path) -> list[Path]:
    files = sorted(f for f in processed_dir.glob("pageviews-*_clean.csv")
                   if FNAME_RE.match(f.name))
    if not files:
        sys.exit(f"ERROR: No files found in {processed_dir}")
    return files


# =============================================================================
# PASS 1: Tinh tong views cua moi bai viet (doc tung file, chi dung dict)
# =============================================================================
def pass1_count_totals(files: list[Path]) -> dict:
    """
    Doc tung file, cong don views vao dict {title: total_views}.
    RAM su dung: chi 1 CSV (filtered) + dict ~200 MB.
    """
    print(f"Pass 1/{len(files)}: counting total views per article...")
    total_views: dict = defaultdict(int)

    for i, fp in enumerate(files, 1):
        df = pd.read_csv(fp, usecols=['title', 'views'],
                         dtype={'title': str, 'views': 'int32'})
        for title, views in zip(df['title'], df['views']):
            total_views[title] += int(views)

        if i % 12 == 0 or i == len(files):
            print(f"  [{i:>2}/{len(files)}] done")

    return total_views


# =============================================================================
# PASS 2: Build pivot matrix (chi giu top-N)
# =============================================================================
def pass2_build_pivot(files: list[Path], top_titles: set) -> pd.DataFrame:
    """
    Doc lai tung file, chi giu hang co title trong top_titles.
    Tra ve DataFrame dang long: [title, timestamp, views].
    RAM su dung: 1 file filtered + long dataframe (top_N x n_hours ~ nho).
    """
    print(f"\nPass 2/{len(files)}: building pivot for top-{len(top_titles)} articles...")
    records = []

    for i, fp in enumerate(files, 1):
        ts = fname_to_timestamp(fp)
        df = pd.read_csv(fp, usecols=['title', 'views'],
                         dtype={'title': str, 'views': 'int32'})
        df = df[df['title'].isin(top_titles)]
        df['timestamp'] = ts
        records.append(df[['title', 'timestamp', 'views']])

        if i % 12 == 0 or i == len(files):
            print(f"  [{i:>2}/{len(files)}] done")

    df_long = pd.concat(records, ignore_index=True)
    return df_long


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = get_files(PROCESSED_DIR)
    print(f"Found {len(files)} hourly files.\n")

    # ── Pass 1 ────────────────────────────────────────────────────────────────
    total_views = pass1_count_totals(files)
    print(f"\nUnique articles found: {len(total_views):,}")

    # Chon top-N bai viet bang heapq (khong can chuyen sang pd.Series)
    top_items  = heapq.nlargest(TOP_N, total_views.items(), key=lambda x: x[1])
    top_titles = {title for title, _ in top_items}
    min_views  = top_items[-1][1]
    print(f"Top-{TOP_N} cutoff: min total views = {min_views:,}")

    # Giai phong RAM truoc Pass 2
    del total_views, top_items
    gc.collect()

    # ── Pass 2 ────────────────────────────────────────────────────────────────
    df_long = pass2_build_pivot(files, top_titles)

    # ── Pivot sang wide format ─────────────────────────────────────────────────
    print("\nPivoting to wide format...")
    df_wide = df_long.pivot_table(
        index='title', columns='timestamp',
        values='views', aggfunc='sum', fill_value=0
    ).sort_index(axis=1)
    print(f"Matrix shape: {df_wide.shape[0]:,} articles x {df_wide.shape[1]} hours")

    # ── Luu file ──────────────────────────────────────────────────────────────
    out_wide = OUTPUT_DIR / "timeseries_matrix.csv"
    df_wide.to_csv(out_wide)
    print(f"\n[SAVED] Wide matrix -> {out_wide}  ({out_wide.stat().st_size/1e6:.1f} MB)")

    out_long = OUTPUT_DIR / "timeseries_long.csv"
    df_long.sort_values(['title', 'timestamp']).to_csv(out_long, index=False)
    print(f"[SAVED] Long format -> {out_long}  ({out_long.stat().st_size/1e6:.1f} MB)")

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n--- Preview (top 5 articles x first 5 hours) ---")
    preview = df_wide.iloc[:5, :5].copy()
    preview.columns = [c.strftime('%m-%d %H:%M') for c in preview.columns]
    print(preview.to_string())
    print("\nDone!")


if __name__ == "__main__":
    main()
