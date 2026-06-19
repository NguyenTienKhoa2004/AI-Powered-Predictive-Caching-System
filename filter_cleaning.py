import pandas as pd
import re
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union

NOISE_PATTERN = r'^(Special|File|User|Category|Wikipedia|Portal|Talk|Draft):'

INPUT_DIR  = Path(r"C:\Users\as\OneDrive\Desktop\Thesis\wiki_pageviews_raw")
OUTPUT_DIR = Path(r"C:\Users\as\OneDrive\Desktop\Thesis\processed")


def clean_single_hourly_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read one hourly pageviews .gz file, keep only English Wikipedia (en + en.m),
    remove system/noise pages, merge desktop+mobile counts, and return a clean
    DataFrame with columns: [title, views].
    """
    file_path = Path(file_path)

    df = pd.read_csv(
        file_path, sep=' ', header=None,
        usecols=[0, 1, 2], names=['domain', 'title', 'views'],
        compression='gzip',
    )

    df = df[df['domain'].isin(['en', 'en.m'])]

    df = df.dropna(subset=['title'])

    df = df[~df['title'].str.contains(NOISE_PATTERN, flags=re.IGNORECASE, regex=True, na=False)]
    df = df[~df['title'].isin(['Main_Page', 'Main_Page_-_Wikipedia'])]

    df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0).astype(int)

    df_clean = df.groupby('title')['views'].sum().reset_index()

    return df_clean


def process_one(gz_path: Path) -> tuple[str, int, bool]:
    """Worker: clean one file and write its output CSV. Returns (stem, row_count)."""
    stem       = gz_path.stem                          
    out_path   = OUTPUT_DIR / f"{stem}_clean.csv"

    if out_path.exists():
        rows = sum(1 for _ in open(out_path, encoding='utf-8')) - 1  
        return (stem, rows, True)                      

    df_clean = clean_single_hourly_file(gz_path)
    df_clean.to_csv(out_path, index=False)
    return (stem, len(df_clean), False)


def batch_process(max_workers: int = 4):
    """Process all .gz files in INPUT_DIR in parallel."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gz_files = sorted(INPUT_DIR.glob("pageviews-*.gz"))
    total    = len(gz_files)

    if total == 0:
        print(f"No .gz files found in {INPUT_DIR}")
        return

    print(f"Found {total} hourly files  ->  output: {OUTPUT_DIR}")
    print(f"Using {max_workers} parallel workers\n")
    print(f"{'#':>4}  {'File':<35}  {'Rows':>10}  Status")
    print("-" * 65)

    completed = 0
    skipped   = 0
    errors    = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_one, f): f for f in gz_files}

        for idx, future in enumerate(as_completed(futures), start=1):
            gz_path = futures[future]
            try:
                stem, rows, was_skipped = future.result()
                flag = "SKIP" if was_skipped else "OK"
                print(f"{idx:>4}  {stem:<35}  {rows:>10,}  {flag}")
                completed += 1
                if was_skipped:
                    skipped += 1
            except Exception as exc:
                name = gz_path.name
                print(f"{idx:>4}  {name:<35}  {'ERROR':>10}  {exc}")
                errors.append((name, str(exc)))

    print("-" * 65)
    print(f"\nDone!  {completed} files processed  ({skipped} skipped / already done)")
    if errors:
        print(f"\n  {len(errors)} error(s):")
        for name, msg in errors:
            print(f"    {name}: {msg}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_process(max_workers=4)