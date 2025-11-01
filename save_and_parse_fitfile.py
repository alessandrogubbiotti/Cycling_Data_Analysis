#!/usr/bin/env python3
"""
Zwift Organizer: Athlete-rooted workflow

- Interactive script for organizing Zwift .fit files
- Keeps each athlete's data separate
- Copies .fit to AllData/
- Parses metrics to ParsedData/ride_name/
- Creates symlinks in by_month and by_training
"""

import os
from pathlib import Path
import shutil
import datetime
import json
from fitparse import FitFile
import csv

# ======= CONFIG =======
ATHLETES = ["Diego", "Alessandro", "Alberto"]
ROOT_FOLDER = Path("Athlete").expanduser()
USE_SYMLINKS = True  # True -> create symlinks
TRAINING_TYPES = ["Z2", "Z3", "Z4", "Climb", "Sprints"]
# =====================

def ask_athlete():
    while True:
        name = input(f"Enter athlete name ({', '.join(ATHLETES)}): ").strip()
        if name in ATHLETES:
            return name
        print("Invalid athlete name. Try again.")


def ask_file_path():
    while True:
        path_str = input("In my mac, the Zwift files are found in Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Zwift/Activities/YYYY-MM-DD-HH-MM-SS.fit format \n\n You do not need to remove the backslash as this will be removed authomatically by the program \n\n Enter path to .fit file to organize: ").strip()
        # Automatically fix shell-style escaped spaces
        path_str = path_str.replace(r"\ ", " ")
        path = Path(path_str).expanduser()
        if path.is_file() and path.suffix.lower() == ".fit":
            return path
        print("Invalid file path or not a .fit file. Try again.")


def ask_training_type():
    while True:
        t = input(f"Enter training type ({', '.join(TRAINING_TYPES)}): ").strip()
        if t in TRAINING_TYPES:
            return t
        print("Invalid training type. Try again.")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    if USE_SYMLINKS:
        try:
            dst.symlink_to(src.resolve())
        except Exception as e:
            print(f"⚠️ Could not create symlink: {e}")
    else:
        shutil.copy2(src, dst)
def parse_fit_metrics(fit_path: Path):
    """
    Extract session-level metrics and time-dependent record messages from a .fit file.
    Returns:
        metrics: dict with session summary
        records: list of dicts with per-record data (time-dependent)
    """
    metrics = {}
    records = []

    try:
        fit = FitFile(str(fit_path))

        # --- Session summary ---
        for msg in fit.get_messages("session"):
            for field in msg:
                metrics[field.name] = field.value
        metrics['filename'] = fit_path.name

        # --- Record-level (time-dependent) ---
        for record in fit.get_messages("record"):
            r = {}
            for field in record:
                r[field.name] = field.value
            records.append(r)

    except Exception as e:
        print(f"⚠️ Could not parse {fit_path.name}: {e}")

    return metrics, records

def get_fit_date(fit_path: Path):
    metrics = parse_fit_metrics(fit_path)
    start_time = metrics.get("start_time")
    if start_time is None:
        start_time = datetime.datetime.fromtimestamp(fit_path.stat().st_mtime)
    return start_time

def save_records_csv(records, save_path):
    """
    Save time-dependent record data to a CSV file.
    records: list of dicts returned by parse_fit_metrics
    save_path: Path to CSV file
    """
    if not records:
        print("⚠️ No record data to save.")
        return

    keys = records[0].keys()  # assume all records have same keys
    with open(save_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ Record data saved to {save_path}")

def main():
    print("=== Zwift Organizer Interactive (Athlete-rooted) ===")

    athlete = ask_athlete()
    athlete_folder = ROOT_FOLDER / athlete

    # Define main folders per athlete
    all_data_folder = athlete_folder / "AllData"
    parsed_data_root = athlete_folder / "ParsedData"
    by_month_root = athlete_folder / "by_month"
    by_training_root = athlete_folder / "by_training"

    # Ensure main folders exist
    ensure_dir(all_data_folder)
    ensure_dir(parsed_data_root)
    ensure_dir(by_month_root)
    ensure_dir(by_training_root)

    # Ask for .fit file and training type
    fit_file = ask_file_path()
    training_type = ask_training_type()

    # Copy .fit to AllData
    dest_fit = all_data_folder / fit_file.name
    shutil.copy2(fit_file, dest_fit)
    print(f"✅ File copied to {dest_fit}")

    # Parse .fit and save metrics JSON/CSV
    ride_name = dest_fit.stem
    parsed_folder = parsed_data_root / ride_name
    ensure_dir(parsed_folder)

    metrics, records  = parse_fit_metrics(dest_fit)
    # Save JSON
    with open(parsed_folder / f"{ride_name}.json", "w") as f:
        json.dump(metrics, f, default=str, indent=2)

    # Save CSV (one row with ride metrics)
    csv_path = parsed_folder / f"{ride_name}.csv"
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
# Save record-level (time-dependent) CSV
    records_csv_path = parsed_folder / f"{dest_fit.stem}_records.csv"
    save_records_csv(records, records_csv_path)
    print(f"✅ Parsed data saved to {parsed_folder}")

    # Get ride date for by_month
    start_time = get_fit_date(dest_fit)
    month_str = start_time.strftime("%Y-%m")

    # Create symlinks in by_month
    month_folder = by_month_root / month_str
    ensure_dir(month_folder)
    link_or_copy(dest_fit, month_folder / dest_fit.name)
    link_or_copy(parsed_folder, month_folder / f"{ride_name}_parsed")

    # Create symlinks in by_training
    training_folder = by_training_root / training_type
    ensure_dir(training_folder)
    link_or_copy(dest_fit, training_folder / dest_fit.name)
    link_or_copy(parsed_folder, training_folder / f"{ride_name}_parsed")

    print("✅ Symlinks created in by_month and by_training")
    print(f"By month folder: {month_folder}")
    print(f"By training folder: {training_folder}")
    print("Done.")

if __name__ == "__main__":
    main()

