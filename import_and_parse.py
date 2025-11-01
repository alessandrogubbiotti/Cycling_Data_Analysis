import json
import os
import shutil
import re
from pathlib import Path
from datetime import datetime
from fitparse import FitFile

# === Root directory ===
ROOT = Path(__file__).resolve().parent

# === Detect athletes (folders in Athlete directory) ===
ATHLETE_BASE_DIR = ROOT / "Athlete"
if not ATHLETE_BASE_DIR.exists():
    print(f"❌ Athlete directory not found at: {ATHLETE_BASE_DIR}")
    exit(1)

ATHLETES = [d.name for d in ATHLETE_BASE_DIR.iterdir() if d.is_dir() and d.name not in ("Trainings", ".git")]

if not ATHLETES:
    print("❌ No athlete folders found in the Athlete directory.")
    print(f"Please create athlete folders in: {ATHLETE_BASE_DIR}")
    exit(1)

# === Paths ===
TRAININGS_DIR = ROOT / "Trainings"

def choose(prompt, options):
    """Helper function for clean multiple-choice input"""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  {i + 1}) {opt}")
    while True:
        try:
            choice = int(input("Choose: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print("❌ Invalid choice, try again.")


def parse_date_from_filename(filename):
    """Extract date from filename in format YYYY-MM-DD-HH-MM-SS or similar"""
    # Try common FIT file date patterns
    patterns = [
        r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD-HH-MM-SS
        r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})',        # YYYY-MM-DD-HH-MM
        r'(\d{4}-\d{2}-\d{2})',                    # YYYY-MM-DD
        r'(\d{8}_\d{6})',                          # YYYYMMDD_HHMMSS
        r'(\d{8}-\d{6})',                          # YYYYMMDD-HHMMSS
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            date_str = match.group(1)
            try:
                if len(date_str) == 19:  # YYYY-MM-DD-HH-MM-SS
                    return datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S")
                elif len(date_str) == 16:  # YYYY-MM-DD-HH-MM
                    return datetime.strptime(date_str, "%Y-%m-%d-%H-%M")
                elif len(date_str) == 10:  # YYYY-MM-DD
                    return datetime.strptime(date_str, "%Y-%m-%d")
                elif len(date_str) == 15 and '_' in date_str:  # YYYYMMDD_HHMMSS
                    return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                elif len(date_str) == 15 and '-' in date_str:  # YYYYMMDD-HHMMSS
                    return datetime.strptime(date_str, "%Y%m%d-%H%M%S")
            except ValueError:
                continue
    return None


def get_date_from_user(fit_filename):
    """Get date from user input when cannot parse from filename"""
    print(f"❌ Could not parse date from filename: {fit_filename}")
    print("Please enter the date of the training:")
    
    while True:
        year = input("Year (YYYY): ").strip()
        month = input("Month (MM): ").strip()
        day = input("Day (DD): ").strip()
        time_input = input("Time (HH:MM, or leave blank): ").strip()
        
        try:
            date_str = f"{year}-{month}-{day}"
            if time_input:
                ride_date = datetime.strptime(f"{date_str} {time_input}", "%Y-%m-%d %H:%M")
            else:
                ride_date = datetime.strptime(date_str, "%Y-%m-%d")
            return ride_date
        except ValueError as e:
            print(f"❌ Invalid date format: {e}")
            print("Please try again with valid numbers.")
            use_current = input("Use current date and time instead? (y/N): ").strip().lower()
            if use_current == 'y':
                return datetime.now()


def get_unique_ride_name(athlete_dir, base_date, training_type):
    """Generate a unique ride name, adding suffix if same date training exists"""
    base_name = f"{base_date.strftime('%Y-%m-%d')}"
    
    # Check if this ride name already exists
    all_data_dir = athlete_dir / "AllData"
    existing_rides = [f.stem for f in all_data_dir.glob("*.fit")]
    
    if base_name not in existing_rides:
        return base_name
    
    # If base name exists, try with time component
    name_with_time = f"{base_date.strftime('%Y-%m-%d-%H-%M')}"
    if name_with_time not in existing_rides:
        return name_with_time
    
    # If still exists, add incremental suffix
    counter = 1
    while True:
        new_name = f"{base_name}_{counter}"
        if new_name not in existing_rides:
            return new_name
        counter += 1


def datetime_serializer(obj):
    """Custom JSON serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def parse_fit_metrics(fit_path: Path, parsed_dir: Path):
    """Extract session-level metrics and record data into files."""
    print(f"🔍 Parsing FIT file: {fit_path.name}")
    
    try:
        fit = FitFile(str(fit_path))
        
        # Parse session data (time-independent metrics)
        session_data = {}
        session_count = 0
        for msg in fit.get_messages("session"):
            session_count += 1
            for d in msg:
                # Convert non-serializable types to strings
                if hasattr(d.value, '__dict__'):
                    session_data[d.name] = str(d.value)
                else:
                    session_data[d.name] = d.value
        
        print(f"📊 Found {session_count} session(s) with {len(session_data)} metrics")
        
        # Save session data as CSV
        session_csv = parsed_dir / "session_metrics.csv"
        with open(session_csv, "w") as f:
            f.write("metric,value\n")
            for key, value in session_data.items():
                # Convert datetime objects to string for CSV
                if isinstance(value, datetime):
                    value = value.isoformat()
                f.write(f'"{key}","{value}"\n')
        
        # Save session data as JSON as well
        with open(parsed_dir / "session_metrics.json", "w") as f:
            json.dump(session_data, f, indent=4, default=datetime_serializer)
        
        # Parse record data (time-series metrics)
        records_csv = parsed_dir / "time_series.csv"
        record_count = 0
        with open(records_csv, "w") as f:
            # Write header
            f.write("timestamp,heart_rate,power,cadence,speed,altitude,distance,temperature\n")
            
            # Write data
            for record in fit.get_messages("record"):
                record_count += 1
                row_data = {
                    "timestamp": "",
                    "heart_rate": "",
                    "power": "",
                    "cadence": "",
                    "speed": "",
                    "altitude": "",
                    "distance": "",
                    "temperature": ""
                }
                
                for d in record:
                    if d.name in row_data:
                        # Convert datetime to string for CSV
                        if isinstance(d.value, datetime):
                            row_data[d.name] = d.value.isoformat()
                        else:
                            row_data[d.name] = d.value
                
                # Write the row
                f.write(",".join(str(row_data[x]) for x in row_data.keys()) + "\n")
        
        print(f"⏱️  Found {record_count} time series records")
        print(f"✅ Fit file parsed successfully!")
        print(f"   → Session metrics: {session_csv}")
        print(f"   → Time series data: {records_csv}")

    except Exception as e:
        print(f"❌ Error parsing {fit_path.name}: {e}")
        import traceback
        traceback.print_exc()


def create_symlinks(athlete_dir, ride_name, training_type, ride_date, dest_fit, parsed_dir):
    """Create organized symlinks in by_month and by_training directories"""
    # Create directory paths
    month_dir = athlete_dir / "by_month" / ride_date.strftime("%Y-%m")
    training_dir = athlete_dir / "by_training" / training_type
    
    # Create directories
    month_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training folder in each organizational directory (using just date, no training type)
    base_folder_name = ride_name  # This is now just the date (YYYY-MM-DD) or with suffix
    month_training_dir = month_dir / base_folder_name
    training_type_dir = training_dir / base_folder_name
    
    month_training_dir.mkdir(exist_ok=True)
    training_type_dir.mkdir(exist_ok=True)
    
    # Create symlinks to both FIT file and ParsedData
    symlink_pairs = [
        (month_training_dir, dest_fit, parsed_dir),
        (training_type_dir, dest_fit, parsed_dir)
    ]
    
    for target_dir, fit_source, parsed_source in symlink_pairs:
        try:
            # Symlink to FIT file
            fit_symlink = target_dir / f"{ride_name}.fit"
            if not fit_symlink.exists():
                os.symlink(os.path.relpath(fit_source, target_dir), fit_symlink)
            
            # Symlink to ParsedData directory
            parsed_symlink = target_dir / "ParsedData"
            if not parsed_symlink.exists():
                os.symlink(os.path.relpath(parsed_source, target_dir), parsed_symlink)
                
        except FileExistsError:
            pass
    
    return month_training_dir, training_type_dir


def main():
    print("=== Zwift Training Organizer ===")

    # === Choose athlete ===
    print(f"\nAvailable athletes in {ATHLETE_BASE_DIR}:")
    for i, a in enumerate(ATHLETES):
        print(f"  {i + 1}) {a}")
    
    while True:
        try:
            choice = int(input("Select athlete (number): "))
            if 1 <= choice <= len(ATHLETES):
                athlete = ATHLETES[choice - 1]
                break
            else:
                print("❌ Invalid choice, try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

    athlete_dir = ATHLETE_BASE_DIR / athlete
    print(f"✅ Selected athlete: {athlete}")
    print(f"✅ Athlete directory: {athlete_dir}")

    # Create necessary subdirectories
    all_data_dir = athlete_dir / "AllData"
    all_data_dir.mkdir(exist_ok=True)
    
    # === Ask for .fit path ===
    print("\nPlease enter the path of the .fit file to import:")
    print("(For example: ../Library/Mobile\\ Documents/com~apple~CloudDocs/Documents/Zwift/Activities/2025-10-31-19-56-19.fit)")
    fit_input = input("Path to .fit file: ").strip()

    # Clean up backslashes from macOS shell paths
    fit_input = fit_input.replace("\\", "")
    fit_path = Path(fit_input).expanduser().resolve()

    if not fit_path.exists():
        print(f"❌ File not found: {fit_path}")
        return
    else:
        print(f"✅ Found file: {fit_path}")

    # === Parse date from filename or get from user ===
    ride_date = parse_date_from_filename(fit_path.name)

    if not ride_date:
        ride_date = get_date_from_user(fit_path.name)
    else:
        print(f"✅ Parsed training date: {ride_date.strftime('%Y-%m-%d %H:%M')}")
        # Confirm with user if this is correct
        confirm = input("Is this date correct? (Y/n): ").strip().lower()
        if confirm == 'n':
            ride_date = get_date_from_user(fit_path.name)

    # === List available training templates ===
    trainings = sorted([f.name for f in TRAININGS_DIR.glob("*.zwo")])
    if not trainings:
        print("⚠️ No training templates found in Trainings/.")
        chosen_training = None
    else:
        print("\nAvailable training templates:")
        for i, t in enumerate(trainings):
            print(f"  {i + 1}) {t}")
        chosen_training = trainings[int(input("Select training: ")) - 1]

    # === Training type ===
    training_types = ["Z2", "Z3", "Z4", "Climb", "Sprint"]
    training_type = choose("Select the type of training:", training_types)

    # === Build unique ride name (now without training type) ===
    ride_name = get_unique_ride_name(athlete_dir, ride_date, training_type)
    print(f"✅ Using ride name: {ride_name}")

    parsed_dir = athlete_dir / "ParsedData" / ride_name
    parsed_dir.mkdir(parents=True, exist_ok=True)

    dest_fit = all_data_dir / f"{ride_name}.fit"
    shutil.copy(fit_path, dest_fit)

    # === Collect metadata ===
    nutrition_opts = ["Strong Deficit", "Deficit", "Normal", "Surplus", "Big Surplus"]
    effort_opts = ["Very easy", "Easy", "Medium", "Difficult", "Very difficult"]

    metadata = {
        "athlete": athlete,
        "ride_name": ride_name,
        "date": ride_date.strftime("%Y-%m-%d"),
        "time": ride_date.strftime("%H:%M"),
        "day_of_week": ride_date.strftime("%A"),
        "training_type": training_type,
        "training_template": chosen_training,
        "preworkout_nutrition": choose("Pre-workout nutrition:", nutrition_opts),
        "hydration_ml_pre": int(input("Water before training (ml): ")),
        "hydration_ml_during": int(input("Water during training (ml): ")),
        "minerals": {
            "potassium_g": float(input("Potassium (g): ")),
            "magnesium_g": float(input("Magnesium (g): ")),
            "zinc_g": float(input("Zinc (g): ")),
            "vitamin_c_g": float(input("Vitamin C (g): "))
        },
        "perceived_effort": choose("Perceived effort:", effort_opts),
        "notes": input("Notes (optional): ")
    }

    # === Save metadata.json ===
    with open(parsed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, default=datetime_serializer)

    # === Parse the fit file ===
    print(f"📂 Parsing FIT file and creating CSV files...")
    parse_fit_metrics(dest_fit, parsed_dir)

    # === Create organized symlinks ===
    month_dir, training_type_dir = create_symlinks(
        athlete_dir, ride_name, training_type, ride_date, dest_fit, parsed_dir
    )

    print(f"\n🎉 SUCCESS: Training '{ride_name}' added for {athlete}")
    print("=" * 50)
    print(f"📁 Files created:")
    print(f"   • Fit file: {dest_fit}")
    print(f"   • Metadata: {parsed_dir / 'metadata.json'}")
    print(f"   • Session metrics: {parsed_dir / 'session_metrics.csv'}")
    print(f"   • Time series data: {parsed_dir / 'time_series.csv'}")
    print(f"   • Session JSON: {parsed_dir / 'session_metrics.json'}")
    print(f"\n📂 Organizational structure:")
    print(f"   • by_month: {month_dir} (with symlinks to .fit and ParsedData)")
    print(f"   • by_training: {training_type_dir} (with symlinks to .fit and ParsedData)")
    print("=" * 50)


if __name__ == "__main__":
    main()
