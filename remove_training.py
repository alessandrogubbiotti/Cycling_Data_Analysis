import json
import os
import shutil
from pathlib import Path

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

def find_training_by_name(athlete_dir, training_name):
    """Find all instances of a training across the organizational structure"""
    locations = {}
    
    # Check AllData
    all_data_file = athlete_dir / "AllData" / f"{training_name}.fit"
    if all_data_file.exists():
        locations["all_data"] = all_data_file
    
    # Check ParsedData
    parsed_data_dir = athlete_dir / "ParsedData" / training_name
    if parsed_data_dir.exists():
        locations["parsed_data"] = parsed_data_dir
    
    # Check by_month directories
    month_dirs = list((athlete_dir / "by_month").glob("*/" + training_name))
    if month_dirs:
        locations["by_month"] = month_dirs[0].parent / training_name
    
    # Check by_training directories
    training_type_dirs = list((athlete_dir / "by_training").glob("*/" + training_name))
    if training_type_dirs:
        locations["by_training"] = training_type_dirs[0].parent / training_name
    
    return locations

def get_months_with_trainings(athlete_dir):
    """Get all months that have trainings"""
    by_month_dir = athlete_dir / "by_month"
    if not by_month_dir.exists():
        return []
    
    months = []
    for month_dir in by_month_dir.iterdir():
        if month_dir.is_dir():
            # Check if this month has any training folders
            trainings = [d for d in month_dir.iterdir() if d.is_dir()]
            if trainings:
                months.append(month_dir.name)
    
    return sorted(months)

def get_trainings_in_month(athlete_dir, month):
    """Get all trainings in a specific month"""
    month_dir = athlete_dir / "by_month" / month
    if not month_dir.exists():
        return []
    
    trainings = []
    for training_dir in month_dir.iterdir():
        if training_dir.is_dir():
            trainings.append(training_dir.name)
    
    return sorted(trainings)

def get_training_type_from_metadata(athlete_dir, training_name):
    """Get training type from metadata.json"""
    parsed_data_dir = athlete_dir / "ParsedData" / training_name
    metadata_file = parsed_data_dir / "metadata.json"
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata.get('training_type', 'Unknown')
        except:
            return 'Unknown'
    return 'Unknown'

def remove_training(athlete_dir, training_name):
    """Remove a specific training and all its associated data"""
    training_type = get_training_type_from_metadata(athlete_dir, training_name)
    print(f"\n🗑️  Removing training: {training_name} ({training_type})")
    
    # Find all locations of this training
    locations = find_training_by_name(athlete_dir, training_name)
    
    if not locations:
        print(f"❌ Training '{training_name}' not found for athlete {athlete_dir.name}")
        return False
    
    # Show what will be removed
    print("The following items will be removed:")
    for loc_type, path in locations.items():
        if isinstance(path, list):
            # Handle lists of files (like CSV files)
            for file_path in path:
                if file_path.exists():
                    print(f"  - {loc_type}: {file_path}")
        else:
            # Handle single paths
            if path.exists():
                print(f"  - {loc_type}: {path}")
    
    # Confirm deletion
    confirm = input("\nAre you sure you want to delete this training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Deletion cancelled.")
        return False
    
    # Remove all found locations
    success = True
    for loc_type, path in locations.items():
        try:
            if isinstance(path, list):
                # Handle lists of files
                for file_path in path:
                    if file_path.exists():
                        if file_path.is_file():
                            file_path.unlink()
                            print(f"✅ Removed file: {file_path}")
                        else:
                            shutil.rmtree(file_path)
                            print(f"✅ Removed directory: {file_path}")
            else:
                # Handle single paths
                if path.exists():
                    if path.is_file():
                        path.unlink()
                        print(f"✅ Removed file: {path}")
                    else:
                        shutil.rmtree(path)
                        print(f"✅ Removed directory: {path}")
        except Exception as e:
            print(f"❌ Error removing {path}: {e}")
            success = False
    
    # Clean up empty directories in organizational structure
    clean_empty_directories(athlete_dir)
    
    return success

def clean_empty_directories(athlete_dir):
    """Remove empty directories from organizational structure"""
    organizational_dirs = [
        athlete_dir / "by_month",
        athlete_dir / "by_training"
    ]
    
    for org_dir in organizational_dirs:
        if org_dir.exists():
            # Walk through all subdirectories and remove empty ones
            for root, dirs, files in os.walk(org_dir, topdown=False):
                current_dir = Path(root)
                if not any(current_dir.iterdir()):  # Directory is empty
                    try:
                        current_dir.rmdir()
                        print(f"🧹 Removed empty directory: {current_dir}")
                    except OSError:
                        pass  # Directory not empty or other issue

def main():
    print("=== Zwift Training Removal Tool ===")
    
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
    
    # === Get months with trainings ===
    months = get_months_with_trainings(athlete_dir)
    
    if not months:
        print(f"❌ No trainings found for athlete {athlete}")
        return
    
    # === Choose month ===
    print(f"\nAvailable months for {athlete}:")
    for i, month in enumerate(months):
        print(f"  {i + 1}) {month}")
    
    while True:
        try:
            choice = int(input("Select month (number): "))
            if 1 <= choice <= len(months):
                selected_month = months[choice - 1]
                break
            else:
                print("❌ Invalid choice, try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

    # === Get trainings in selected month ===
    trainings = get_trainings_in_month(athlete_dir, selected_month)
    
    if not trainings:
        print(f"❌ No trainings found in {selected_month}")
        return
    
    # Display trainings with their types
    print(f"\nAvailable trainings in {selected_month}:")
    training_info = []
    for training in trainings:
        training_type = get_training_type_from_metadata(athlete_dir, training)
        training_info.append((training, training_type))
    
    selected_indices = []
    
    while True:
        print("\n" + "="*50)
        for i, (training, training_type) in enumerate(training_info):
            marker = "✓" if i in selected_indices else " "
            print(f"  {marker} {i + 1}) {training} ({training_type})")
        
        print("\nOptions:")
        print("  [number] - Toggle selection of training")
        print("  a - Select all trainings")
        print("  c - Clear all selections")
        print("  d - Delete selected trainings")
        print("  q - Quit without deleting")
        
        user_input = input("\nEnter choice: ").strip().lower()
        
        if user_input == 'q':
            print("❌ Cancelled.")
            return
        elif user_input == 'a':
            selected_indices = list(range(len(trainings)))
            print("✅ Selected all trainings")
        elif user_input == 'c':
            selected_indices = []
            print("✅ Cleared all selections")
        elif user_input == 'd':
            if not selected_indices:
                print("❌ No trainings selected.")
                continue
            
            # Remove selected trainings
            trainings_to_remove = [trainings[i] for i in selected_indices]
            print(f"\nThe following trainings will be REMOVED:")
            for training in trainings_to_remove:
                training_type = get_training_type_from_metadata(athlete_dir, training)
                print(f"  - {training} ({training_type})")
            
            confirm = input("\n🚨 ARE YOU SURE? This cannot be undone! (type 'DELETE' to confirm): ").strip()
            if confirm == 'DELETE':
                all_success = True
                for training in trainings_to_remove:
                    success = remove_training(athlete_dir, training)
                    if not success:
                        all_success = False
                        print(f"❌ Failed to remove training: {training}")
                
                if all_success:
                    print("\n🎉 SUCCESS: All selected trainings removed successfully!")
                else:
                    print("\n⚠️  WARNING: Some trainings may not have been completely removed.")
            else:
                print("❌ Deletion cancelled.")
            return
        else:
            try:
                idx = int(user_input) - 1
                if 0 <= idx < len(trainings):
                    if idx in selected_indices:
                        selected_indices.remove(idx)
                        training_type = training_info[idx][1]
                        print(f"❌ Deselected: {trainings[idx]} ({training_type})")
                    else:
                        selected_indices.append(idx)
                        training_type = training_info[idx][1]
                        print(f"✅ Selected: {trainings[idx]} ({training_type})")
                else:
                    print("❌ Invalid training number.")
            except ValueError:
                print("❌ Please enter a valid number or option.")


if __name__ == "__main__":
    main()
