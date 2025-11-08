# ui_helpers.py
"""
UI helper functions for the Cycling Data Analysis System
"""

from pathlib import Path

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

def yes_no(prompt):
    """Simple yes/no input"""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("❌ Please enter 'y' or 'n'")

def get_athletes():
    """Get list of athletes from Athlete directory"""
    athlete_dir = Path("Athlete")
    if not athlete_dir.exists():
        print("❌ Athlete directory not found!")
        return []
    
    athletes = [d.name for d in athlete_dir.iterdir() if d.is_dir() and d.name not in (".git",)]
    return sorted(athletes)

def get_months(athlete_name):
    """Get list of months with trainings for an athlete"""
    by_month_dir = Path("Athlete") / athlete_name / "by_month"
    if not by_month_dir.exists():
        return []
    
    months = [d.name for d in by_month_dir.iterdir() if d.is_dir()]
    return sorted(months, reverse=True)  # Most recent first

def get_trainings_by_month(athlete_name, month):
    """Get list of trainings for a specific month"""
    month_dir = Path("Athlete") / athlete_name / "by_month" / month
    if not month_dir.exists():
        return []
    
    trainings = [d.name for d in month_dir.iterdir() if d.is_dir()]
    return sorted(trainings)

def follow_symlink_to_parsed_data(athlete_name, month, training_name):
    """Follow the symlink from by_month to get the actual ParsedData path"""
    month_training_dir = Path("Athlete") / athlete_name / "by_month" / month / training_name
    
    # Check if ParsedData symlink exists
    parsed_symlink = month_training_dir / "ParsedData"
    if parsed_symlink.exists() and parsed_symlink.is_symlink():
        try:
            # Resolve the symlink to get the actual ParsedData path
            actual_parsed_path = parsed_symlink.resolve()
            return actual_parsed_path
        except Exception as e:
            print(f"⚠️ Could not resolve symlink: {e}")
    
    # Fallback: try the direct ParsedData path
    fallback_path = Path("Athlete") / athlete_name / "ParsedData" / training_name
    if fallback_path.exists():
        return fallback_path
    
    return None
