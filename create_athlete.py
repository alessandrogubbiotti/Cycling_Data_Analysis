#### This is no a so innocent script as expected. We should create in a second moment some functions that update the biobetric data of the athlete, such as FTP, maxHR, (but also the age). We should put all of this in a class. 

import json
from pathlib import Path
from datetime import datetime

def create_athlete():
    """
    Create a new athlete folder with proper structure and athlete info file
    """
    # Define root directory
    ROOT = Path(__file__).resolve().parent
    ATHLETE_BASE_DIR = ROOT / "Athlete"
    
    # Create Athlete directory if it doesn't exist
    ATHLETE_BASE_DIR.mkdir(exist_ok=True)
    
    print("🚴 Create New Athlete")
    print("=" * 50)
    
    # Get athlete name
    while True:
        athlete_name = input("Enter athlete name: ").strip()
        
        if not athlete_name:
            print("❌ Athlete name cannot be empty.")
            continue
            
        # Clean the athlete name for folder naming
        clean_name = "".join(c for c in athlete_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_')
        
        if not clean_name:
            print("❌ Invalid athlete name. Please use letters, numbers, spaces, hyphens, or underscores.")
            continue
        
        # Check if athlete folder already exists
        athlete_dir = ATHLETE_BASE_DIR / clean_name
        if athlete_dir.exists():
            print(f"❌ Athlete folder already exists: {clean_name}")
            continue
            
        break
    
    # Create the athlete directory structure
    directories = [
        "AllData",
        "ParsedData", 
        "by_month",
        "by_zone",
        "by_training"
    ]
    
    print(f"\n📁 Creating directory structure for {athlete_name}...")
    for dir_name in directories:
        (athlete_dir / dir_name).mkdir(parents=True)
        print(f"  ✅ {dir_name}")
    
    # Collect basic athlete information
    print(f"\n📝 Athlete Information (press Enter to skip any field)")
    
    athlete_info = {
        "name": athlete_name,
        "folder_name": clean_name,
        "created_date": datetime.now().isoformat(),
        "personal_info": {
            "birth_date": input("Birth date (YYYY-MM-DD): ").strip() or None,
            "height_cm": safe_float_input("Height (cm): "),
            "weight_kg": safe_float_input("Weight (kg): "),
            "ftp": safe_int_input("Current FTP: "),
            "max_hr": safe_int_input("Max Heart Rate: "),
            "resting_hr": safe_int_input("Resting Heart Rate: "),
        },
        "contact_info": {
            "email": input("Email: ").strip() or None,
            "phone": input("Phone: ").strip() or None,
        },
        "training_preferences": {
            "preferred_training_days": input("Preferred training days (e.g., Mon,Wed,Fri): ").strip() or None,
            "preferred_training_time": input("Preferred training time (e.g., Morning, Evening): ").strip() or None,
            "time_availability_hours": safe_float_input("Weekly training time availability (hours): "),
        },
        "goals": {
            "short_term": input("Short-term goals: ").strip() or None,
            "long_term": input("Long-term goals: ").strip() or None,
            "target_events": input("Target events/races: ").strip() or None,
        },
        "medical_notes": {
            "injuries": input("Current/recent injuries: ").strip() or None,
            "limitations": input("Physical limitations: ").strip() or None,
            "medications": input("Medications: ").strip() or None,
        },
        "notes": input("Additional notes: ").strip() or None
    }
    
    # Save athlete info
    athlete_info_file = athlete_dir / "athlete_info.json"
    with open(athlete_info_file, 'w') as f:
        json.dump(athlete_info, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Successfully created athlete: {athlete_name}")
    print(f"📁 Athlete folder: {athlete_dir}")
    print(f"📄 Athlete info: {athlete_info_file}")
    
    # Show what was created
    print(f"\n📋 Created structure:")
    print(f"  • AllData/ - for .fit files")
    print(f"  • ParsedData/ - for analyzed training data") 
    print(f"  • by_month/ - monthly organization")
    print(f"  • by_zone/ - zone-based organization")
    print(f"  • by_training/ - template-based organization")
    print(f"  • athlete_info.json - athlete information")

def safe_float_input(prompt):
    """Safely get float input with error handling"""
    try:
        value = input(prompt).strip()
        return float(value) if value else None
    except ValueError:
        return None

def safe_int_input(prompt):
    """Safely get integer input with error handling"""
    try:
        value = input(prompt).strip()
        return int(value) if value else None
    except ValueError:
        return None

def main():
    """
    Main function for creating new athletes
    """
    print("🚴 Athlete Creation Tool")
    print("=" * 50)
    print("This script will:")
    print("  • Create a new athlete folder with proper structure")
    print("  • Create an athlete information file")
    print("  • Set up organizational directories (by_month, by_zone, by_training)")
    print()
    
    create_athlete()

if __name__ == "__main__":
    main()
