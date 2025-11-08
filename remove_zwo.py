import json
import shutil
from pathlib import Path
from datetime import datetime

def remove_training_template():
    """
    Remove a training template and clean up associated symlinks and metadata
    """
    # Define root directory
    ROOT = Path(__file__).resolve().parent
    TRAININGS_DIR = ROOT / "Trainings"
    ATHLETE_BASE_DIR = ROOT / "Athlete"
    
    # Check if directories exist
    if not TRAININGS_DIR.exists():
        print(f"❌ Trainings directory not found at: {TRAININGS_DIR}")
        return
    
    if not ATHLETE_BASE_DIR.exists():
        print(f"❌ Athlete directory not found at: {ATHLETE_BASE_DIR}")
        return
    
    # Get available training templates
    training_templates = sorted([f.name for f in TRAININGS_DIR.iterdir() if f.is_dir()])
    
    if not training_templates:
        print("❌ No training templates found in Trainings/ directory.")
        return
    
    print("🗑️  Training Template Removal")
    print("=" * 50)
    print("Available training templates:")
    for i, template in enumerate(training_templates):
        print(f"  {i + 1}) {template}")
    
    # Let user select training template to remove
    while True:
        try:
            choice = int(input("\nSelect training template to remove (number): "))
            if 1 <= choice <= len(training_templates):
                template_to_remove = training_templates[choice - 1]
                break
            else:
                print("❌ Invalid choice, try again.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Confirm removal
    confirm = input(f"\n⚠️  Are you sure you want to remove training template '{template_to_remove}'? This cannot be undone! (y/N): ").strip().lower()
    if confirm != 'y':
        print("🚫 Removal cancelled.")
        return
    
    print(f"\n🔍 Removing training template: {template_to_remove}")
    
    # Get all athletes
    athletes = [d for d in ATHLETE_BASE_DIR.iterdir() if d.is_dir()]
    
    # Track affected training sessions
    all_affected_sessions = []
    
    if not athletes:
        print("⚠️  No athlete folders found.")
    else:
        print(f"👥 Processing {len(athletes)} athletes...")
        
        for athlete_dir in athletes:
            athlete_name = athlete_dir.name
            print(f"  👤 Processing athlete: {athlete_name}")
            
            # Path to by_training folder for this template
            by_training_template_dir = athlete_dir / "by_training" / template_to_remove
            
            if by_training_template_dir.exists():
                # Find all training sessions that used this template
                training_sessions = []
                for session_dir in by_training_template_dir.iterdir():
                    if session_dir.is_dir():
                        training_sessions.append(session_dir.name)
                
                print(f"    📁 Found {len(training_sessions)} training sessions using this template")
                
                # Update metadata for each training session
                athlete_affected_sessions = []
                for session_name in training_sessions:
                    parsed_data_dir = athlete_dir / "ParsedData" / session_name
                    metadata_file = parsed_data_dir / "metadata.json"
                    
                    if metadata_file.exists():
                        try:
                            # Read and update metadata
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Store the original template for warning message
                            original_template = metadata.get('training_template', 'Unknown')
                            
                            # Update metadata
                            metadata['training_template'] = "Free Ride"
                            metadata['template_removal_warning'] = {
                                'original_template': original_template,
                                'removal_date': datetime.now().isoformat(),
                                'warning': 'This training was originally based on a template that has been removed. Some metrics might need manual adjustment.'
                            }
                            
                            # Write updated metadata
                            with open(metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=4)
                            
                            # Add to affected sessions list
                            session_info = {
                                'athlete': athlete_name,
                                'session_name': session_name,
                                'date': metadata.get('date', 'Unknown'),
                                'original_template': original_template
                            }
                            athlete_affected_sessions.append(session_info)
                            all_affected_sessions.append(session_info)
                            
                            print(f"    ⚠️  Updated metadata for session: {session_name}")
                            print(f"      Changed training_template from '{original_template}' to 'Free Ride'")
                            
                        except Exception as e:
                            print(f"    ❌ Error updating metadata for {session_name}: {e}")
                    else:
                        print(f"    ⚠️  Metadata file not found for session: {session_name}")
                
                # Print affected sessions for this athlete
                if athlete_affected_sessions:
                    print(f"    📋 Affected sessions for {athlete_name}:")
                    for session in athlete_affected_sessions:
                        print(f"      • {session['session_name']} ({session['date']})")
                
                # Remove the by_training template directory
                try:
                    shutil.rmtree(by_training_template_dir)
                    print(f"    ✅ Removed by_training directory: {by_training_template_dir}")
                except Exception as e:
                    print(f"    ❌ Error removing by_training directory: {e}")
            else:
                print(f"    ℹ️  No by_training directory found for this template")
    
    # Remove the training template from Trainings directory
    training_template_dir = TRAININGS_DIR / template_to_remove
    if training_template_dir.exists():
        try:
            shutil.rmtree(training_template_dir)
            print(f"\n✅ Successfully removed training template: {training_template_dir}")
        except Exception as e:
            print(f"❌ Error removing training template directory: {e}")
    else:
        print(f"❌ Training template directory not found: {training_template_dir}")
    
    # Print comprehensive summary of affected sessions
    print(f"\n📊 REMOVAL SUMMARY")
    print("=" * 60)
    print(f"Training template removed: {template_to_remove}")
    print(f"Total affected training sessions: {len(all_affected_sessions)}")
    
    if all_affected_sessions:
        print(f"\n📋 AFFECTED TRAINING SESSIONS:")
        print("-" * 60)
        
        # Group by athlete for better display
        athletes_affected = {}
        for session in all_affected_sessions:
            athlete = session['athlete']
            if athlete not in athletes_affected:
                athletes_affected[athlete] = []
            athletes_affected[athlete].append(session)
        
        for athlete, sessions in athletes_affected.items():
            print(f"\n👤 {athlete}:")
            for session in sessions:
                print(f"   • {session['session_name']} - {session['date']}")
                print(f"     (was: {session['original_template']})")
    
    print(f"\n🎯 Actions Completed:")
    print(f"   • Training template '{template_to_remove}' removed from Trainings/")
    print(f"   • by_training directories removed for all athletes")
    print(f"   • {len(all_affected_sessions)} training sessions updated to 'Free Ride'")
    print(f"   • Warning messages added to affected training sessions")
    
    if all_affected_sessions:
        print(f"\n⚠️  IMPORTANT: {len(all_affected_sessions)} training sessions have been affected.")
        print("   These sessions have been marked as 'Free Ride' and may need manual review.")

def main():
    """
    Main function for training template removal
    """
    print("🚴 Training Template Remover")
    print("=" * 50)
    print("This script will:")
    print("  • Remove a training template from the Trainings/ directory")
    print("  • Remove associated by_training directories for all athletes")
    print("  • Update training session metadata to mark them as 'Free Ride'")
    print("  • Add warning messages to affected training sessions")
    print("  • Display all affected training sessions on screen")
    print()
    
    remove_training_template()

if __name__ == "__main__":
    main()
