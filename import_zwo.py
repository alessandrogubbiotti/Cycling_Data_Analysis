import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import json
import os
import shutil
from datetime import datetime

########## 2025-11-06 TO ADD A FUNCTION THAT COPIES THE FILE TO THE PARSED DIRECTORY. 


def safe_float_convert(value, default=0.0):
    """Safely convert value to float with error handling"""
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
def classify_interval(interval_type, power_pct=None, power_low_pct=None, power_high_pct=None):
    """
    Classify an interval based on its power values.

    Returns:
        zone_class (str): Z1–Z6 based on average power percentage
        power_type (str): 'steady', 'ramp_up', or 'ramp_down'
    """
    # Handle missing values safely
    power_pct = safe_float_convert(power_pct, default=0.0)
    power_low_pct = safe_float_convert(power_low_pct, default=power_pct)
    power_high_pct = safe_float_convert(power_high_pct, default=power_pct)

    # Determine ramp/steady classification
    if abs(power_high_pct - power_low_pct) < 1e-3:
        power_type = 'steady'
        avg_power = power_pct or power_low_pct
    elif power_high_pct > power_low_pct:
        power_type = 'ramp_up'
        avg_power = (power_low_pct + power_high_pct) / 2
    else:
        power_type = 'ramp_down'
        avg_power = (power_low_pct + power_high_pct) / 2

    # Zone classification based on average power (% of FTP)
    if avg_power < 0.55:
        zone_class = 'Z1'
    elif avg_power < 0.75:
        zone_class = 'Z2'
    elif avg_power < 0.90:
        zone_class = 'Z3'
    elif avg_power < 1.05:
        zone_class = 'Z4'
    elif avg_power < 1.20:
        zone_class = 'Z5'
    else:
        zone_class = 'Z6'

    # Override for special interval types
    if interval_type == 'warmup':
        zone_class = 'Z1'
    elif interval_type == 'cooldown':
        zone_class = 'Z1'

    return zone_class, power_type


def parse_zwo_workout(zwo_file_path):
    """
    Parse ZWO file and return sequential intervals without gaps.
    
    Args:
        zwo_file_path: Path to .zwo file
        
    Returns:
        list: Sequential intervals with proper classification
    """
    try:
        print(f"🔍 Parsing ZWO file: {zwo_file_path}")
        
        if not zwo_file_path.exists():
            raise FileNotFoundError(f"ZWO file not found: {zwo_file_path}")
        
        tree = ET.parse(zwo_file_path)
        root = tree.getroot()
        
        # Find workout element
        workout_element = root.find('workout')
        if workout_element is None:
            workout_element = root
        
        intervals = []
        current_time = 0.0  # Start at 0 seconds
        
        for element in workout_element:
            if element.tag in ['Warmup', 'Cooldown', 'SteadyState', 'Ramp', 'IntervalsT']:
                duration = safe_float_convert(element.get('Duration', 0))
                
                # Extract power values (as percentages) - MULTIPLY BY 100!
                power = safe_float_convert(element.get('Power', 0)) 
                power_low = safe_float_convert(element.get('PowerLow', 0)) 
                power_high = safe_float_convert(element.get('PowerHigh', 0)) 
                
                # Extract cadence
                target_cadence = safe_float_convert(element.get('Cadence', 0))
                
                interval_data = {
                    'start_time': current_time,
                    'end_time': current_time + duration,
                    'duration': duration,
                    'xml_type': element.tag.lower(),
                    'target_power_pct': power,
                    'target_power_low_pct': power_low,
                    'target_power_high_pct': power_high,
                    'target_cadence': target_cadence,
                    'description': f"{element.tag}",
                    'source': 'zwo'
                }
                
                # Handle IntervalsT (repeated intervals)
                if element.tag == 'IntervalsT':
                    repeat = int(element.get('Repeat', 1))
                    on_duration = safe_float_convert(element.get('OnDuration', duration))
                    off_duration = safe_float_convert(element.get('OffDuration', 0))
                    on_power = safe_float_convert(element.get('OnPower', 0))
                    off_power = safe_float_convert(element.get('OffPower', 0))
                    
                    # If OnPower/OffPower not specified, use PowerOn/PowerOff attributes
                    if on_power == 0:
                        on_power = safe_float_convert(element.get('PowerOn', 0)) 
                    if off_power == 0:
                        off_power = safe_float_convert(element.get('PowerOff', 0))
                    
                    for i in range(repeat):
                        # On interval (work interval)
                        zone_class, power_type = classify_interval('interval_on', on_power, on_power, on_power)
                        on_interval = {
                            **interval_data,
                            'start_time': current_time,
                            'end_time': current_time + on_duration,
                            'duration': on_duration,
                            'type': 'interval_on',
                            'zone_class': zone_class,
                            'power_type': power_type,
                            'target_power_pct': on_power,
                            'target_power_low_pct': on_power,
                            'target_power_high_pct': on_power,
                            'description': f'Interval {i+1}/{repeat} (On)'
                        }
                        intervals.append(on_interval)
                        current_time += on_duration
                        
                        # Off interval (recovery interval) - only if not last interval
                        if off_duration > 0 and i < repeat - 1:
                            zone_class, power_type = classify_interval('interval_off', off_power, off_power, off_power)
                            off_interval = {
                                **interval_data,
                                'start_time': current_time,
                                'end_time': current_time + off_duration,
                                'duration': off_duration,
                                'type': 'interval_off',
                                'zone_class': zone_class,
                                'power_type': power_type,
                                'target_power_pct': off_power,
                                'target_power_low_pct': off_power,
                                'target_power_high_pct': off_power,
                                'description': f'Interval {i+1}/{repeat} (Off)'
                            }
                            intervals.append(off_interval)
                            current_time += off_duration
                else:
                    # Classify non-IntervalsT elements
                    zone_class, power_type = classify_interval(
                        element.tag.lower(), 
                        power, 
                        power_low, 
                        power_high
                    )
                    interval_data.update({
                        'type': element.tag.lower(),
                        'zone_class': zone_class,
                        'power_type': power_type
                    })
                    intervals.append(interval_data)
                    current_time += duration
        
        print(f"✅ Parsed {len(intervals)} intervals from ZWO file")
        return intervals
        
    except Exception as e:
        print(f"❌ Error parsing ZWO file: {e}")
        raise

def write_intervals_to_file(intervals, file_path, format='csv'):
    """
    Write parsed intervals to file in various formats.
    """
    try:
        df = pd.DataFrame(intervals)
        
        if format.lower() == 'csv':
            file_path = Path(file_path).with_suffix('.csv')
            df.to_csv(file_path, index=False)
            print(f"✅ Intervals written to CSV: {file_path}")
            
        elif format.lower() == 'json':
            file_path = Path(file_path).with_suffix('.json')
            with open(file_path, 'w') as f:
                json.dump(intervals, f, indent=2, default=str)
            print(f"✅ Intervals written to JSON: {file_path}")
            
        elif format.lower() == 'excel':
            try:
                import openpyxl
                file_path = Path(file_path).with_suffix('.xlsx')
                df.to_excel(file_path, index=False)
                print(f"✅ Intervals written to Excel: {file_path}")
            except ImportError:
                print("⚠️  Excel support requires 'openpyxl' module. Skipping Excel export.")
                return None
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return file_path
        
    except Exception as e:
        print(f"❌ Error writing intervals to file: {e}")
        raise

def get_zwo_workout_summary(intervals):
    """Get summary of ZWO workout with proper classification"""
    if not intervals:
        return "No intervals found"
    
    total_duration = intervals[-1]['end_time']
    
    # Count by zone class and power type
    zone_classes = {}
    power_types = {}
    
    for interval in intervals:
        zone = interval.get('zone_class', 'Unknown')
        power_type = interval.get('power_type', 'Unknown')
        
        zone_classes[zone] = zone_classes.get(zone, 0) + 1
        power_types[power_type] = power_types.get(power_type, 0) + 1
    
    summary = f"Workout Summary:\n"
    summary += f"Total duration: {total_duration/60:.1f} minutes\n"
    summary += f"Number of intervals: {len(intervals)}\n"
    summary += f"Zone classes: {', '.join([f'{k}({v})' for k, v in zone_classes.items()])}\n"
    summary += f"Power types: {', '.join([f'{k}({v})' for k, v in power_types.items()])}\n"
    
    return summary

def print_intervals_table(intervals, max_rows=None):
    """Print intervals in a formatted table with explanations"""
    if not intervals:
        print("No intervals to display")
        return
    
    df = pd.DataFrame(intervals)
    
    # Select and order columns for display
    display_columns = ['start_time', 'end_time', 'duration', 'type', 'zone_class', 'power_type',
                      'target_power_pct', 'target_power_low_pct', 'target_power_high_pct', 'description']
    
    # Only include columns that exist in the data
    available_columns = [col for col in display_columns if col in df.columns]
    
    if max_rows is None or max_rows > len(df):
        max_rows = len(df)
    
    print(f"\n📊 Intervals Table (showing {min(max_rows, len(df))} of {len(df)} rows):")
    
    # Print column descriptions
    print("\n📋 Column Descriptions:")
    print("  - start_time/end_time/duration: Time in seconds")
    print("  - type: Interval type (warmup, steadystate, interval_on, interval_off)")
    print("  - zone_class: Warm up, Cooling down, Resting, Z2, Z3, Z4, Z5, Z6")
    print("  - power_type: 'steady' (constant power) or 'ramp' (changing power)")
    print("  - target_power_pct: Target power as % of FTP")
    print("  - target_power_low_pct: Starting power for ramps, same as target for steady")
    print("  - target_power_high_pct: Ending power for ramps, same as target for steady")
    print("  - description: Human-readable description")
    
    print(f"\n")
    print(df[available_columns].head(max_rows).to_string(index=False, float_format='%.1f'))
    
    if len(df) > max_rows:
        print(f"... and {len(df) - max_rows} more rows")

def confirm_parsing_correctness(intervals):
    """
    Display parsed intervals and ask user to confirm correctness.
    """
    print("\n" + "="*80)
    print("📋 PARSED INTERVALS - PLEASE REVIEW")
    print("="*80)
    
    # Print summary first
    summary = get_zwo_workout_summary(intervals)
    print(summary)
    
    # Print detailed table with explanations
    print_intervals_table(intervals, max_rows=20)
    
    # Explain power computation
    print("\n💡 Power Computation:")
    print("  - For 'steady' intervals: use target_power_pct directly")
    print("  - For 'ramp' intervals: power changes linearly from target_power_low_pct to target_power_high_pct")
    
    # Ask for confirmation
    print("\n" + "="*80)
    while True:
        response = input("❓ Is the parsing correct? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("✅ Parsing confirmed - proceeding with file save...")
            return True
        elif response in ['n', 'no']:
            print("❌ Parsing not confirmed - stopping execution.")
            return False
        else:
           print("⚠️  Please answer 'y' for yes or 'n' for no.")

def get_unique_training_name():
    """
    Ask user for training name and ensure it's unique in the Trainings directory.
    
    Returns:
        str: Unique training name
    """
    # Define root directory and trainings directory
    ROOT = Path(__file__).resolve().parent
    TRAININGS_DIR = ROOT / "Trainings"
    
    # Create Trainings directory if it doesn't exist
    TRAININGS_DIR.mkdir(exist_ok=True)
    
    # Get existing training names
    existing_trainings = [f.name for f in TRAININGS_DIR.iterdir() if f.is_dir()]
    
    print("\n📝 Training Naming")
    print("=" * 40)
    
    if existing_trainings:
        print("Existing trainings:")
        for training in sorted(existing_trainings):
            print(f"  - {training}")
        print()
    
    while True:
        training_name = input("Enter a name for this training: ").strip()
        
        if not training_name:
            print("❌ Training name cannot be empty. Please try again.")
            continue
            
        # Clean the training name
        clean_name = "".join(c for c in training_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_')
        
        if not clean_name:
            print("❌ Invalid training name. Please use letters, numbers, spaces, hyphens, or underscores.")
            continue
        
        # Check if training name already exists
        if clean_name in existing_trainings:
            print(f"❌ Training '{clean_name}' already exists.")
            overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
            if overwrite == 'y':
                # Remove existing directory
                import shutil
                existing_dir = TRAININGS_DIR / clean_name
                shutil.rmtree(existing_dir)
                print(f"✅ Removed existing training: {clean_name}")
                return clean_name
            else:
                print("Please choose a different name.")
                continue
        else:
            print(f"✅ Training name '{clean_name}' is available.")
            return clean_name

def create_training_folder(training_name):
    """
    Create training folder in ./Trainings/Name_of_the_training (relative path)
    """
    # Use current directory for relative paths
    current_dir = Path.cwd()
    training_base = Path("Trainings")
    
    # Create training base directory if it doesn't exist
    training_base.mkdir(exist_ok=True)
    
    # Create safe folder name
    safe_name = "".join(c for c in training_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    
    training_folder = training_base / safe_name
    
    # Add timestamp if folder already exists to avoid overwriting
    counter = 1
    original_name = training_folder
    while training_folder.exists():
        training_folder = original_name.parent / f"{original_name.name}_{counter}"
        counter += 1
    
    # Create the training folder
    training_folder.mkdir(parents=True)
    print(f"📁 Created training folder: {training_folder}")
    
    return training_folder

def get_training_name_from_zwo(zwo_file_path):
    """
    Extract training name from ZWO file.
    """
    try:
        tree = ET.parse(zwo_file_path)
        root = tree.getroot()
        
        # Try to get name from XML
        workout_name = root.get('name')
        if workout_name:
            return workout_name
        
        # Try to get name from workout element
        workout_element = root.find('workout')
        if workout_element is not None:
            workout_name = workout_element.get('name')
            if workout_name:
                return workout_name
        
        # Fall back to filename without extension
        return zwo_file_path.stem
        
    except Exception:
        return zwo_file_path.stem

def clean_file_path(file_path):
    """
    Clean file path by removing backslashes and normalizing.
    """
    # Replace backslashes with forward slashes
    cleaned_path = file_path.replace('\\', '')
    
    # Remove any quotes that might be around the path
    cleaned_path = cleaned_path.strip('"\'')
    
    print(f"🔄 Cleaned path: {cleaned_path}")
    return cleaned_path

def main():
    """
    Main function that uses relative paths throughout
    """
    print("🚴 ZWO Workout Parser")
    print("=" * 50)
    
    # Ask for ZWO file path
    while True:
        zwo_file_path = input("📁 Enter the path to your .zwo file: ").strip()
        
        if not zwo_file_path:
            print("❌ No file path provided. Please try again.")
            continue
        
        # Clean the path
        zwo_file_path = clean_file_path(zwo_file_path)
            
        zwo_path = Path(zwo_file_path)
        
        if not zwo_path.exists():
            print(f"❌ File not found: {zwo_path}")
            print("💡 Please check the path and try again.")
            continue
            
        if zwo_path.suffix.lower() != '.zwo':
            print("❌ Please provide a .zwo file")
            continue
            
        break
    
    try:
        # Parse the ZWO file
        print("\n" + "=" * 50)
        intervals = parse_zwo_workout(zwo_path)
        
        # Ask user to confirm parsing correctness
        if not confirm_parsing_correctness(intervals):
            print("🚫 Execution stopped by user.")
            return
    # Get unique training name from user
        training_name = get_unique_training_name()        
        # Get training name
#        training_name = get_training_name_from_zwo(zwo_path)
        print(f"🏷️  Training name: {training_name}")
        
        # Create training folder (relative path)
        training_folder = create_training_folder(training_name)
        
        # Copy original ZWO file to training folder
        destination_zwo = training_folder / zwo_path.name
        shutil.copy2(zwo_path, destination_zwo)
        print(f"✅ Copied original ZWO file to: {destination_zwo}")
        
        # Save intervals in multiple formats
        print("\n💾 Saving parsed intervals...")
        base_output_path = training_folder / "intervals"
        
        # Save as CSV
        csv_file = write_intervals_to_file(intervals, base_output_path, format='csv')
        
        # Save as JSON
        json_file = write_intervals_to_file(intervals, base_output_path, format='json')
        
        # Save as Excel
        excel_file = write_intervals_to_file(intervals, base_output_path, format='excel')
        
        # Generate and save summary
        summary = get_zwo_workout_summary(intervals)
        summary_file = training_folder / "workout_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"✅ Workout summary saved to: {summary_file}")
        
        # Display final results
        print("\n" + "=" * 50)
        print("🎯 WORKOUT PARSING COMPLETE")
        print("=" * 50)
        
        print(f"\n📁 All files saved in: {training_folder}")
        print("Files created:")
        print(f"  - {destination_zwo.name} (original ZWO file)")
        print(f"  - {csv_file.name} (intervals data)")
        print(f"  - {json_file.name} (intervals data)")
        if excel_file:
            print(f"  - {excel_file.name} (intervals data)")
        print(f"  - {summary_file.name} (workout summary)")
        
        print(f"\n✅ Success! Your training data has been organized in: {training_folder}")
        
    except Exception as e:
        print(f"❌ Error processing ZWO file: {e}")
        raise

if __name__ == "__main__":
    main()


