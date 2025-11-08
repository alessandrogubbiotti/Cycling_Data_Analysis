import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import json
import os
import shutil
from datetime import datetime

def safe_float_convert(value, default=0.0):
    """Safely convert value to float with error handling"""
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def classify_zone(power_pct, interval_type=None):
    """
    Classify interval into power zones based on target power percentage.
    
    Zone classification (typical FTP-based):
    - Recovery: < 55%
    - Endurance (Z2): 55-75%
    - Tempo (Z3): 75-90%
    - Threshold (Z4): 90-105%
    - VO2Max (Z5): 105-120%
    - Anaerobic (Z6): > 120%
    """
    if interval_type in ['warmup', 'cooldown']:
        return interval_type.capitalize()
    
    power = power_pct
    
    if power < 55:
        return "Recovery"
    elif 55 <= power < 75:
        return "Endurance (Z2)"
    elif 75 <= power < 90:
        return "Tempo (Z3)"
    elif 90 <= power < 105:
        return "Threshold (Z4)"
    elif 105 <= power < 120:
        return "VO2Max (Z5)"
    else:
        return "Anaerobic (Z6)"

def classify_ramp_steady(interval_type, element):
    """
    Classify interval as ramp or steady state.
    Warmup and cooldown are always ramping.
    """
    if interval_type in ['warmup', 'cooldown']:
        return 'ramp'
    elif interval_type == 'ramp':
        return 'ramp'
    elif interval_type == 'steadystate':
        return 'steady'
    elif interval_type == 'intervalsT':
        # For IntervalsT, check if it's on or off
        if 'interval_on' in element.get('description', ''):
            return 'steady'  # Interval work is typically steady
        elif 'interval_off' in element.get('description', ''):
            return 'steady'  # Recovery is typically steady
        else:
            return 'steady'
    else:
        return 'steady'

def parse_zwo_workout(zwo_file_path):
    """
    Parse ZWO file and return sequential intervals without gaps.
    
    Args:
        zwo_file_path: Path to .zwo file
        
    Returns:
        list: Sequential intervals with type, duration, target_power_pct, target_cadence, etc.
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
                
                # Extract power values (as percentages)
                power = safe_float_convert(element.get('Power', 0))
                power_low = safe_float_convert(element.get('PowerLow', 0))
                power_high = safe_float_convert(element.get('PowerHigh', 0))
                
                # Determine target power based on element type
                if element.tag == 'Warmup':
                    target_power_pct = power_high
                elif element.tag == 'Cooldown':
                    target_power_pct = power_low
                elif element.tag == 'SteadyState':
                    target_power_pct = power
                elif element.tag == 'Ramp':
                    target_power_pct = (power_low + power_high) / 2
                elif element.tag == 'IntervalsT':
                    target_power_pct = safe_float_convert(element.get('PowerOn', power))
                
                # Extract cadence
                target_cadence = safe_float_convert(element.get('Cadence', 0))
                
                # Classify zone
                zone = classify_zone(target_power_pct, element.tag.lower())
                
                interval_data = {
                    'start_time': current_time,
                    'end_time': current_time + duration,
                    'duration': duration,
                    'type': element.tag.lower(),
                    'target_power_pct': target_power_pct,
                    'target_power_low_pct': power_low,
                    'target_power_high_pct': power_high,
                    'target_cadence': target_cadence,
                    'zone': zone,
                    'description': f"{element.tag}",
                    'source': 'zwo'
                }
                
                # Handle IntervalsT (repeated intervals)
                if element.tag == 'IntervalsT':
                    repeat = int(element.get('Repeat', 1))
                    on_duration = safe_float_convert(element.get('OnDuration', duration))
                    off_duration = safe_float_convert(element.get('OffDuration', 0))
                    on_power = safe_float_convert(element.get('PowerOn', target_power_pct))
                    off_power = safe_float_convert(element.get('PowerOff', power_low))
                    
                    for i in range(repeat):
                        # On interval
                        on_zone = classify_zone(on_power, 'interval_on')
                        on_interval = {
                            **interval_data,
                            'start_time': current_time,
                            'end_time': current_time + on_duration,
                            'duration': on_duration,
                            'type': 'interval_on',
                            'target_power_pct': on_power,
                            'zone': on_zone,
                            'description': f'Interval {i+1}/{repeat} (On)'
                        }
                        # Add ramp/steady classification
                        on_interval['ramp_steady'] = classify_ramp_steady('intervalsT', on_interval)
                        intervals.append(on_interval)
                        current_time += on_duration
                        
                        # Off interval (if not last)
                        if off_duration > 0 and i < repeat - 1:
                            off_zone = classify_zone(off_power, 'interval_off')
                            off_interval = {
                                **interval_data,
                                'start_time': current_time,
                                'end_time': current_time + off_duration,
                                'duration': off_duration,
                                'type': 'interval_off',
                                'target_power_pct': off_power,
                                'zone': off_zone,
                                'description': f'Interval {i+1}/{repeat} (Off)'
                            }
                            # Add ramp/steady classification
                            off_interval['ramp_steady'] = classify_ramp_steady('intervalsT', off_interval)
                            intervals.append(off_interval)
                            current_time += off_duration
                else:
                    # Add ramp/steady classification for non-IntervalsT elements
                    interval_data['ramp_steady'] = classify_ramp_steady(element.tag.lower(), element)
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
    
    Args:
        intervals: List of interval dictionaries
        file_path: Output file path
        format: 'csv', 'json', or 'excel'
    """
    try:
        df = pd.DataFrame(intervals)
        
        if format.lower() == 'csv':
            file_path = Path(file_path).with_suffix('.csv')
            df.to_csv(file_path, index=False)
            print(f"✅ Intervals written to CSV: {file_path}")
            
        elif format.lower() == 'json':
            file_path = Path(file_path).with_suffix('.json')
            # Convert to serializable format
            serializable_intervals = []
            for interval in intervals:
                serializable_interval = {}
                for key, value in interval.items():
                    # Convert non-serializable types
                    if isinstance(value, (int, float, str, bool, type(None))):
                        serializable_interval[key] = value
                    else:
                        serializable_interval[key] = str(value)
                serializable_intervals.append(serializable_interval)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_intervals, f, indent=2)
            print(f"✅ Intervals written to JSON: {file_path}")
            
        elif format.lower() == 'excel':
            try:
                # Try to import openpyxl to check if it's available
                import openpyxl
                file_path = Path(file_path).with_suffix('.xlsx')
                df.to_excel(file_path, index=False)
                print(f"✅ Intervals written to Excel: {file_path}")
            except ImportError:
                print("⚠️  Excel support requires 'openpyxl' module. Skipping Excel export.")
                print("💡 Install it with: pip install openpyxl")
                return None
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return file_path
        
    except Exception as e:
        print(f"❌ Error writing intervals to file: {e}")
        raise

def read_intervals_from_file(file_path):
    """
    Read intervals from file (CSV, JSON, or Excel).
    
    Args:
        file_path: Input file path
        
    Returns:
        list: List of interval dictionaries
    """
    try:
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            intervals = df.to_dict('records')
            
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                intervals = json.load(f)
                
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(file_path)
                intervals = df.to_dict('records')
            except ImportError:
                print("⚠️  Excel support requires 'openpyxl' module.")
                raise ImportError("Please install openpyxl: pip install openpyxl")
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        print(f"✅ Read {len(intervals)} intervals from {file_path}")
        return intervals
        
    except Exception as e:
        print(f"❌ Error reading intervals from file: {e}")
        raise

def get_zwo_workout_summary(intervals):
    """Get summary of ZWO workout with zone classification"""
    if not intervals:
        return "No intervals found"
    
    total_duration = intervals[-1]['end_time']
    
    # Count by type and zone
    interval_types = {}
    zone_distribution = {}
    ramp_steady_distribution = {}
    power_zones = {}
    
    for interval in intervals:
        # Count by type
        interval_types[interval['type']] = interval_types.get(interval['type'], 0) + 1
        
        # Count by zone
        zone = interval.get('zone', 'Unknown')
        zone_distribution[zone] = zone_distribution.get(zone, 0) + 1
        
        # Count by ramp/steady
        ramp_steady = interval.get('ramp_steady', 'Unknown')
        ramp_steady_distribution[ramp_steady] = ramp_steady_distribution.get(ramp_steady, 0) + 1
        
        # Track power by zone
        if zone not in power_zones:
            power_zones[zone] = []
        power_zones[zone].append(interval['target_power_pct'])
    
    # Calculate average power by zone
    avg_power_by_zone = {}
    for zone, powers in power_zones.items():
        avg_power_by_zone[zone] = sum(powers) / len(powers)
    
    summary = f"ZWO Workout Summary:\n"
    summary += f"Total duration: {total_duration/60:.1f} minutes\n"
    summary += f"Number of intervals: {len(intervals)}\n"
    summary += f"Interval types: {', '.join([f'{k}({v})' for k, v in interval_types.items()])}\n"
    summary += f"Zone distribution: {', '.join([f'{k}({v})' for k, v in zone_distribution.items()])}\n"
    summary += f"Ramp/Steady: {', '.join([f'{k}({v})' for k, v in ramp_steady_distribution.items()])}\n"
    summary += "Average power by zone:\n"
    for zone, avg_power in avg_power_by_zone.items():
        summary += f"  - {zone}: {avg_power:.1f}%\n"
    
    return summary

def print_intervals_table(intervals, max_rows=None):
    """Print intervals in a formatted table"""
    if not intervals:
        print("No intervals to display")
        return
    
    df = pd.DataFrame(intervals)
    
    # Select and order columns for display
    display_columns = ['start_time', 'end_time', 'duration', 'type', 'zone', 'ramp_steady',
                      'target_power_pct', 'target_cadence', 'description']
    
    # Only include columns that exist in the data
    available_columns = [col for col in display_columns if col in df.columns]
    
    if max_rows is None or max_rows > len(df):
        max_rows = len(df)
    
    print(f"\n📊 Intervals Table (showing {min(max_rows, len(df))} of {len(df)} rows):")
    print(df[available_columns].head(max_rows).to_string(index=False, float_format='%.1f'))
    
    if len(df) > max_rows:
        print(f"... and {len(df) - max_rows} more rows")

def confirm_parsing_correctness(intervals):
    """
    Display parsed intervals and ask user to confirm correctness.
    
    Args:
        intervals: List of parsed intervals
        
    Returns:
        bool: True if user confirms correctness, False otherwise
    """
    print("\n" + "="*60)
    print("📋 PARSED INTERVALS - PLEASE REVIEW")
    print("="*60)
    
    # Print summary first
    summary = get_zwo_workout_summary(intervals)
    print(summary)
    
    # Print detailed table
    print_intervals_table(intervals, max_rows=20)  # Show first 20 rows
    
    # Ask for confirmation
    print("\n" + "="*60)
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

def create_training_folder(training_name):
    """
    Create training folder in ~/Training/Name_of_the_training
    
    Args:
        training_name: Name of the training (will be used for folder name)
        
    Returns:
        Path: Path to the created training folder
    """
    # Expand user home directory
    home_dir = Path.home()
    training_base = home_dir / "Training"
    
    # Create training base directory if it doesn't exist
    training_base.mkdir(exist_ok=True)
    
    # Create safe folder name (remove invalid characters)
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
    First tries to get name from XML, then falls back to filename.
    """
    try:
        tree = ET.parse(zwo_file_path)
        root = tree.getroot()
        
        # Try to get name from workout attributes
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
        # If any error occurs, fall back to filename
        return zwo_file_path.stem

def clean_file_path(file_path):
    """
    Clean file path by removing backslashes and normalizing.
    
    Args:
        file_path: Input file path that might contain backslashes
        
    Returns:
        str: Cleaned file path with forward slashes
    """
    # Replace backslashes with forward slashes
    cleaned_path = file_path.replace('\\','')
    
    # Remove any quotes that might be around the path
    cleaned_path = cleaned_path.strip('"\'')
    
    print(f"🔄 Cleaned path: {cleaned_path}")
    return cleaned_path

def main():
    """
    Main function that:
    1. Asks for a .zwo filepath
    2. Parses the training
    3. Shows parsed intervals and asks for confirmation
    4. Creates a folder in ~/Training/Name_of_the_training
    5. Saves the original .zwo file and parsed intervals in the folder
    """
    print("🚴 ZWO Workout Parser")
    print("=" * 50)
    
    # Ask for ZWO file path
    while True:
        zwo_file_path = input("📁 Enter the path to your .zwo file: ").strip()
        
        if not zwo_file_path:
            print("❌ No file path provided. Please try again.")
            continue
        
        # Clean the path (remove backslashes, quotes, etc.)
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
        
        # Get training name
        training_name = get_training_name_from_zwo(zwo_path)
        print(f"🏷️  Training name: {training_name}")
        
        # Create training folder
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
        
        # Save as Excel (this will be skipped if openpyxl is not available)
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
