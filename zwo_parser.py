import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

def safe_float_convert(value, default=0.0):
    """Safely convert value to float with error handling"""
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

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
                
                interval_data = {
                    'start_time': current_time,
                    'end_time': current_time + duration,
                    'duration': duration,
                    'type': element.tag.lower(),
                    'target_power_pct': target_power_pct,
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
                    on_power = safe_float_convert(element.get('PowerOn', target_power_pct))
                    off_power = safe_float_convert(element.get('PowerOff', power_low))
                    
                    for i in range(repeat):
                        # On interval
                        intervals.append({
                            **interval_data,
                            'start_time': current_time,
                            'end_time': current_time + on_duration,
                            'duration': on_duration,
                            'type': 'interval_on',
                            'target_power_pct': on_power,
                            'description': f'Interval {i+1}/{repeat} (On)'
                        })
                        current_time += on_duration
                        
                        # Off interval (if not last)
                        if off_duration > 0 and i < repeat - 1:
                            intervals.append({
                                **interval_data,
                                'start_time': current_time,
                                'end_time': current_time + off_duration,
                                'duration': off_duration,
                                'type': 'interval_off',
                                'target_power_pct': off_power,
                                'description': f'Interval {i+1}/{repeat} (Off)'
                            })
                            current_time += off_duration
                else:
                    intervals.append(interval_data)
                    current_time += duration
        
        print(f"✅ Parsed {len(intervals)} intervals from ZWO file")
        return intervals
        
    except Exception as e:
        print(f"❌ Error parsing ZWO file: {e}")
        raise

def get_zwo_workout_summary(intervals):
    """Get summary of ZWO workout"""
    if not intervals:
        return "No intervals found"
    
    total_duration = intervals[-1]['end_time']
    interval_types = {}
    for interval in intervals:
        interval_types[interval['type']] = interval_types.get(interval['type'], 0) + 1
    
    summary = f"ZWO Workout Summary:\n"
    summary += f"Total duration: {total_duration/60:.1f} minutes\n"
    summary += f"Number of intervals: {len(intervals)}\n"
    summary += f"Interval types: {', '.join([f'{k}({v})' for k, v in interval_types.items()])}\n"
    
    return summary
