### To change: The way in which the fatigue index in which each interval is found at. I don't think that the tikme is a good parameter. Maybe there should be many parameters. 
### Then I need also to implement an algorithm that authomatically finds intervals. It could be set to be sloppy in outside rides, and more precise with the indoor trainer, where I can also chech the intervals with the automated parser that parses the intervals in .zwo files 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import os
import sys
from datetime import datetime

# === UI Helper Functions ===
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

# === Data Loading Functions ===
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

def safe_float_convert(value, default=0.0):
    """Safely convert value to float with error handling"""
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def parse_iso_timestamp(timestamp_str):
    """Parse ISO 8601 timestamp to datetime object"""
    try:
        # Handle different ISO 8601 formats
        if 'T' in timestamp_str:
            if '.' in timestamp_str:
                # Format: 2025-11-02T11:34:42.123Z
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Format: 2025-11-02T11:34:42
                return datetime.fromisoformat(timestamp_str)
        else:
            # Try other formats if needed
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"⚠️ Could not parse timestamp {timestamp_str}: {e}")
        return None

def load_training_data(athlete_name, month, training_name, ftp=None):
    """Load all data for a specific training by following symlinks"""
    parsed_dir = follow_symlink_to_parsed_data(athlete_name, month, training_name)
    
    if parsed_dir is None:
        raise FileNotFoundError(f"Could not find data for {training_name}")
    
    data = {
        'path': parsed_dir,
        'athlete': athlete_name,
        'month': month,
        'training': training_name,
        'symlink_path': Path("Athlete") / athlete_name / "by_month" / month / training_name
    }
    
    # Load time series with proper type conversion
    ts_path = parsed_dir / "time_series.csv"
    if ts_path.exists():
        print(f"📁 Loading CSV from: {ts_path}")
        df = pd.read_csv(ts_path)
        
        # Debug: Show what columns we have
        print(f"📊 Available columns: {list(df.columns)}")
        
        # Handle timestamp conversion - FIXED PART
        if 'timestamp' in df.columns:
            print("🕐 Converting timestamps...")
            
            # Check if timestamp is already numeric (seconds)
            first_timestamp = str(df['timestamp'].iloc[0])
            if first_timestamp.replace('.', '').replace('-', '').isdigit():
                print("✅ Timestamps are already in numeric format (seconds)")
                df['timestamp'] = df['timestamp'].apply(lambda x: safe_float_convert(x, np.nan))
            else:
                print("🔁 Converting ISO timestamps to relative seconds...")
                # Parse ISO timestamps and convert to seconds from start
                timestamps = df['timestamp'].apply(parse_iso_timestamp)
                start_time = timestamps.min()
                df['timestamp'] = timestamps.apply(lambda x: (x - start_time).total_seconds() if x else np.nan)
        
        # Convert numeric columns to float, handling any conversion errors
        numeric_columns = ['heart_rate', 'power', 'cadence', 'speed', 'altitude', 'distance', 'temperature']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: safe_float_convert(x, np.nan))
                non_nan_count = df[col].notna().sum()
                if non_nan_count > 0:
                    print(f"✅ Loaded {col}: {non_nan_count} non-NaN values")
        
        # Fill NaN values with forward/backward fill for numeric columns only
        for col in numeric_columns + ['timestamp']:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        data['time_series'] = df
        print(f"✅ Loaded {len(df)} records with timestamp range: {df['timestamp'].min():.1f} - {df['timestamp'].max():.1f}s")
    else:
        raise FileNotFoundError(f"Time series not found: {ts_path}")
    
    # Load metadata
    meta_path = parsed_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            data['metadata'] = json.load(f)
    
    # Load auto-detected intervals
    auto_int_path = parsed_dir / "intervals.json"
    if auto_int_path.exists():
        with open(auto_int_path, 'r') as f:
            data['auto_intervals'] = json.load(f)
        # Ensure interval times are floats
        for interval in data['auto_intervals']:
            interval['start_time'] = safe_float_convert(interval.get('start_time', 0))
            interval['end_time'] = safe_float_convert(interval.get('end_time', 0))
    
    # Load ZWO intervals
    zwo_int_path = parsed_dir / "zwo_intervals.json"
    if zwo_int_path.exists():
        with open(zwo_int_path, 'r') as f:
            data['zwo_intervals'] = json.load(f)
        # Ensure interval times are floats
        for interval in data['zwo_intervals']:
            interval['start_time'] = safe_float_convert(interval.get('start_time', 0))
            interval['end_time'] = safe_float_convert(interval.get('end_time', 0))
    
    # Load ZWO file if exists - USING THE NEW PARSER
    training_template = data['metadata'].get('training_template')
    if training_template:
        zwo_file = Path("Trainings") / training_template
        if zwo_file.exists():
            data['zwo_file'] = zwo_file
            data['zwo_parsed'] = parse_zwo_to_intervals(zwo_file, ftp=ftp)
            
            # Add target power to the DataFrame
            if data['zwo_parsed']:
                data['time_series'] = add_target_power_to_df(data['time_series'], data['zwo_parsed'])
        else:
            print(f"❌ ZWO file not found: {zwo_file}")
    
    return data

# === WORKING ZWO Parser ===
def parse_zwo_to_intervals(zwo_file_path, ftp=None):
    """
    Parse ZWO file and return intervals with target power information.
    """
    try:
        print(f"🔍 Parsing ZWO file: {zwo_file_path}")
        
        if not zwo_file_path.exists():
            print(f"❌ ZWO file does not exist: {zwo_file_path}")
            return []
        
        tree = ET.parse(zwo_file_path)
        root = tree.getroot()
        
        print(f"✅ XML parsing successful")
        print(f"📋 Root element: {root.tag}")
        
        intervals = []
        current_time = 0.0  # Start at 0 seconds
        
        # Find the workout element
        workout_element = root.find('workout')
        if workout_element is None:
            print(f"❌ No <workout> element found in ZWO file")
            workout_element = root
        
        print(f"📋 Workout element found: {workout_element.tag}")
        
        # Process all workout elements
        for element in workout_element:
            if element.tag in ['Warmup', 'Cooldown', 'SteadyState', 'Ramp']:
                # Get duration in seconds from ZWO
                duration = safe_float_convert(element.get('Duration', 0))
                
                # Get power values
                power = safe_float_convert(element.get('Power', 0))
                power_low = safe_float_convert(element.get('PowerLow', 0))
                power_high = safe_float_convert(element.get('PowerHigh', 0))
                
                # Determine target power based on element type
                if element.tag == 'Warmup':
                    target_power = power_high
                elif element.tag == 'Cooldown':
                    target_power = power_low
                elif element.tag == 'SteadyState':
                    target_power = power
                elif element.tag == 'Ramp':
                    target_power = (power_low + power_high) / 2
                
                # Convert percentages to watts if FTP provided
                original_target = target_power
                if ftp and target_power <= 2.0:
                    target_power = target_power * ftp
                    power_display = f"{target_power:.0f}W (from {original_target:.1%})"
                else:
                    power_display = f"{target_power}W"
                
                # Get cadence if available
                cadence = safe_float_convert(element.get('Cadence', 0))
                
                print(f"📝 Processing: {element.tag} - {duration}s @ {power_display}")
                
                # Create interval data
                interval_data = {
                    'start_time': current_time,
                    'end_time': current_time + duration,
                    'duration': duration,
                    'type': element.tag.lower(),
                    'target_power': target_power,
                    'power': power,
                    'power_low': power_low,
                    'power_high': power_high,
                    'cadence': cadence,
                    'description': f"{element.tag} @ {power_display}",
                    'source': 'zwo'
                }
                
                intervals.append(interval_data)
                current_time += duration
        
        print(f"✅ SUCCESS: Parsed {len(intervals)} intervals from ZWO file")
        for i, interval in enumerate(intervals):
            print(f"   {i+1}: {interval['type']} - {interval['start_time']:.0f}-{interval['end_time']:.0f}s @ {interval['target_power']}W")
        
        return intervals
        
    except Exception as e:
        print(f"❌ Error parsing ZWO file: {e}")
        import traceback
        traceback.print_exc()
        return []

def add_target_power_to_df(df, intervals):
    """
    Add target_power column to DataFrame based on ZWO intervals
    """
    try:
        print(f"📁 Adding target power to DataFrame...")
        
        # Initialize target_power column with NaN
        df['target_power'] = float('nan')
        
        # Fill target_power based on intervals
        intervals_added = 0
        for interval in intervals:
            mask = (df['timestamp'] >= interval['start_time']) & (df['timestamp'] <= interval['end_time'])
            interval_rows = mask.sum()
            if interval_rows > 0:
                df.loc[mask, 'target_power'] = interval['target_power']
                intervals_added += 1
        
        print(f"✅ Added target_power to {intervals_added} intervals in DataFrame")
        
        # Show target power statistics
        target_power_data = df['target_power'].dropna()
        if len(target_power_data) > 0:
            print(f"📊 Target power range: {target_power_data.min():.1f} - {target_power_data.max():.1f}W")
        
        return df
        
    except Exception as e:
        print(f"❌ Error adding target power to DataFrame: {e}")
        return df

# === Analysis Functions ===
def get_smoothing_params(interval_type, total_duration):
    """Get smoothing parameters based on interval type"""
    total_duration = safe_float_convert(total_duration, 3600)
    
    if interval_type in ['sprint', 'power_surge']:
        return {'power': 3, 'hr': 10, 'cadence': 3}
    elif total_duration > 3600:
        return {'power': 10, 'hr': 30, 'cadence': 10}
    else:
        return {'power': 5, 'hr': 15, 'cadence': 5}

def smooth_series(series, window):
    """Smooth a pandas series"""
    if len(series) < window:
        window = max(1, len(series) // 2)
    return (series.rolling(window=window, center=True)
                  .mean()
                  .fillna(series))

def calculate_interval_metrics(interval, df, total_duration):
    """Calculate comprehensive interval metrics with HR drift analysis"""
    # Ensure we have float values for interval times
    start_time = safe_float_convert(interval.get('start_time', 0))
    end_time = safe_float_convert(interval.get('end_time', 0))
    
    interval_data = df[
        (df['timestamp'] >= start_time) & 
        (df['timestamp'] <= end_time)
    ].copy()
    
    if interval_data.empty:
        print(f"⚠️ No data found for interval {start_time}-{end_time}")
        return None
    
    # Calculate HR drift metrics
    hr_drift_metrics = calculate_hr_drift(interval_data, interval)
    
    metrics = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'time_into_workout': start_time,
        'fatigue_index': start_time / total_duration if total_duration > 0 else 0,
        'type': interval.get('type', 'unknown'),
        'source': interval.get('source', 'unknown')
    }
    
    # Power analysis
    if 'power' in interval_data.columns:
        power_data = interval_data['power'].dropna()
        if len(power_data) > 0:
            metrics.update({
                'power_mean': float(power_data.mean()),
                'power_std': float(power_data.std()),
                'power_initial': float(interval_data['power'].iloc[0]),
                'power_final': float(interval_data['power'].iloc[-1]),
                'power_min': float(power_data.min()),
                'power_max': float(power_data.max()),
                'power_cv': float(power_data.std() / power_data.mean()) if power_data.mean() > 0 else 0
            })
            
            # Trend analysis
            if len(power_data) > 10:
                time_idx = np.arange(len(power_data))
                try:
                    slope, _, r_value, _, _ = linregress(time_idx, power_data.values)
                    metrics.update({
                        'power_slope': float(slope),
                        'power_r_squared': float(r_value**2),
                        'power_trend': 'increasing' if slope > 1 else 'decreasing' if slope < -1 else 'constant'
                    })
                except:
                    metrics.update({
                        'power_slope': 0.0,
                        'power_r_squared': 0.0,
                        'power_trend': 'constant'
                    })
    
    # Add HR drift metrics
    metrics.update(hr_drift_metrics)
    
    # Cadence analysis
    if 'cadence' in interval_data.columns:
        cadence_data = interval_data['cadence'].dropna()
        if len(cadence_data) > 0:
            metrics.update({
                'cadence_mean': float(cadence_data.mean()),
                'cadence_std': float(cadence_data.std()),
                'cadence_initial': float(interval_data['cadence'].iloc[0]),
                'cadence_final': float(interval_data['cadence'].iloc[-1]),
                'cadence_drift': float(interval_data['cadence'].iloc[-1] - interval_data['cadence'].iloc[0])
            })
    
    # Efficiency metrics
    if 'power_mean' in metrics and 'hr_mean' in metrics and metrics['hr_mean'] > 0:
        metrics['efficiency_ratio'] = float(metrics['power_mean'] / metrics['hr_mean'])
        metrics['efficiency_drift'] = float(metrics['power_mean'] / metrics['hr_final']) if metrics['hr_final'] > 0 else 0
    
    return metrics

def calculate_hr_drift(interval_data, interval):
    """Calculate detailed heart rate drift metrics"""
    hr_metrics = {}
    
    if 'heart_rate' in interval_data.columns:
        hr_data = interval_data['heart_rate'].dropna()
        if len(hr_data) > 0:
            hr_initial = float(interval_data['heart_rate'].iloc[0])
            hr_final = float(interval_data['heart_rate'].iloc[-1])
            hr_drift_absolute = hr_final - hr_initial
            hr_drift_relative = (hr_drift_absolute / hr_initial * 100) if hr_initial > 0 else 0
            
            hr_metrics.update({
                'hr_mean': float(hr_data.mean()),
                'hr_std': float(hr_data.std()),
                'hr_initial': hr_initial,
                'hr_final': hr_final,
                'hr_drift_absolute': hr_drift_absolute,
                'hr_drift_relative': hr_drift_relative,
                'hr_min': float(hr_data.min()),
                'hr_max': float(hr_data.max()),
                'hr_cv': float(hr_data.std() / hr_data.mean()) if hr_data.mean() > 0 else 0
            })
            
            # HR trend analysis
            if len(hr_data) > 10:
                time_idx = np.arange(len(hr_data))
                try:
                    slope, _, r_value, _, _ = linregress(time_idx, hr_data.values)
                    hr_metrics.update({
                        'hr_slope': float(slope),
                        'hr_r_squared': float(r_value**2),
                        'hr_trend': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'
                    })
                except:
                    hr_metrics.update({
                        'hr_slope': 0.0,
                        'hr_r_squared': 0.0,
                        'hr_trend': 'stable'
                    })
            
            # HR variability (using rolling standard deviation)
            if len(hr_data) > 30:
                try:
                    hr_rolling_std = hr_data.rolling(window=min(30, len(hr_data)), center=True).std().mean()
                    hr_metrics['hr_variability'] = float(hr_rolling_std)
                except:
                    hr_metrics['hr_variability'] = 0.0
    
    return hr_metrics

# === Plotting Functions ===
def plot_basic_training(data, save_path=None):
    """Plot basic training data without intervals"""
    df = data['time_series']
    metadata = data.get('metadata', {})
    
    # Apply smoothing
    total_duration = safe_float_convert(df['timestamp'].max(), 3600)
    smooth_params = get_smoothing_params('basic', total_duration)
    
    df_smooth = df.copy()
    if 'power' in df.columns and df['power'].notna().any():
        df_smooth['power_smooth'] = smooth_series(df['power'], smooth_params['power'])
    if 'heart_rate' in df.columns and df['heart_rate'].notna().any():
        df_smooth['hr_smooth'] = smooth_series(df['heart_rate'], smooth_params['hr'])
    if 'cadence' in df.columns and df['cadence'].notna().any():
        df_smooth['cadence_smooth'] = smooth_series(df['cadence'], smooth_params['cadence'])
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    time_minutes = df['timestamp'] / 60
    
    # Power plot
    if 'power' in df_smooth.columns and 'power_smooth' in df_smooth.columns:
        axes[0].plot(time_minutes, df_smooth['power_smooth'], label='Power (smoothed)', color='red', linewidth=1.5)
        axes[0].plot(time_minutes, df['power'], label='Power (raw)', color='red', alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel('Power (W)')
        axes[0].set_title(f"Basic Training: {data['training']} - {metadata.get('training_type', '')}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Heart rate plot
    if 'heart_rate' in df_smooth.columns and 'hr_smooth' in df_smooth.columns:
        axes[1].plot(time_minutes, df_smooth['hr_smooth'], label='HR (smoothed)', color='blue', linewidth=1.5)
        axes[1].plot(time_minutes, df['heart_rate'], label='HR (raw)', color='blue', alpha=0.3, linewidth=0.5)
        axes[1].set_ylabel('Heart Rate (bpm)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # Cadence plot
    if 'cadence' in df_smooth.columns and 'cadence_smooth' in df_smooth.columns:
        axes[2].plot(time_minutes, df_smooth['cadence_smooth'], label='Cadence (smoothed)', color='green', linewidth=1.5)
        axes[2].plot(time_minutes, df['cadence'], label='Cadence (raw)', color='green', alpha=0.3, linewidth=0.5)
        axes[2].set_ylabel('Cadence (rpm)')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Basic plot saved: {save_path}")
    
    plt.show()
    return fig

def plot_with_intervals(data, interval_type, save_path=None):
    """Plot training data with intervals highlighted and HR drift analysis"""
    df = data['time_series']
    metadata = data.get('metadata', {})
    
    # Get intervals based on type
    if interval_type == 'zwo':
        intervals = data.get('zwo_parsed', []) or data.get('zwo_intervals', [])
        interval_source = "ZWO File"
    elif interval_type == 'auto':
        intervals = data.get('auto_intervals', [])
        interval_source = "Auto-detected"
    else:  # combined
        intervals = (data.get('zwo_parsed', []) or data.get('zwo_intervals', [])) + data.get('auto_intervals', [])
        interval_source = "Combined (ZWO + Auto)"
    
    # Calculate metrics for all intervals with HR drift
    all_metrics = []
    total_duration = safe_float_convert(df['timestamp'].max(), 3600)
    
    for interval in intervals:
        metrics = calculate_interval_metrics(interval, df, total_duration)
        if metrics:
            all_metrics.append(metrics)
    
    # Apply smoothing
    smooth_params = get_smoothing_params('combined', total_duration)
    df_smooth = df.copy()
    if 'power' in df.columns and df['power'].notna().any():
        df_smooth['power_smooth'] = smooth_series(df['power'], smooth_params['power'])
    if 'heart_rate' in df.columns and df['heart_rate'].notna().any():
        df_smooth['hr_smooth'] = smooth_series(df['heart_rate'], smooth_params['hr'])
    if 'cadence' in df.columns and df['cadence'].notna().any():
        df_smooth['cadence_smooth'] = smooth_series(df['cadence'], smooth_params['cadence'])
    
    # Create plot with 4 subplots to include target power
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    time_minutes = df['timestamp'] / 60
    
    # Define colors for different interval types
    interval_colors = {
        'warmup': 'cyan', 'cooldown': 'brown', 'steadystate': 'purple',
        'interval_on': 'red', 'interval_off': 'orange', 'sprint': 'yellow',
        'steady_interval': 'purple', 'auto_detected': 'pink', 'ramp': 'orange'
    }
    
    # Plot 1: Power with target power
    if 'power' in df_smooth.columns and 'power_smooth' in df_smooth.columns:
        # Actual power
        axes[0].plot(time_minutes, df_smooth['power_smooth'], label='Actual Power', color='red', linewidth=1.5)
        
        # Target power if available
        if 'target_power' in df.columns and df['target_power'].notna().any():
            axes[0].plot(time_minutes, df['target_power'], label='Target Power', 
                       color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # ZWO target power from intervals as horizontal lines
        for interval in intervals:
            if 'target_power' in interval and interval['target_power'] > 0:
                start_min = interval['start_time'] / 60
                end_min = interval['end_time'] / 60
                target_power = interval['target_power']
                axes[0].hlines(y=target_power, xmin=start_min, xmax=end_min, 
                             color='green', linestyle='-', linewidth=2, alpha=0.7,
                             label='ZWO Target' if interval == intervals[0] else "")
    
    axes[0].set_ylabel('Power (W)')
    axes[0].set_title(f"Training with {interval_source} Intervals: {data['training']} - {metadata.get('training_type', '')}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Power Delta (Actual - Target)
    if 'power' in df.columns and 'target_power' in df.columns:
        power_delta = df['power'] - df['target_power']
        power_delta_smooth = smooth_series(power_delta, smooth_params['power'])
        axes[1].plot(time_minutes, power_delta_smooth, label='Power Delta (Actual - Target)', 
                   color='purple', linewidth=1.5)
        axes[1].fill_between(time_minutes, 0, power_delta_smooth, where=power_delta_smooth >= 0, 
                           alpha=0.3, color='red', label='Above Target')
        axes[1].fill_between(time_minutes, 0, power_delta_smooth, where=power_delta_smooth < 0, 
                           alpha=0.3, color='blue', label='Below Target')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    axes[1].set_ylabel('Power Δ (W)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Heart rate
    if 'heart_rate' in df_smooth.columns and 'hr_smooth' in df_smooth.columns:
        axes[2].plot(time_minutes, df_smooth['hr_smooth'], label='Heart Rate', color='blue', linewidth=1.5)
    axes[2].set_ylabel('Heart Rate (bpm)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot 4: Cadence
    if 'cadence' in df_smooth.columns and 'cadence_smooth' in df_smooth.columns:
        axes[3].plot(time_minutes, df_smooth['cadence_smooth'], label='Cadence', color='green', linewidth=1.5)
    axes[3].set_ylabel('Cadence (rpm)')
    axes[3].set_xlabel('Time (minutes)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Highlight intervals and mark HR drift
    legend_added = set()
    for i, (interval, metrics) in enumerate(zip(intervals, all_metrics)):
        color = interval_colors.get(interval.get('type', 'unknown'), 'gray')
        start_min = safe_float_convert(interval.get('start_time', 0)) / 60
        end_min = safe_float_convert(interval.get('end_time', 0)) / 60
        
        # Highlight interval area
        for ax in axes:
            label = interval.get('type', 'interval') if interval.get('type') not in legend_added else ""
            ax.axvspan(start_min, end_min, alpha=0.2, color=color, label=label)
            if label:
                legend_added.add(interval.get('type'))
        
        # Mark HR drift with text annotation on HR plot
        if 'hr_drift_absolute' in metrics and abs(metrics['hr_drift_absolute']) > 2:
            drift_color = 'red' if metrics['hr_drift_absolute'] > 5 else 'orange' if metrics['hr_drift_absolute'] > 2 else 'green'
            axes[2].annotate(f"ΔHR: {metrics['hr_drift_absolute']:+.1f}", 
                           xy=((start_min + end_min) / 2, metrics['hr_mean']),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=drift_color, alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    # Add legends
    for ax in axes:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Interval plot saved: {save_path}")
    
    plt.show()
    
    # Create comprehensive metrics table with HR drift
    if all_metrics:
        create_metrics_table(all_metrics, interval_source, save_path)
    
    return all_metrics

def create_metrics_table(all_metrics, interval_source, save_path=None):
    """Create a comprehensive metrics table with HR drift analysis"""
    
    # Basic metrics table
    fig_basic, ax_basic = plt.subplots(figsize=(16, len(all_metrics) * 0.4 + 2))
    ax_basic.axis('tight')
    ax_basic.axis('off')
    
    basic_headers = ['#', 'Type', 'Start', 'Dur(s)', 'P_avg', 'P_std', 'P_trend', 'Fatigue']
    basic_data = []
    
    for i, metrics in enumerate(all_metrics):
        trend_symbol = {
            'increasing': '↗', 'decreasing': '↘', 'constant': '→'
        }.get(metrics.get('power_trend', 'constant'), '→')
        
        row = [
            i+1,
            metrics.get('type', 'unknown')[:10],
            f"{metrics['start_time']/60:.1f}m",
            f"{metrics['duration']:.0f}",
            f"{metrics.get('power_mean', 0):.0f}",
            f"{metrics.get('power_std', 0):.1f}",
            trend_symbol,
            f"{metrics.get('fatigue_index', 0):.2f}"
        ]
        basic_data.append(row)
    
    basic_table = ax_basic.table(
        cellText=basic_data,
        colLabels=basic_headers,
        cellLoc='center',
        loc='center'
    )
    basic_table.auto_set_font_size(False)
    basic_table.set_fontsize(8)
    basic_table.scale(1, 1.8)
    
    plt.title(f'Basic Interval Metrics - {interval_source}')
    plt.tight_layout()
    
    if save_path:
        basic_metrics_path = save_path.parent / f"{save_path.stem}_basic_metrics.png"
        plt.savefig(basic_metrics_path, dpi=150, bbox_inches='tight')
        print(f"📊 Basic metrics table saved: {basic_metrics_path}")
    
    plt.show()
    
    # HR analysis table
    fig_hr, ax_hr = plt.subplots(figsize=(16, len(all_metrics) * 0.4 + 2))
    ax_hr.axis('tight')
    ax_hr.axis('off')
    
    hr_headers = ['#', 'HR_init', 'HR_final', 'HR_avg', 'ΔHR_abs', 'ΔHR_%', 'HR_trend', 'Eff_ratio']
    hr_data = []
    
    for i, metrics in enumerate(all_metrics):
        hr_trend_symbol = {
            'increasing': '↗', 'decreasing': '↘', 'stable': '→'
        }.get(metrics.get('hr_trend', 'stable'), '→')
        
        # Color code HR drift
        hr_drift_abs = metrics.get('hr_drift_absolute', 0)
        
        row = [
            i+1,
            f"{metrics.get('hr_initial', 0):.0f}",
            f"{metrics.get('hr_final', 0):.0f}",
            f"{metrics.get('hr_mean', 0):.0f}",
            f"{hr_drift_abs:+.1f}",
            f"{metrics.get('hr_drift_relative', 0):+.1f}%",
            hr_trend_symbol,
            f"{metrics.get('efficiency_ratio', 0):.2f}"
        ]
        hr_data.append(row)
    
    hr_table = ax_hr.table(
        cellText=hr_data,
        colLabels=hr_headers,
        cellLoc='center',
        loc='center'
    )
    
    # Color cells based on HR drift
    for i in range(len(all_metrics)):
        hr_drift_abs = all_metrics[i].get('hr_drift_absolute', 0)
        cell_color = 'lightcoral' if hr_drift_abs > 5 else 'lightyellow' if hr_drift_abs > 2 else 'lightgreen'
        hr_table[(i+1, 4)].set_facecolor(cell_color)  # ΔHR_abs column
        hr_table[(i+1, 5)].set_facecolor(cell_color)  # ΔHR_% column
    
    hr_table.auto_set_font_size(False)
    hr_table.set_fontsize(8)
    hr_table.scale(1, 1.8)
    
    plt.title(f'Heart Rate Analysis - {interval_source}')
    plt.tight_layout()
    
    if save_path:
        hr_metrics_path = save_path.parent / f"{save_path.stem}_hr_metrics.png"
        plt.savefig(hr_metrics_path, dpi=150, bbox_inches='tight')
        print(f"❤️ HR metrics table saved: {hr_metrics_path}")
    
    plt.show()
    
    # Save detailed metrics to JSON
    if save_path:
        json_path = save_path.parent / f"{save_path.stem}_detailed_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"📋 Detailed metrics saved: {json_path}")
    
    return all_metrics

# === Debug Function ===
def debug_data_structure(data):
    """Debug function to understand data structure"""
    print("\n🔍 DEBUG DATA STRUCTURE:")
    print(f"📁 Path: {data['path']}")
    print(f"👤 Athlete: {data['athlete']}")
    print(f"📅 Month: {data['month']}")
    print(f"🚴 Training: {data['training']}")
    
    df = data['time_series']
    print(f"📊 DataFrame shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"📊 Data types:")
    print(df.dtypes)
    print(f"📊 First 3 rows:")
    print(df.head(3))
    
    # Check for specific columns
    for col in ['timestamp', 'power', 'heart_rate', 'cadence', 'target_power']:
        if col in df.columns:
            non_nan = df[col].notna().sum()
            if non_nan > 0:
                print(f"✅ {col}: {non_nan} non-NaN, range: {df[col].min():.1f}-{df[col].max():.1f}")
            else:
                print(f"❌ {col}: ALL VALUES ARE NaN")
        else:
            print(f"❌ {col}: COLUMN NOT FOUND")

# === Main UI ===
def main():
    print("🎯 Training Analysis UI - Browse by Month")
    print("=" * 50)
    
    # Get FTP from user
    try:
        ftp_input = input("Enter athlete's FTP in watts (or press Enter to skip): ").strip()
        if ftp_input:
            ftp = float(ftp_input)
            print(f"✅ Using FTP: {ftp}W")
        else:
            ftp = None
            print("⚠️ No FTP provided - power values will be shown as percentages")
    except ValueError:
        ftp = None
        print("⚠️ Invalid FTP - power values will be shown as percentages")
    
    # Step 1: Choose athlete
    athletes = get_athletes()
    if not athletes:
        print("❌ No athletes found in Athlete directory")
        return
    
    athlete = choose("Select athlete:", athletes)
    print(f"✅ Selected athlete: {athlete}")
    
    # Step 2: Choose month
    months = get_months(athlete)
    if not months:
        print(f"❌ No months found for {athlete}")
        return
    
    month = choose("Select month:", months)
    print(f"✅ Selected month: {month}")
    
    # Step 3: Choose training
    trainings = get_trainings_by_month(athlete, month)
    if not trainings:
        print(f"❌ No trainings found in {month}")
        return
    
    training = choose("Select training:", trainings)
    print(f"✅ Selected training: {training}")
    
    # Step 4: Load data
    try:
        data = load_training_data(athlete, month, training, ftp=ftp)
        print("✅ Data loaded successfully")
        
        # Show available data
        print(f"📊 Time series: {len(data['time_series'])} records")
        if 'auto_intervals' in data:
            print(f"🔍 Auto intervals: {len(data['auto_intervals'])}")
        if 'zwo_intervals' in data:
            print(f"📁 ZWO intervals: {len(data['zwo_intervals'])}")
        if 'zwo_parsed' in data:
            print(f"⚙️ ZWO parsed: {len(data['zwo_parsed'])}")
        if 'target_power' in data['time_series'].columns:
            target_non_nan = data['time_series']['target_power'].notna().sum()
            print(f"🎯 Target power data: {target_non_nan} non-NaN values")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Debug data structure (optional)
    if yes_no("Show debug data structure?"):
        debug_data_structure(data)
    
    # Step 6: Plot options
    while True:
        print(f"\n📈 Plot Options for {training}:")
        options = [
            "View basic training plot (no intervals)",
            "View with ZWO intervals",
            "View with auto-detected intervals", 
            "View with all intervals",
            "Save all plots",
            "Exit"
        ]
        
        choice = choose("Choose plot type:", options)
        
        if choice == "Exit":
            break
            
        save_plot = yes_no("Save this plot?")
        
        if choice == "View basic training plot (no intervals)":
            save_path = data['path'] / "basic_training_plot.png" if save_plot else None
            plot_basic_training(data, save_path)
            
        elif choice == "View with ZWO intervals":
            save_path = data['path'] / "zwo_intervals_plot.png" if save_plot else None
            plot_with_intervals(data, 'zwo', save_path)
            
        elif choice == "View with auto-detected intervals":
            save_path = data['path'] / "auto_intervals_plot.png" if save_plot else None
            plot_with_intervals(data, 'auto', save_path)
            
        elif choice == "View with all intervals":
            save_path = data['path'] / "all_intervals_plot.png" if save_plot else None
            plot_with_intervals(data, 'combined', save_path)
            
        elif choice == "Save all plots":
            print("💾 Saving all plots...")
            plot_basic_training(data, data['path'] / "basic_training_plot.png")
            plot_with_intervals(data, 'zwo', data['path'] / "zwo_intervals_plot.png")
            plot_with_intervals(data, 'auto', data['path'] / "auto_intervals_plot.png")
            plot_with_intervals(data, 'combined', data['path'] / "all_intervals_plot.png")
            print("✅ All plots saved!")
        
        if not yes_no("Continue analyzing this training?"):
            break
    
    print(f"\n🎉 Analysis complete for {training}!")
    print(f"📁 All files saved in: {data['path']}")

if __name__ == "__main__":
    main()
