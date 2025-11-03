# main.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import modular components
from zwo_parser import parse_zwo_workout, get_zwo_workout_summary
from interval_finder import IntervalFinder, find_training_intervals
from training_enhancer import TrainingEnhancer, load_metadata, get_ftp_from_metadata
from statistics import StatisticsComputer, compute_training_statistics
from plotter import TrainingPlotter, create_training_plots

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

def debug_data_structure(data):
    """Debug function to check data structure before plotting"""
    print("\n🔍 DEBUG DATA STRUCTURE:")
    print(f"📊 DataFrame shape: {data['time_series'].shape}")
    print(f"📋 Columns: {list(data['time_series'].columns)}")
    
    # Check timestamp
    if 'timestamp' in data['time_series'].columns:
        ts = data['time_series']['timestamp']
        print(f"🕐 Timestamp range: {ts.min():.1f} - {ts.max():.1f}s ({ts.max()/60:.1f}min)")
    
    # Check power data
    if 'power' in data['time_series'].columns:
        power = data['time_series']['power']
        print(f"⚡ Power range: {power.min():.1f} - {power.max():.1f}W")
    
    # Check target power
    if 'target_power' in data['time_series'].columns:
        target_power = data['time_series']['target_power']
        non_nan = target_power.notna().sum()
        print(f"🎯 Target power: {non_nan} non-NaN values")
        if non_nan > 0:
            print(f"   Target range: {target_power.min():.1f} - {target_power.max():.1f}W")
    
    # Check intervals
    if 'zwo_parsed' in data:
        print(f"📈 ZWO intervals: {len(data['zwo_parsed'])}")
        for i, interval in enumerate(data['zwo_parsed'][:5]):  # Show first 5
            print(f"   {i+1}: {interval.get('type', 'unknown')} - {interval.get('start_time', 0):.0f}-{interval.get('end_time', 0):.0f}s @ {interval.get('target_power', 0):.0f}W")

def load_training_data(athlete_name, month, training_name):
    """Load training data and enhance with ZWO targets using modular approach"""
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
    
    # Load time series
    ts_path = parsed_dir / "time_series.csv"
    if ts_path.exists():
        print(f"📁 Loading CSV from: {ts_path}")
        df = pd.read_csv(ts_path)
        
        # Handle timestamp conversion
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
        
        # Convert numeric columns to float
        numeric_columns = ['heart_rate', 'power', 'cadence', 'speed', 'altitude', 'distance', 'temperature']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: safe_float_convert(x, np.nan))
                non_nan_count = df[col].notna().sum()
                if non_nan_count > 0:
                    print(f"✅ Loaded {col}: {non_nan_count} non-NaN values")
        
        # Fill NaN values
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
        
        # Check for FTP in metadata
        ftp = get_ftp_from_metadata(data['metadata'])
        if ftp:
            print(f"✅ Using FTP from metadata: {ftp}W")
        else:
            print("⚠️ No FTP found in metadata - power values will be shown as percentages")
        
        print("✅ Loaded metadata")
    else:
        print("❌ Metadata not found")
        data['metadata'] = {}
    
    # Load auto-detected intervals if they exist
    auto_int_path = parsed_dir / "intervals.json"
    if auto_int_path.exists():
        with open(auto_int_path, 'r') as f:
            data['auto_intervals'] = json.load(f)
        # Ensure interval times are floats
        for interval in data['auto_intervals']:
            interval['start_time'] = safe_float_convert(interval.get('start_time', 0))
            interval['end_time'] = safe_float_convert(interval.get('end_time', 0))
        print(f"✅ Loaded {len(data['auto_intervals'])} auto-detected intervals")
    
    # === ENHANCE WITH ZWO TARGETS USING MODULAR APPROACH ===
    training_template = data['metadata'].get('training_template')
    if training_template:
        zwo_file = Path("Trainings") / training_template
        if zwo_file.exists():
            print(f"🎯 Enhancing training with ZWO targets...")
            
            # Step 1: Parse ZWO file
            abstract_intervals = parse_zwo_workout(zwo_file)
            print(get_zwo_workout_summary(abstract_intervals))
            
            if abstract_intervals:
                # Step 2: Find intervals in training data
                finder = IntervalFinder(data['time_series'])
                found_intervals = finder.find_intervals(abstract_intervals)
                
                if found_intervals:
                    # Step 3: Enhance training data with targets
                    enhancer = TrainingEnhancer(data['time_series'], data['metadata'])
                    enhanced_df = enhancer.enhance_with_intervals(found_intervals)
                    
                    data['time_series'] = enhanced_df
                    data['zwo_parsed'] = found_intervals
                    data['abstract_intervals'] = abstract_intervals
                    
                    # Apply ramp target power for linear progression
                    print("🔄 Applying linear target power for ramps, warmups, and cooldowns...")
                    plotter = TrainingPlotter(enhanced_df, metadata=data['metadata'])
                    data['time_series'] = plotter.add_ramp_target_power(enhanced_df, found_intervals)
                    
                    print("✅ Successfully enhanced training data with ZWO targets")
                else:
                    print("⚠️ No intervals found in training data")
            else:
                print("⚠️ No intervals parsed from ZWO file")
        else:
            print(f"❌ ZWO file not found: {zwo_file}")
    else:
        print("ℹ️ No training template found in metadata")
    
    return data

# === Analysis Workflows ===
def run_quick_analysis(data):
    """Run a quick analysis with basic plots"""
    print("⚡ Running quick analysis...")
    
    # Create basic plots directory
    plots_dir = data['path'] / "quick_analysis"
    plots_dir.mkdir(exist_ok=True)
    
    # Use the new plotter with proper parameters
    plotter = TrainingPlotter(
        enhanced_df=data['time_series'],
        metadata=data.get('metadata', {}),
        zwo_intervals=data.get('zwo_parsed', [])
    )
    plotter.create_all_plots(plots_dir)
    
    print(f"✅ Quick analysis complete! Check {plots_dir}")

def run_zwo_analysis(data):
    """Run complete ZWO-guided analysis"""
    if 'zwo_parsed' not in data or not data['zwo_parsed']:
        print("❌ No ZWO intervals found for analysis")
        return None
    
    print("🎯 Running ZWO-guided analysis...")
    
    # Step 1: Compute statistics
    computer = StatisticsComputer(data['time_series'], data['zwo_parsed'])
    results = computer.compute_all_statistics()
    
    # Step 2: Create plots with proper parameters
    plots_dir = data['path'] / "zwo_analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    create_training_plots(
        enhanced_df=data['time_series'], 
        statistics_results=results, 
        output_dir=plots_dir,
        zwo_intervals=data['zwo_parsed'],
        metadata=data.get('metadata', {})
    )
    
    # Step 3: Save statistics
    stats_dir = data['path'] / "zwo_statistics"
    stats_dir.mkdir(exist_ok=True)
    computer.save_statistics(stats_dir)
    
    # Step 4: Save enhanced data
    enhanced_csv_path = data['path'] / "enhanced_time_series.csv"
    data['time_series'].to_csv(enhanced_csv_path, index=False)
    
    print(f"✅ ZWO analysis complete!")
    print(f"   📊 Statistics: {stats_dir}")
    print(f"   📈 Plots: {plots_dir}")
    print(f"   💾 Enhanced data: {enhanced_csv_path}")
    
    return results

def run_auto_analysis(data):
    """Run analysis on auto-detected intervals"""
    if 'auto_intervals' not in data or not data['auto_intervals']:
        print("❌ No auto-detected intervals found")
        return None
    
    print("🔍 Running auto-intervals analysis...")
    
    # Convert auto intervals to the expected format for statistics
    formatted_intervals = []
    for i, interval in enumerate(data['auto_intervals']):
        formatted_interval = {
            'interval_index': i + 1,
            'start_time': interval.get('start_time', 0),
            'end_time': interval.get('end_time', 0),
            'type': interval.get('type', 'auto'),
            'duration': interval.get('end_time', 0) - interval.get('start_time', 0),
            'start_idx': 0,  
            'end_idx': 0,    
            'actual_start_time': interval.get('start_time', 0),
            'actual_end_time': interval.get('end_time', 0),
            'data_points': 0
        }
        formatted_intervals.append(formatted_interval)
    
    # Compute basic statistics
    computer = StatisticsComputer(data['time_series'], formatted_intervals)
    results = computer.compute_all_statistics()
    
    # Create plots for auto-intervals
    plots_dir = data['path'] / "auto_analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Use the plotter for basic visualization
    plotter = TrainingPlotter(
        enhanced_df=data['time_series'],
        interval_stats=results.get('interval_statistics', []),
        overall_stats=results.get('overall_statistics', {}),
        metadata=data.get('metadata', {})
    )
    
    # Create basic plots
    plotter.plot_basic_training({
        'time_series': data['time_series'],
        'metadata': data.get('metadata', {}),
        'training': data['training']
    }, plots_dir / "auto_intervals_basic.png")
    
    # Create statistical plots
    if results.get('interval_statistics'):
        plotter.plot_interval_compliance(plots_dir)
        plotter.plot_interval_performance(plots_dir)
        plotter.plot_compliance_heatmap(plots_dir)
    
    # Save statistics
    stats_dir = data['path'] / "auto_statistics"
    stats_dir.mkdir(exist_ok=True)
    computer.save_statistics(stats_dir)
    
    print(f"✅ Auto-intervals analysis complete! Check {plots_dir}")
    return results

def run_complete_analysis(data):
    """Run all available analyses"""
    print("🚀 Running complete analysis...")
    
    results = {}
    
    # ZWO analysis if available
    if 'zwo_parsed' in data and data['zwo_parsed']:
        print("\n🎯 Running ZWO Analysis...")
        results['zwo'] = run_zwo_analysis(data)
    
    # Auto-intervals analysis if available
    if 'auto_intervals' in data and data['auto_intervals']:
        print("\n🔍 Running Auto-Intervals Analysis...")
        results['auto'] = run_auto_analysis(data)
    
    # Quick analysis always
    print("\n⚡ Running Quick Analysis...")
    run_quick_analysis(data)
    
    print("🎉 Complete analysis finished!")
    return results

# === Main UI ===
def main():
    print("🚴‍♂️ CYCLING TRAINING ANALYSIS SUITE")
    print("=" * 50)
    print("📁 Directory structure: Athlete/by_month/symlinks")
    print("⚡ Modular analysis: Interval Finder → Enhancer → Statistics → Plotter")
    print("🎯 FTP automatically loaded from metadata.json")
    print("=" * 50)
    
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
    
    # Step 4: Load and enhance data
    try:
        print("\n📥 Loading training data...")
        data = load_training_data(athlete, month, training)
        print("✅ Data loaded successfully")
        
        # Show data summary
        print(f"\n📊 DATA SUMMARY:")
        print(f"   • Records: {len(data['time_series'])}")
        print(f"   • Duration: {data['time_series']['timestamp'].max()/60:.1f} min")
        
        # Show FTP info
        ftp = get_ftp_from_metadata(data['metadata'])
        if ftp:
            print(f"   • FTP: {ftp}W")
        else:
            print(f"   • FTP: Not found in metadata")
        
        if 'auto_intervals' in data:
            print(f"   • Auto intervals: {len(data['auto_intervals'])}")
        
        if 'zwo_parsed' in data:
            print(f"   • ZWO intervals: {len(data['zwo_parsed'])}")
            target_data = data['time_series']['target_power'].notna().sum()
            print(f"   • Target power data: {target_data} points")
        
        # Optional debug
        if yes_no("Show detailed data structure?"):
            debug_data_structure(data)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Analysis options
    while True:
        print(f"\n🎯 ANALYSIS OPTIONS for {training}:")
        options = []
        
        # Always available
        options.append("Quick analysis (basic plots)")
        
        # Conditionally available
        if 'zwo_parsed' in data and data['zwo_parsed']:
            options.append("ZWO-guided analysis (target power & cadence)")
        
        if 'auto_intervals' in data and data['auto_intervals']:
            options.append("Auto-intervals analysis")
        
        if ('zwo_parsed' in data and data['zwo_parsed']) or ('auto_intervals' in data and data['auto_intervals']):
            options.append("Complete analysis (all available)")
        
        options.append("Change training")
        options.append("Exit")
        
        choice = choose("Choose analysis type:", options)
        
        if choice == "Exit":
            break
            
        elif choice == "Change training":
            return main()  # Restart the process
        
        elif choice == "Quick analysis (basic plots)":
            run_quick_analysis(data)
            
        elif choice == "ZWO-guided analysis (target power & cadence)":
            run_zwo_analysis(data)
            
        elif choice == "Auto-intervals analysis":
            run_auto_analysis(data)
            
        elif choice == "Complete analysis (all available)":
            run_complete_analysis(data)
        
        if not yes_no("Continue with this training?"):
            if yes_no("Analyze another training?"):
                return main()
            else:
                break
    
    print(f"\n🎉 Analysis complete for {training}!")
    print(f"📁 Results saved in: {data['path']}")

# === Command Line Interface ===
def cli_interface():
    """Simple CLI for batch processing"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
Usage:
  python main.py                    # Interactive mode
  python main.py --batch AthleteName Month TrainingName
  
Examples:
  python main.py --batch "John Doe" "2024-01" "interval_workout"
  python main.py --batch "Jane Smith" "2024-02" "endurance_ride"

Note: FTP is automatically loaded from metadata.json in the training directory.
            """)
            return
        
        if sys.argv[1] == "--batch" and len(sys.argv) >= 4:
            athlete = sys.argv[2]
            month = sys.argv[3]
            training = sys.argv[4]
            
            try:
                data = load_training_data(athlete, month, training)
                run_complete_analysis(data)
                print(f"✅ Batch analysis complete for {training}")
            except Exception as e:
                print(f"❌ Batch analysis failed: {e}")
            return
    
    # Default to interactive mode
    main()

if __name__ == "__main__":
    cli_interface()
