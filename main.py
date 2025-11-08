#!/usr/bin/env python3
"""
Cycling Data Analysis System - Main Interface
Clean version using the new Training class
"""

import sys
from pathlib import Path

# Import the new Training class
from training_class import Training, load_training, analyze_training

# Import UI helpers
from ui_helpers import choose, yes_no, get_athletes, get_months, get_trainings_by_month, follow_symlink_to_parsed_data


def main():
    print("🚴‍♂️ CYCLING TRAINING ANALYSIS SUITE")
    print("=" * 50)
    print("📁 Using new Training class for analysis")
    print("🎯 Automated interval detection and statistics")
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
    
    # Step 4: Get the actual training path
    training_path = follow_symlink_to_parsed_data(athlete, month, training)
    if not training_path:
        print(f"❌ Could not find training data for {training}")
        return
    
    print(f"📁 Training path: {training_path}")
    
    # Step 5: Load and analyze using new Training class
    try:
        print("\n📥 Loading training with new Training class...")
        training_obj = Training(training_path)
        
        # Validate data
        is_valid, issues = training_obj.validate_data()
        if not is_valid:
            print("⚠️ Training data validation issues:")
            for issue in issues:
                print(f"   - {issue}")
            
            if not yes_no("Continue anyway?"):
                return
        
        # Step 6: Analysis options
        while True:
            print(f"\n🎯 ANALYSIS OPTIONS for {training}:")
            options = [
                "Quick analysis (full pipeline)",
                "Step-by-step analysis", 
                "Create plots only",
                "Export analysis results",
                "Show training summary",
                "Change training",
                "Exit"
            ]
            
            choice = choose("Choose analysis type:", options)
            
            if choice == "Exit":
                break
                
            elif choice == "Change training":
                return main()  # Restart the process
            
            elif choice == "Quick analysis (full pipeline)":
                print("🚀 Running complete analysis pipeline...")
                results = training_obj.analyze(create_plots=True)
                print("✅ Analysis complete!")
                
                # Show summary
                summary = training_obj.get_summary()
                print(f"\n📊 TRAINING SUMMARY:")
                for key, value in summary.items():
                    print(f"   {key}: {value}")
            
            elif choice == "Step-by-step analysis":
                print("\n🔬 Step-by-step analysis:")
                print("1. Loading metadata...")
                training_obj.load_metadata()
                
                print("2. Reading data...")
                training_obj.read_data()
                
                print("3. Finding intervals...")
                intervals = training_obj.find_intervals()
                print(f"   Found {len(intervals)} intervals")
                
                print("4. Enhancing data...")
                training_obj.enhance_data()
                
                print("5. Computing statistics...")
                training_obj.compute_statistics()
                
                print("6. Creating plots...")
                training_obj.create_plots()
                
                print("✅ Step-by-step analysis complete!")
            
            elif choice == "Create plots only":
                print("📈 Creating plots...")
                training_obj.create_plots()
                print("✅ Plots created!")
            
            elif choice == "Export analysis results":
                print("💾 Exporting analysis...")
                export_path = training_obj.export_analysis()
                print(f"✅ Analysis exported to: {export_path}")
            
            elif choice == "Show training summary":
                summary = training_obj.get_summary()
                print(f"\n📊 TRAINING SUMMARY:")
                for key, value in summary.items():
                    print(f"   {key}: {value}")
            
            if not yes_no("Continue with this training?"):
                if yes_no("Analyze another training?"):
                    return main()
                else:
                    break
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 Analysis complete for {training}!")


def batch_analysis():
    """Batch analysis mode for multiple trainings"""
    print("🔁 BATCH ANALYSIS MODE")
    
    athletes = get_athletes()
    if not athletes:
        return
    
    athlete = choose("Select athlete:", athletes)
    
    months = get_months(athlete)
    if not months:
        return
    
    # Analyze all trainings in selected months
    for month in months:
        print(f"\n📅 Analyzing month: {month}")
        trainings = get_trainings_by_month(athlete, month)
        
        for training in trainings:
            print(f"  🚴 Analyzing: {training}")
            training_path = follow_symlink_to_parsed_data(athlete, month, training)
            
            if training_path:
                try:
                    # Quick analysis for batch mode
                    training_obj = Training(training_path)
                    results = training_obj.analyze(create_plots=True)
                    summary = training_obj.get_summary()
                    print(f"    ✅ Done: {summary['power_avg']:.0f}W avg power, {summary['interval_count']} intervals")
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
            else:
                print(f"    ❌ Could not find data")
    
    print("🎉 Batch analysis complete!")


def cli_interface():
    """Command line interface"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print("""
Cycling Data Analysis System

Usage:
  python main.py                    # Interactive mode
  python main.py --batch           # Batch analysis mode

Examples:
  python main.py
  python main.py --batch
            """)
            return
        
        if sys.argv[1] == "--batch":
            batch_analysis()
            return
    
    # Default to interactive mode
    main()


if __name__ == "__main__":
    cli_interface()
