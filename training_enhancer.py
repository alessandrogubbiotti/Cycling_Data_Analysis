# training_enhancer.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

def safe_float_convert(value, default=0.0):
    """Safely convert value to float with error handling"""
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def load_metadata(metadata_path):
    """Load training metadata"""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return {}

def get_ftp_from_metadata(metadata):
    """Extract FTP from metadata"""
    ftp_fields = ['ftp', 'FTP', 'functional_threshold_power', 'threshold_power']
    
    for field in ftp_fields:
        if field in metadata:
            ftp = safe_float_convert(metadata[field])
            if ftp > 0:
                print(f"✅ Found FTP in metadata: {ftp}W")
                return ftp
    
    print("❌ No valid FTP found in metadata")
    return None

class TrainingEnhancer:
    """
    Enhances training data with target power and cadence based on found intervals.
    """
    
    def __init__(self, df: pd.DataFrame, metadata: Dict = None):
        self.df = df.copy()
        self.metadata = metadata or {}
        self.ftp = get_ftp_from_metadata(metadata)

    
    def _print_enhancement_summary(self) -> None:
        """Print summary of enhancement"""
        target_power_data = self.df['target_power'].dropna()
        target_cadence_data = self.df['target_cadence'].dropna()
        
        enhancement_pct = (len(target_power_data) / len(self.df)) * 100
        
        print(f"📊 Enhancement Summary:")
        print(f"   • Total rows: {len(self.df)}")
        print(f"   • Enhanced rows: {len(target_power_data)} ({enhancement_pct:.1f}%)")
        
        if len(target_power_data) > 0:
            power_stats = f"{target_power_data.min():.1f}-{target_power_data.max():.1f}"
            print(f"   • Target power range: {power_stats}")
        
        if len(target_cadence_data) > 0:
            cadence_stats = f"{target_cadence_data.min():.1f}-{target_cadence_data.max():.1f}"
            print(f"   • Target cadence range: {cadence_stats}")
    

        
    def enhance_with_intervals(self, found_intervals: List[Dict]) -> pd.DataFrame:
        """
        Enhance DataFrame with target power and cadence from found intervals.
        
        Args:
            found_intervals: Intervals found by IntervalFinder
            
        Returns:
            Enhanced DataFrame with target columns
        """
        print("🎯 Enhancing training data with target values...")
        
        # Initialize target columns
        self.df['target_power'] = np.nan
        self.df['target_cadence'] = np.nan
        self.df['interval_type'] = None
        self.df['interval_index'] = None
        self.df['power_target_type'] = None  # 'watts' or 'percentage'
        
        intervals_filled = 0
        total_rows_enhanced = 0
        
        for interval in found_intervals:
            rows_enhanced = self._enhance_single_interval(interval)
            if rows_enhanced > 0:
                intervals_filled += 1
                total_rows_enhanced += rows_enhanced
        
        print(f"✅ Enhanced {total_rows_enhanced} rows across {intervals_filled} intervals")
        self._print_enhancement_summary()
        
        return self.df


    def _enhance_single_interval(self, interval: Dict) -> int:
        """Enhance data for a single interval with COMPREHENSIVE debugging"""
        start_idx = interval['start_idx']
        end_idx = interval['end_idx']
        
        if start_idx is None or end_idx is None or start_idx > end_idx:
            return 0
        
        # Create mask for this interval
        mask = (self.df.index >= start_idx) & (self.df.index <= end_idx)
        interval_rows = mask.sum()
        
        if interval_rows == 0:
            return 0
        
        # Get ALL power values for debugging
        raw_power_pct = interval.get('target_power_pct', 0)
        raw_power_low = interval.get('target_power_low_pct', 0)
        raw_power_high = interval.get('target_power_high_pct', 0)
        
        print(f"🔍 Interval {interval.get('type')}:")
        print(f"   Raw values - pct: {raw_power_pct}, low: {raw_power_low}, high: {raw_power_high}")
        
        # Convert from decimal to percentage
        target_power_pct = raw_power_pct * 100
        target_power_low_pct = raw_power_low * 100
        target_power_high_pct = raw_power_high * 100
        
        print(f"   After % conversion - pct: {target_power_pct}%, low: {target_power_low_pct}%, high: {target_power_high_pct}%")
        
        target_power_watts = interval.get('target_power', 0)
        
        # Determine which power value to use
        final_target_power = np.nan
        power_source = 'none'
        
        # Priority 1: Direct wattage target
        if target_power_watts > 0:
            final_target_power = target_power_watts
            power_source = 'watts_direct'
            print(f"   → Using direct wattage: {target_power_watts}W")
        
        # Priority 2: Use low/high for ramps
        elif interval.get('power_type') in ['ramp_up', 'ramp_down'] and (target_power_low_pct > 0 or target_power_high_pct > 0):
            # Ensure we have both values
            if target_power_low_pct == 0:
                target_power_low_pct = target_power_high_pct
            elif target_power_high_pct == 0:
                target_power_high_pct = target_power_low_pct
            
            avg_pct = (target_power_low_pct + target_power_high_pct) / 2.0
            final_target_power = (avg_pct / 100.0) * self.ftp
            power_source = 'ramp_avg'
            print(f"   → Using ramp average: {avg_pct}% = {final_target_power}W")
        
        # Priority 3: Main percentage with FTP
        elif self.ftp and target_power_pct > 0:
            final_target_power = (target_power_pct / 100.0) * self.ftp
            power_source = 'percentage_ftp'
            print(f"   → Converted {target_power_pct}% to {final_target_power}W (FTP: {self.ftp})")
        
        # Priority 4: Low/High average for steady intervals
        elif self.ftp and (target_power_low_pct > 0 or target_power_high_pct > 0):
            # If one is zero, use the other
            if target_power_low_pct == 0:
                target_power_low_pct = target_power_high_pct
            elif target_power_high_pct == 0:
                target_power_high_pct = target_power_low_pct
            
            avg_pct = (target_power_low_pct + target_power_high_pct) / 2.0
            final_target_power = (avg_pct / 100.0) * self.ftp
            power_source = 'low_high_avg_ftp'
            print(f"   → Using low/high average: {avg_pct}% = {final_target_power}W")
        
        # Priority 5: Raw percentage (no FTP)
        elif target_power_pct > 0:
            final_target_power = target_power_pct
            power_source = 'percentage_raw'
            print(f"   → Using percentage (no FTP): {target_power_pct}%")
        
        else:
            # No target power available
            final_target_power = np.nan
            power_source = 'none'
            print(f"   → No target power available")
        
        # Set the target power
        self.df.loc[mask, 'target_power'] = final_target_power
        self.df.loc[mask, 'power_target_type'] = power_source
        
        # Set target cadence
        target_cadence = interval.get('target_cadence', 0)
        if target_cadence > 0:
            self.df.loc[mask, 'target_cadence'] = target_cadence
            print(f"   → Target cadence: {target_cadence}rpm")
        
        # Set interval metadata
        self.df.loc[mask, 'interval_type'] = interval.get('type', 'unknown')
        self.df.loc[mask, 'interval_index'] = interval.get('interval_index')
        
        return interval_rows
    

def enhance_training_data(csv_path: Path, metadata_path: Path, found_intervals: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to enhance training data.
    
    Args:
        csv_path: Path to time_series.csv
        metadata_path: Path to metadata.json
        found_intervals: Intervals found by IntervalFinder
        
    Returns:
        Tuple of (enhanced DataFrame, metadata)
    """
    # Load data
    df = pd.read_csv(csv_path)
    metadata = load_metadata(metadata_path)
    
    # Enhance data
    enhancer = TrainingEnhancer(df, metadata)
    enhanced_df = enhancer.enhance_with_intervals(found_intervals)
    
    return enhanced_df, metadata
