# interval_finder.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class IntervalFinder:
    """
    Finds actual intervals in training data based on abstract ZWO intervals.
    Assumes training starts at the same time as guided workout.
    """
    
    def __init__(self, df: pd.DataFrame, timestamp_col: str = 'timestamp'):
        self.df = df
        self.timestamp_col = timestamp_col
        self.found_intervals = []
        
    def find_intervals(self, abstract_intervals: List[Dict]) -> List[Dict]:
        """
        Map abstract intervals to actual training data timestamps.
        
        Args:
            abstract_intervals: Intervals from ZWO parser with start_time, end_time
            
        Returns:
            List of intervals with actual start/end indices and timestamps
        """
        print("🔍 Finding intervals in training data...")
        
        found_intervals = []
        
        for i, interval in enumerate(abstract_intervals):
            found_interval = self._find_single_interval(interval, i)
            if found_interval:
                found_intervals.append(found_interval)
        
        self.found_intervals = found_intervals
        self._print_finding_summary(found_intervals)
        return found_intervals
    
    def _find_single_interval(self, abstract_interval: Dict, index: int) -> Optional[Dict]:
        """Find a single interval in the training data"""
        start_time = abstract_interval['start_time']
        end_time = abstract_interval['end_time']
        
        # Find rows within this time range
        mask = (self.df[self.timestamp_col] >= start_time) & (self.df[self.timestamp_col] <= end_time)
        interval_data = self.df[mask]
        
        if interval_data.empty:
            print(f"⚠️  No data found for interval {index + 1} ({start_time}-{end_time}s)")
            return None
        
        # Get the actual start and end indices in the dataframe
        start_idx = interval_data.index[0]
        end_idx = interval_data.index[-1]
        
        # Create enhanced interval info
        found_interval = {
            **abstract_interval,  # Include all original data
            'interval_index': index + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'actual_start_time': interval_data[self.timestamp_col].iloc[0],
            'actual_end_time': interval_data[self.timestamp_col].iloc[-1],
            'data_points': len(interval_data),
            'duration_actual': interval_data[self.timestamp_col].iloc[-1] - interval_data[self.timestamp_col].iloc[0]
        }
        
        return found_interval
    
    def _print_finding_summary(self, found_intervals: List[Dict]) -> None:
        """Print summary of interval finding"""
        total_intervals = len(found_intervals)
        total_data_points = sum(interval['data_points'] for interval in found_intervals)
        coverage_pct = (total_data_points / len(self.df)) * 100
        
        print(f"✅ Found {total_intervals} intervals with {total_data_points} data points")
        print(f"📊 Interval coverage: {coverage_pct:.1f}% of training data")
        
        # Print interval types found
        interval_types = {}
        for interval in found_intervals:
            interval_type = interval.get('type', 'unknown')
            interval_types[interval_type] = interval_types.get(interval_type, 0) + 1
        
        print(f"📋 Interval types: {interval_types}")

def find_training_intervals(df: pd.DataFrame, abstract_intervals: List[Dict]) -> List[Dict]:
    """
    Convenience function to find intervals in training data.
    
    Args:
        df: Training DataFrame with timestamp column
        abstract_intervals: Intervals from ZWO parser
        
    Returns:
        List of intervals mapped to actual training data
    """
    finder = IntervalFinder(df)
    return finder.find_intervals(abstract_intervals)
