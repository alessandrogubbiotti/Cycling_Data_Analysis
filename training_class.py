import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
from datetime import datetime

# Import your existing modules
from interval_finder import IntervalFinder, find_training_intervals
from statistics import StatisticsComputer, compute_training_statistics
from plotter import TrainingPlotter
from training_enhancer import TrainingEnhancer, enhance_training_data, safe_float_convert


def parse_timestamp(timestamp_str):
    """
    Parse various timestamp formats to relative seconds from workout start.
    
    Handles:
    - ISO format: "2025-11-02T11:34:42.123Z"
    - Custom formats
    - Already numeric timestamps
    """
    if pd.isna(timestamp_str) or timestamp_str is None:
        return np.nan
    
    # If it's already a number, assume it's seconds
    try:
        if isinstance(timestamp_str, (int, float)):
            return float(timestamp_str)
        elif str(timestamp_str).replace('.', '').replace('-', '').isdigit():
            return float(timestamp_str)
    except:
        pass
    
    # Try to parse as datetime string
    try:
        # Handle ISO format with T and Z
        if 'T' in str(timestamp_str):
            if 'Z' in str(timestamp_str):
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(timestamp_str)
        # Handle other common formats
        elif ' ' in str(timestamp_str):
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        else:
            # Try your custom format or other common formats
            for fmt in ['%Y-%m-%d-%H-%M-%S', '%Y%m%d%H%M%S']:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    break
                except:
                    continue
            else:
                raise ValueError(f"Unknown timestamp format: {timestamp_str}")
        
        return dt.timestamp()  # Convert to UNIX timestamp
        
    except Exception as e:
        print(f"⚠️ Could not parse timestamp {timestamp_str}: {e}")
        return np.nan


@dataclass
class TrainingMetadata:
    """Structured training metadata"""
    training_name: str
    training_type: str
    date: str
    duration: float
    ftp: Optional[float] = None
    athlete_name: Optional[str] = None
    nutrition: Optional[Dict] = None
    hydration: Optional[Dict] = None
    effort: Optional[int] = None
    notes: Optional[str] = None
    file_paths: Optional[Dict] = None


class Training:
    """
    Main Training class that integrates all analysis components.
    Represents a single cycling training session with comprehensive analysis.
    """
    
    def __init__(self, training_folder: Path):
        """
        Initialize Training from a training folder.
        
        Args:
            training_folder: Path to the training folder containing parsed data
        """
        self.training_folder = Path(training_folder)
        self.training_name = self.training_folder.name
        
        # Core components
        self.metadata = {}
        self.data = None
        self.intervals = []
        self.statistics = {}
        self.plotter = None
        
        # Enhanced data
        self.enhanced_data = None
        self.found_intervals = []
        
        # File paths
        self._file_paths = {
            'metadata': self.training_folder / 'metadata.json',
            'time_series': self.training_folder / 'time_series.csv',
            'session_metrics': self.training_folder / 'session_metrics.csv',
            'zwo_intervals': self.training_folder / 'zwo_intervals.json'
        }
        
        print(f"🚴 Initializing Training: {self.training_name}")
    
    def load_metadata(self) -> Dict:
        """
        Load training metadata from metadata.json
        
        Returns:
            Dictionary containing training metadata
        """
        metadata_path = self._file_paths['metadata']
        
        if not metadata_path.exists():
            warnings.warn(f"Metadata file not found: {metadata_path}")
            self.metadata = {}
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Loaded metadata for {self.training_name}")
            return self.metadata
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            self.metadata = {}
            return {}
    
    def _process_timestamps(self):
        """Convert timestamps to relative seconds from workout start"""
        # Parse all timestamps
        parsed_timestamps = self.data['timestamp'].apply(parse_timestamp)
        
        # Find the start time (minimum timestamp)
        start_time = parsed_timestamps.min()
        
        # Convert to relative seconds from start
        self.data['timestamp'] = parsed_timestamps - start_time
        
        # Fill any NaN values
        self.data['timestamp'] = self.data['timestamp'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Convert to float to ensure numeric type
        self.data['timestamp'] = self.data['timestamp'].astype(float)
    
    def read_data(self) -> pd.DataFrame:
        """
        Load training time series data from CSV and ensure timestamps are in relative seconds.
        
        Returns:
            DataFrame with training time series data
        """
        csv_path = self._file_paths['time_series']
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Time series data not found: {csv_path}")
        
        try:
            self.data = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(self.data)} raw data points from {csv_path}")
            
            # Ensure timestamp column exists and is properly formatted
            if 'timestamp' not in self.data.columns:
                raise ValueError("No 'timestamp' column found in time series data")
            
            # Parse timestamps to relative seconds
            self._process_timestamps()
            
            print(f"✅ Processed timestamps: {self.data['timestamp'].min():.1f} - {self.data['timestamp'].max():.1f}s ({self.data['timestamp'].max()/60:.1f}min)")
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def find_intervals(self) -> List[Dict]:
        """
        Find intervals in training data using ZWO intervals as guidance.
        
        Returns:
            List of found intervals with actual timestamps and indices
        """
        if self.data is None:
            self.read_data()
        
        # Load ZWO intervals if available
        zwo_intervals = self._load_zwo_intervals()
        
        if not zwo_intervals:
            print("⚠️  No ZWO intervals found. Interval analysis will be limited.")
            return []
        
        # Find intervals in actual data
        self.found_intervals = find_training_intervals(self.data, zwo_intervals)
        
        print(f"✅ Found {len(self.found_intervals)} intervals in training data")
        return self.found_intervals
    
    def _load_zwo_intervals(self) -> List[Dict]:
        """
        Load ZWO intervals from JSON file.
        
        Returns:
            List of abstract intervals from ZWO file
        """
        zwo_path = self._file_paths['zwo_intervals']
        
        if not zwo_path.exists():
            print(f"⚠️  ZWO intervals file not found: {zwo_path}")
            return []
        
        try:
            with open(zwo_path, 'r') as f:
                zwo_intervals = json.load(f)
            
            print(f"✅ Loaded {len(zwo_intervals)} ZWO intervals")
            return zwo_intervals
            
        except Exception as e:
            print(f"❌ Error loading ZWO intervals: {e}")
            return []
    
    def enhance_data(self) -> pd.DataFrame:
        """
        Enhance training data with target power, cadence, and interval information.
        
        Returns:
            Enhanced DataFrame with target values and interval metadata
        """
        if self.data is None:
            self.read_data()
        
        if not self.found_intervals:
            self.find_intervals()
        
        # Load metadata for FTP information
        if not self.metadata:
            self.load_metadata()
        
        # Enhance data with intervals
        enhancer = TrainingEnhancer(self.data, self.metadata)
        self.enhanced_data = enhancer.enhance_with_intervals(self.found_intervals)
        
        print(f"✅ Enhanced data with {len(self.found_intervals)} intervals")
        return self.enhanced_data
    
    def compute_statistics(self, save: bool = True) -> Dict:
        """
        Compute comprehensive statistics for training and intervals.
        
        Args:
            save: Whether to save statistics to files
            
        Returns:
            Dictionary containing interval and overall statistics
        """
        if self.enhanced_data is None:
            self.enhance_data()
        
        # Compute statistics
        self.statistics = compute_training_statistics(
            self.enhanced_data, 
            self.found_intervals
        )
        
        # Save statistics if requested
        if save:
            stats_dir = self.training_folder / "statistics"
            computer = StatisticsComputer(self.enhanced_data, self.found_intervals)
            computer.save_statistics(stats_dir)
        
        print(f"✅ Computed statistics for {len(self.found_intervals)} intervals")
        return self.statistics
    
    def create_plots(self, output_dir: Path = None) -> None:
        """
        Create comprehensive training visualizations.
        
        Args:
            output_dir: Directory to save plots (defaults to training_folder/plots)
        """
        if self.enhanced_data is None:
            self.enhance_data()
        
        if not self.statistics:
            self.compute_statistics()
        
        # Set output directory
        if output_dir is None:
            output_dir = self.training_folder / "plots"
        else:
            output_dir = Path(output_dir)
        
        # Load ZWO intervals for plotting
        zwo_intervals = self._load_zwo_intervals()
	# Create plots using TrainingPlotter class
        plotter = TrainingPlotter(
	    enhanced_df=self.enhanced_data,
	    interval_stats=self.statistics.get('interval_statistics', []),
	    overall_stats=self.statistics.get('overall_statistics', {}),
	    zwo_intervals=zwo_intervals,
	    metadata=self.metadata
	)
        plotter.create_all_plots(output_dir)        
        
        print(f"✅ Created plots in {output_dir}")
    
    def analyze(self, create_plots: bool = True) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            create_plots: Whether to create visualizations
            
        Returns:
            Comprehensive analysis results
        """
        print(f"🔬 Starting comprehensive analysis for {self.training_name}")
        
        # Run analysis pipeline
        self.load_metadata()
        self.read_data()
        self.find_intervals()
        self.enhance_data()
        results = self.compute_statistics()
        
        if create_plots:
            self.create_plots()
        
        print(f"✅ Completed analysis for {self.training_name}")
        return results
    
    def get_summary(self) -> Dict:
        """
        Get training summary statistics.
        
        Returns:
            Dictionary with key training metrics
        """
        if not self.statistics:
            self.compute_statistics(save=False)
        
        overall_stats = self.statistics.get('overall_statistics', {})
        interval_stats = self.statistics.get('interval_statistics', [])
        
        summary = {
            'training_name': self.training_name,
            'date': self.metadata.get('date', 'Unknown'),
            'total_duration': overall_stats.get('total_duration', 0),
            'total_records': overall_stats.get('total_records', 0),
            'interval_count': len(interval_stats),
            'power_avg': overall_stats.get('power_overall_avg', 0),
            'cadence_avg': overall_stats.get('cadence_overall_avg', 0),
            'hr_avg': overall_stats.get('hr_overall_avg', 0),
            'normalized_power': overall_stats.get('normalized_power', 0)
        }
        
        return summary
    
    def validate_data(self) -> Tuple[bool, List[str]]:
        """
        Validate training data integrity.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required files
        required_files = ['metadata', 'time_series']
        for file_type in required_files:
            if not self._file_paths[file_type].exists():
                issues.append(f"Missing required file: {file_type}")
        
        # Check data quality if available
        if self.data is not None:
            if len(self.data) == 0:
                issues.append("Time series data is empty")
            
            # Check for required columns
            required_columns = ['timestamp']
            for col in required_columns:
                if col not in self.data.columns:
                    issues.append(f"Missing required column: {col}")
            
            # Check for NaN values in critical columns
            critical_columns = ['power', 'cadence', 'heart_rate']
            for col in critical_columns:
                if col in self.data.columns and self.data[col].isna().all():
                    issues.append(f"All values are NaN in column: {col}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def export_analysis(self, output_dir: Path = None) -> Path:
        """
        Export complete analysis to a structured directory.
        
        Args:
            output_dir: Directory for export (defaults to training_folder/analysis_export)
            
        Returns:
            Path to export directory
        """
        if output_dir is None:
            output_dir = self.training_folder / "analysis_export"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Ensure we have all analysis components
        if self.enhanced_data is None:
            self.enhance_data()
        if not self.statistics:
            self.compute_statistics()
        
        # Export enhanced data
        enhanced_data_path = output_dir / "enhanced_training_data.csv"
        self.enhanced_data.to_csv(enhanced_data_path, index=False)
        
        # Export intervals
        intervals_path = output_dir / "found_intervals.json"
        with open(intervals_path, 'w') as f:
            json.dump(self.found_intervals, f, indent=2)
        
        # Export statistics
        stats_path = output_dir / "training_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.statistics, f, indent=2, default=str)
        
        # Export summary
        summary_path = output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        print(f"✅ Analysis exported to {output_dir}")
        return output_dir


# Convenience functions
def load_training(training_folder: Path) -> Training:
    """
    Convenience function to load and validate a training.
    
    Args:
        training_folder: Path to training folder
        
    Returns:
        Training object
    """
    training = Training(training_folder)
    
    # Validate data
    is_valid, issues = training.validate_data()
    if not is_valid:
        print("⚠️  Training data validation issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    return training

def analyze_training(training_folder: Path, create_plots: bool = True) -> Dict:
    """
    Convenience function for one-shot training analysis.
    
    Args:
        training_folder: Path to training folder
        create_plots: Whether to create visualizations
        
    Returns:
        Analysis results
    """
    training = load_training(training_folder)
    return training.analyze(create_plots=create_plots)
