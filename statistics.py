# statistics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path

class StatisticsComputer:
    """
    Computes statistics for intervals and overall training.
    """
    
    def __init__(self, enhanced_df: pd.DataFrame, found_intervals: List[Dict]):
        self.df = enhanced_df
        self.found_intervals = found_intervals
        self.interval_stats = None
        self.overall_stats = None
        
    def compute_all_statistics(self) -> Dict[str, Any]:
        """
        Compute all statistics.
        
        Returns:
            Dictionary with 'interval_statistics' and 'overall_statistics'
        """
        print("📊 Computing training statistics...")
        
        results = {
            'interval_statistics': self.compute_interval_statistics(),
            'overall_statistics': self.compute_overall_statistics()
        }
        
        self._print_statistics_summary(results)
        return results
    
    def compute_interval_statistics(self) -> List[Dict]:
        """Compute statistics for each individual interval"""
        interval_stats = []
        
        for interval in self.found_intervals:
            stats = self._compute_single_interval_stats(interval)
            if stats:
                interval_stats.append(stats)
        
        self.interval_stats = interval_stats
        return interval_stats
    
    def _compute_single_interval_stats(self, interval: Dict) -> Optional[Dict]:
        """Compute statistics for a single interval"""
        start_idx = interval.get('start_idx')
        end_idx = interval.get('end_idx')
        
        if start_idx is None or end_idx is None:
            return None
        
        # Get interval data
        interval_data = self.df.loc[start_idx:end_idx].copy()
        if interval_data.empty:
            return None
        
        # Basic interval info
        stats = {
            'interval_index': interval.get('interval_index'),
            'type': interval.get('type', 'unknown'),
            'start_time': interval.get('actual_start_time'),
            'end_time': interval.get('actual_end_time'),
            'duration': interval.get('duration_actual'),
            'data_points': len(interval_data),
            'target_power_pct': interval.get('target_power_pct', 0),
            'target_cadence': interval.get('target_cadence', 0)
        }
        
        # Power statistics
        stats.update(self._compute_power_stats(interval_data, interval))
        
        # Cadence statistics
        stats.update(self._compute_cadence_stats(interval_data, interval))
        
        # Heart rate statistics
        stats.update(self._compute_heart_rate_stats(interval_data))
        
        return stats
    
    def _compute_power_stats(self, interval_data: pd.DataFrame, interval: Dict) -> Dict:
        """Compute power-related statistics"""
        stats = {}
        
        if 'power' in interval_data.columns and 'target_power' in interval_data.columns:
            power_actual = interval_data['power'].dropna()
            power_target = interval_data['target_power'].dropna()
            
            if len(power_actual) > 0 and len(power_target) > 0:
                actual_avg = power_actual.mean()
                target_avg = power_target.mean()
                
                stats.update({
                    'power_actual_avg': actual_avg,
                    'power_target_avg': target_avg,
                    'power_variance': actual_avg - target_avg,
                    'power_std': power_actual.std(),
                    'power_compliance_pct': self._calculate_compliance(actual_avg, target_avg),
                    'power_actual_min': power_actual.min(),
                    'power_actual_max': power_actual.max()
                })
        
        return stats
    
    def _compute_cadence_stats(self, interval_data: pd.DataFrame, interval: Dict) -> Dict:
        """Compute cadence-related statistics"""
        stats = {}
        
        if 'cadence' in interval_data.columns:
            cadence_actual = interval_data['cadence'].dropna()
            target_cadence = interval.get('target_cadence', 0)
            
            if len(cadence_actual) > 0:
                actual_avg = cadence_actual.mean()
                
                stats.update({
                    'cadence_actual_avg': actual_avg,
                    'cadence_target_avg': target_cadence,
                    'cadence_variance': actual_avg - target_cadence,
                    'cadence_std': cadence_actual.std(),
                    'cadence_compliance_pct': self._calculate_compliance(actual_avg, target_cadence) if target_cadence > 0 else 0,
                    'cadence_actual_min': cadence_actual.min(),
                    'cadence_actual_max': cadence_actual.max()
                })
        
        return stats
    
    def _compute_heart_rate_stats(self, interval_data: pd.DataFrame) -> Dict:
        """Compute heart rate statistics"""
        stats = {}
        
        if 'heart_rate' in interval_data.columns:
            hr_data = interval_data['heart_rate'].dropna()
            
            if len(hr_data) > 5:
                hr_initial = hr_data.iloc[0]
                hr_final = hr_data.iloc[-1]
                
                stats.update({
                    'hr_initial': hr_initial,
                    'hr_final': hr_final,
                    'hr_avg': hr_data.mean(),
                    'hr_drift_absolute': hr_final - hr_initial,
                    'hr_drift_relative': ((hr_final - hr_initial) / hr_initial * 100) if hr_initial > 0 else 0
                })
        
        return stats
    
    def _calculate_compliance(self, actual: float, target: float) -> float:
        """Calculate compliance percentage"""
        if target == 0:
            return 0
        return max(0, (1 - abs(actual - target) / target) * 100)
    
    def compute_overall_statistics(self) -> Dict:
        """Compute overall training statistics"""
        if self.df.empty:
            return {}
        
        stats = {
            'total_duration': self.df['timestamp'].max() - self.df['timestamp'].min(),
            'total_records': len(self.df),
            'interval_count': len(self.found_intervals)
        }
        
        # Power statistics
        if 'power' in self.df.columns:
            power_data = self.df['power'].dropna()
            if len(power_data) > 0:
                stats.update({
                    'power_overall_avg': power_data.mean(),
                    'power_overall_std': power_data.std(),
                    'normalized_power': self._calculate_normalized_power(power_data)
                })
        
        # Cadence statistics
        if 'cadence' in self.df.columns:
            cadence_data = self.df['cadence'].dropna()
            if len(cadence_data) > 0:
                stats['cadence_overall_avg'] = cadence_data.mean()
        
        # Heart rate statistics
        if 'heart_rate' in self.df.columns:
            hr_data = self.df['heart_rate'].dropna()
            if len(hr_data) > 0:
                stats.update({
                    'hr_overall_avg': hr_data.mean(),
                    'hr_overall_max': hr_data.max()
                })
        
        # Compliance statistics
        stats.update(self._compute_overall_compliance())
        
        self.overall_stats = stats
        return stats
    
    def _calculate_normalized_power(self, power_series: pd.Series, window_seconds: int = 30) -> float:
        """Calculate Normalized Power"""
        if len(power_series) == 0:
            return 0
        
        rolling_avg = power_series.rolling(window=window_seconds, center=True).mean()
        rolling_avg_clean = rolling_avg.dropna()
        
        if len(rolling_avg_clean) == 0:
            return 0
        
        fourth_power_avg = np.mean(np.power(rolling_avg_clean, 4))
        return np.power(fourth_power_avg, 0.25)
    
    def _compute_overall_compliance(self) -> Dict:
        """Compute overall compliance statistics"""
        compliance = {}
        
        if self.interval_stats:
            power_compliances = [s.get('power_compliance_pct', 0) for s in self.interval_stats if s.get('power_compliance_pct') is not None]
            cadence_compliances = [s.get('cadence_compliance_pct', 0) for s in self.interval_stats if s.get('cadence_compliance_pct') is not None]
            
            if power_compliances:
                compliance['power_compliance_avg'] = np.mean(power_compliances)
                compliance['power_compliance_std'] = np.std(power_compliances)
            
            if cadence_compliances:
                compliance['cadence_compliance_avg'] = np.mean(cadence_compliances)
                compliance['cadence_compliance_std'] = np.std(cadence_compliances)
        
        return compliance
    
    def save_statistics(self, output_dir: Path) -> None:
        """Save all statistics to files"""
        output_dir.mkdir(exist_ok=True)
        
        # Save interval statistics
        if self.interval_stats:
            interval_df = pd.DataFrame(self.interval_stats)
            interval_path = output_dir / "interval_statistics.csv"
            interval_df.to_csv(interval_path, index=False)
            print(f"💾 Interval statistics saved to: {interval_path}")
        
        # Save overall statistics
        if self.overall_stats:
            overall_path = output_dir / "overall_statistics.json"
            with open(overall_path, 'w') as f:
                json.dump(self.overall_stats, f, indent=2)
            print(f"💾 Overall statistics saved to: {overall_path}")
    
    def _print_statistics_summary(self, results: Dict) -> None:
        """Print summary of computed statistics"""
        overall = results['overall_statistics']
        interval_stats = results['interval_statistics']
        
        print(f"✅ Computed statistics for {len(interval_stats)} intervals")
        print(f"📈 Overall duration: {overall.get('total_duration', 0):.0f}s")
        
        if 'power_overall_avg' in overall:
            print(f"⚡ Average power: {overall['power_overall_avg']:.0f}W")
        
        if 'power_compliance_avg' in overall:
            print(f"🎯 Power compliance: {overall['power_compliance_avg']:.1f}%")

def compute_training_statistics(enhanced_df: pd.DataFrame, found_intervals: List[Dict], 
                              output_dir: Path = None) -> Dict:
    """
    Convenience function to compute all statistics.
    
    Args:
        enhanced_df: Enhanced training DataFrame
        found_intervals: Found intervals from IntervalFinder
        output_dir: Optional directory to save statistics
        
    Returns:
        Dictionary with all statistics
    """
    computer = StatisticsComputer(enhanced_df, found_intervals)
    results = computer.compute_all_statistics()
    
    if output_dir:
        computer.save_statistics(output_dir)
    
    return results
