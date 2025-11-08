from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class Interval:
    """
    Represents a single training interval with comprehensive metrics.
    """
    # Core identification
    interval_index: int
    interval_type: str
    start_time: float
    end_time: float
    
    # Target values
    target_power: Optional[float] = None
    target_power_pct: Optional[float] = None
    target_cadence: Optional[float] = None
    
    # Actual performance metrics
    power_avg: Optional[float] = None
    power_std: Optional[float] = None
    cadence_avg: Optional[float] = None
    hr_avg: Optional[float] = None
    hr_drift: Optional[float] = None
    
    # Compliance metrics
    power_compliance: Optional[float] = None
    cadence_compliance: Optional[float] = None
    
    # Additional metadata
    description: Optional[str] = None
    zone_class: Optional[str] = None
    power_type: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Interval duration in seconds"""
        return self.end_time - self.start_time
    
    @property
    def is_compliant(self) -> bool:
        """Check if interval meets compliance threshold"""
        threshold = 80  # 80% compliance threshold
        if self.power_compliance is not None:
            return self.power_compliance >= threshold
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interval to dictionary"""
        return {
            'interval_index': self.interval_index,
            'type': self.interval_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'target_power': self.target_power,
            'target_power_pct': self.target_power_pct,
            'target_cadence': self.target_cadence,
            'power_avg': self.power_avg,
            'power_std': self.power_std,
            'cadence_avg': self.cadence_avg,
            'hr_avg': self.hr_avg,
            'hr_drift': self.hr_drift,
            'power_compliance': self.power_compliance,
            'cadence_compliance': self.cadence_compliance,
            'description': self.description,
            'zone_class': self.zone_class,
            'power_type': self.power_type,
            'is_compliant': self.is_compliant
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interval':
        """Create Interval from dictionary"""
        return cls(
            interval_index=data.get('interval_index', 0),
            interval_type=data.get('type', 'unknown'),
            start_time=data.get('start_time', 0),
            end_time=data.get('end_time', 0),
            target_power=data.get('target_power'),
            target_power_pct=data.get('target_power_pct'),
            target_cadence=data.get('target_cadence'),
            power_avg=data.get('power_avg'),
            power_std=data.get('power_std'),
            cadence_avg=data.get('cadence_avg'),
            hr_avg=data.get('hr_avg'),
            hr_drift=data.get('hr_drift'),
            power_compliance=data.get('power_compliance'),
            cadence_compliance=data.get('cadence_compliance'),
            description=data.get('description'),
            zone_class=data.get('zone_class'),
            power_type=data.get('power_type')
        )


class IntervalCollection:
    """
    Collection of intervals with aggregate statistics and analysis methods.
    """
    
    def __init__(self, intervals: List[Interval] = None):
        self.intervals = intervals or []
    
    def add_interval(self, interval: Interval) -> None:
        """Add an interval to the collection"""
        self.intervals.append(interval)
    
    def get_by_type(self, interval_type: str) -> List[Interval]:
        """Get all intervals of a specific type"""
        return [interval for interval in self.intervals 
                if interval.interval_type == interval_type]
    
    def get_compliant_intervals(self) -> List[Interval]:
        """Get all intervals that meet compliance threshold"""
        return [interval for interval in self.intervals 
                if interval.is_compliant]
    
    def get_power_intervals(self) -> List[Interval]:
        """Get intervals with power targets"""
        return [interval for interval in self.intervals 
                if interval.target_power is not None]
    
    def compute_aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics for all intervals"""
        if not self.intervals:
            return {}
        
        power_intervals = self.get_power_intervals()
        
        stats = {
            'total_intervals': len(self.intervals),
            'total_duration': sum(interval.duration for interval in self.intervals),
            'interval_types': list(set(interval.interval_type for interval in self.intervals)),
            'compliance_rate': len(self.get_compliant_intervals()) / len(self.intervals) * 100,
        }
        
        if power_intervals:
            stats.update({
                'avg_power_compliance': np.mean([interval.power_compliance 
                                               for interval in power_intervals 
                                               if interval.power_compliance]),
                'avg_cadence_compliance': np.mean([interval.cadence_compliance 
                                                 for interval in self.intervals 
                                                 if interval.cadence_compliance]),
                'total_power_intervals': len(power_intervals)
            })
        
        return stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert intervals to pandas DataFrame"""
        if not self.intervals:
            return pd.DataFrame()
        
        return pd.DataFrame([interval.to_dict() for interval in self.intervals])
    
    @classmethod
    def from_statistics_dict(cls, statistics_dict: List[Dict[str, Any]]) -> 'IntervalCollection':
        """Create IntervalCollection from statistics dictionary"""
        intervals = []
        
        for interval_data in statistics_dict:
            interval = Interval.from_dict(interval_data)
            intervals.append(interval)
        
        return cls(intervals)
