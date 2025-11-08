import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import json
from scipy.stats import linregress

class TrainingPlotter:
    """ 
    Creates visualizations from enhanced training data and statistics.
    Recreates exact functionality from your working script.
    """
    
    def __init__(self, enhanced_df: pd.DataFrame, interval_stats: List[Dict] = None, 
                 overall_stats: Dict = None, smoothing_params: Dict = None, 
                 zwo_intervals: List[Dict] = None, metadata: Dict = None):
        self.df = enhanced_df
        self.interval_stats = interval_stats or []
        self.overall_stats = overall_stats or {}
        self.zwo_intervals = zwo_intervals or []
        self.metadata = metadata or {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def safe_float_convert(self, value, default=0.0):
        """Safely convert value to float with error handling"""
        if pd.isna(value) or value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_smoothing_params(self, interval_type='basic', total_duration=3600):
        """Get smoothing parameters based on interval type"""
        total_duration = self.safe_float_convert(total_duration, 3600)
        
        if interval_type in ['sprint', 'power_surge']:
            return {'power': 3, 'hr': 10, 'cadence': 3}
        elif total_duration > 3600:
            return {'power': 10, 'hr': 30, 'cadence': 10}
        else:
            return {'power': 5, 'hr': 15, 'cadence': 5}

    def smooth_series(self, series, window):
        """Smooth a pandas series"""
        if len(series) < window:
            window = max(1, len(series) // 2)
        return (series.rolling(window=window, center=True)
                      .mean()
                      .fillna(series))

    def debug_interval_data(self, intervals):
        """Comprehensive debug of interval structure"""
        print("🔍 COMPREHENSIVE INTERVAL DEBUG")
        print("=" * 60)
        
        if not intervals:
            print("❌ No intervals found!")
            return
        
        for i, interval in enumerate(intervals):
            print(f"📋 Interval {i}:")
            print(f"   Type: {interval.get('type')}")
            print(f"   Zone: {interval.get('zone_class')}")
            print(f"   Source: {interval.get('source')}")
            print(f"   Power Type: {interval.get('power_type')}")
            print(f"   Target Power %: {interval.get('target_power_pct')}")
            print(f"   Target Power Low %: {interval.get('target_power_low_pct')}")
            print(f"   Target Power High %: {interval.get('target_power_high_pct')}")
            print(f"   Start Time: {interval.get('start_time')}")
            print(f"   End Time: {interval.get('end_time')}")
            print(f"   Duration: {interval.get('duration')}")
            print(f"   Target Cadence: {interval.get('target_cadence')}")
            
            # Check all keys in the interval
            all_keys = list(interval.keys())
            print(f"   All Keys: {all_keys}")
            print("-" * 40)


    def create_all_plots(self, output_dir: Path) -> None:
        """Create all available plots matching your working script exactly"""
        output_dir.mkdir(exist_ok=True)
        print("📈 Creating training visualizations...")
        
        # Create data structure compatible with your working script
        data = {
            'time_series': self.df,
            'metadata': self.metadata,
            'training': self.metadata.get('training_name', 'Unknown Training'),
            'zwo_parsed': self.zwo_intervals,
            'zwo_intervals': self.zwo_intervals,
            'auto_intervals': [],
            'path': output_dir
        }
        
        # Create plots using your exact working functions
        self.plot_basic_training(data, output_dir / "basic_training_plot.png")
        
        if self.zwo_intervals:
            self.plot_with_intervals(data, 'zwo', output_dir / "zwo_intervals_plot.png")
        
        # Create additional statistical plots
        if self.interval_stats:
            self.plot_interval_compliance(output_dir)
            self.plot_interval_performance(output_dir)
            self.plot_compliance_heatmap(output_dir)
            
        print(f"✅ Plots saved to: {output_dir}")

    def plot_basic_training(self, data, save_path=None):
        """Plot basic training data without intervals"""
        df = data['time_series']
        metadata = data.get('metadata', {})
        
        # Apply smoothing
        total_duration = self.safe_float_convert(df['timestamp'].max(), 3600)
        smooth_params = self.get_smoothing_params('basic', total_duration)
        
        df_smooth = df.copy()
        if 'power' in df.columns and df['power'].notna().any():
            df_smooth['power_smooth'] = self.smooth_series(df['power'], smooth_params['power'])
        if 'heart_rate' in df.columns and df['heart_rate'].notna().any():
            df_smooth['hr_smooth'] = self.smooth_series(df['heart_rate'], smooth_params['hr'])
        if 'cadence' in df.columns and df['cadence'].notna().any():
            df_smooth['cadence_smooth'] = self.smooth_series(df['cadence'], smooth_params['cadence'])
        
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
        
        plt.close()


    def add_ramp_target_power(self, df, intervals, ftp=250):
        """Add target power handling ALL interval types correctly"""
        df_enhanced = df.copy()
        
        # Initialize target columns
        if 'target_power' not in df_enhanced.columns:
            df_enhanced['target_power'] = np.nan
        if 'interval_type' not in df_enhanced.columns:
            df_enhanced['interval_type'] = None
        if 'zone_class' not in df_enhanced.columns:
            df_enhanced['zone_class'] = None
        
        print(f"🔄 Processing {len(intervals)} intervals with FTP: {ftp}W")
        
        for i, interval in enumerate(intervals):
            start_time = self.safe_float_convert(interval.get('start_time', 0))
            end_time = self.safe_float_convert(interval.get('end_time', 0))
            interval_type = interval.get('type', 'unknown')
            zone_class = interval.get('zone_class', 'unknown')
            power_type = interval.get('power_type', 'steady')
            
            print(f"   Interval {i}: {interval_type} ({zone_class}), power_type: {power_type}")
            
            # Get ALL power values
            raw_power_pct = interval.get('target_power_pct', 0)
            raw_power_low = interval.get('target_power_low_pct', 0)
            raw_power_high = interval.get('target_power_high_pct', 0)
            
            print(f"      Raw power values: pct={raw_power_pct}, low={raw_power_low}, high={raw_power_high}")
            
            # Convert from decimal to percentage
            power_pct = raw_power_pct * 100
            power_low_pct = raw_power_low * 100
            power_high_pct = raw_power_high * 100
            
            print(f"      After % conversion: pct={power_pct}%, low={power_low_pct}%, high={power_high_pct}%")
            
            # Handle different interval types
            if power_type in ['ramp_up', 'ramp_down']:
                # For ramps, we need both low and high
                if power_low_pct == 0 and power_pct > 0:
                    power_low_pct = power_pct
                if power_high_pct == 0 and power_pct > 0:
                    power_high_pct = power_pct
                print(f"      → Ramp adjustment: low={power_low_pct}%, high={power_high_pct}%")
            elif power_pct > 0 and (power_low_pct == 0 or power_high_pct == 0):
                # For steady intervals, if we have main power but missing low/high, use main power
                power_low_pct = power_pct
                power_high_pct = power_pct
                print(f"      → Using main power for both: {power_pct}%")
            
            # Convert to absolute watts
            power_low_watts = (power_low_pct / 100.0) * ftp
            power_high_watts = (power_high_pct / 100.0) * ftp
            
            print(f"      → Absolute power: low={power_low_watts}W, high={power_high_watts}W")
            
            # Find data points in this interval
            mask = (df_enhanced['timestamp'] >= start_time) & (df_enhanced['timestamp'] <= end_time)
            interval_points = df_enhanced[mask]
            
            if len(interval_points) == 0:
                print(f"      ⚠️  No data points found in interval {start_time}-{end_time}")
                continue
                
            if power_low_watts <= 0 and power_high_watts <= 0:
                print(f"      ⚠️  No valid power values for interval")
                continue
            
            timestamps = interval_points['timestamp'].values
            
            # Calculate target power based on power_type
            if power_type in ['ramp_up', 'ramp_down'] and power_low_watts != power_high_watts:
                # Ramp interval
                progress = (timestamps - start_time) / (end_time - start_time)
                
                if power_type == 'ramp_down':
                    target_powers = power_low_watts + (power_high_watts - power_low_watts) * progress
                    print(f"      → Ramp DOWN: {power_high_watts:.1f}W → {power_low_watts:.1f}W")
                else:
                    target_powers = power_low_watts + (power_high_watts - power_low_watts) * progress
                    print(f"      → Ramp UP: {power_low_watts:.1f}W → {power_high_watts:.1f}W")
            else:
                # Steady interval
                target_powers = (power_low_watts + power_high_watts) / 2.0
                print(f"      → Steady: {target_powers:.1f}W")
            
            # Apply to dataframe
            df_enhanced.loc[mask, 'target_power'] = target_powers
            df_enhanced.loc[mask, 'interval_type'] = interval_type
            df_enhanced.loc[mask, 'zone_class'] = zone_class
            
            print(f"      ✅ Set target power for {len(interval_points)} data points")
        
        return df_enhanced

    def test_expected_zwo_structure(self):
        """Test what the ZWO parser should produce"""
        print("🧪 EXPECTED ZWO STRUCTURE TEST")
        print("=" * 60)
        
        # This is what your ZWO intervals SHOULD look like based on your parser
        expected_interval = {
            'type': 'steadystate',  # or 'warmup', 'cooldown', 'ramp'
            'zone_class': 'Z3',     # Z1, Z2, Z3, Z4, Z5, Z6
            'source': 'zwo',
            'power_type': 'steady', # 'steady', 'ramp_up', 'ramp_down'
            'target_power_pct': 75.0,
            'target_power_low_pct': 75.0,  # Same as target_power_pct for steady
            'target_power_high_pct': 75.0, # Same as target_power_pct for steady  
            'start_time': 300.0,
            'end_time': 600.0,
            'duration': 300.0,
            'target_cadence': 90.0
        }
        
        print("Expected interval structure:")
        for key, value in expected_interval.items():
            print(f"   {key}: {value}")
        
        return expected_interval

    
    
    def plot_with_intervals(self, data, interval_type, save_path=None):
        """Plot training data with intervals highlighted - KEEPS ALL ANALYTICS"""
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
        
        print(f"📊 Plotting with {len(intervals)} intervals from {interval_source}")
        
        # COMPREHENSIVE DEBUG
        self.debug_interval_data(intervals)
        zone_classes, interval_types = self.debug_zone_classes(intervals)
        
        # Add ramp target power to dataframe
        ftp = metadata.get('ftp', 250)
        print(f"🔧 Using FTP: {ftp}W")
        
        df_enhanced = self.add_ramp_target_power(df, intervals, ftp)
        
        # Check if target power was actually set
        target_power_non_nan = df_enhanced['target_power'].notna().sum()
        print(f"✅ Target power set for {target_power_non_nan} data points")
        
        # Calculate metrics for all intervals with HR drift
        all_metrics = []
        total_duration = self.safe_float_convert(df['timestamp'].max(), 3600)
        
        for interval in intervals:
            metrics = self.calculate_interval_metrics(interval, df_enhanced, total_duration)
            if metrics:
                all_metrics.append(metrics)
        
        # Apply smoothing
        smooth_params = self.get_smoothing_params('combined', total_duration)
        df_smooth = df_enhanced.copy()
        if 'power' in df_enhanced.columns and df_enhanced['power'].notna().any():
            df_smooth['power_smooth'] = self.smooth_series(df_enhanced['power'], smooth_params['power'])
        if 'heart_rate' in df_enhanced.columns and df_enhanced['heart_rate'].notna().any():
            df_smooth['hr_smooth'] = self.smooth_series(df_enhanced['heart_rate'], smooth_params['hr'])
        if 'cadence' in df_enhanced.columns and df_enhanced['cadence'].notna().any():
            df_smooth['cadence_smooth'] = self.smooth_series(df_enhanced['cadence'], smooth_params['cadence'])
        
        # Create plot with 4 subplots to include target power
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        time_minutes = df_enhanced['timestamp'] / 60
        
        # KEEP YOUR ORIGINAL INTERVAL COLORS STRUCTURE
        interval_colors = {
            # Power Zones
            'Z1': 'lightblue', 
            'Z2': 'blue', 
            'Z3': 'green',
            'Z4': 'yellow', 
            'Z5': 'orange', 
            'Z6': 'red',
            
            # Interval Types (fallbacks)
            'warmup': 'lightgray', 
            'cooldown': 'lightgray',
            'steadystate': 'purple', 
            'ramp': 'brown',
            'interval_on': 'coral',
            'interval_off': 'lightgreen',
            
            # Add any other zone classes you might have
            'unknown': 'gray'
        }
        
        # Plot intervals as colored backgrounds WITHOUT individual legends
        zone_count = {}
        for interval in intervals:
            zone_class = interval.get('zone_class', 'unknown')
            interval_type = interval.get('type', 'unknown')
            
            # Use zone_class for color, fallback to interval_type
            color_key = zone_class if zone_class != 'unknown' else interval_type
            color = interval_colors.get(color_key, 'gray')
            
            # Count zones
            zone_count[zone_class] = zone_count.get(zone_class, 0) + 1
            
            # Add colored background for each interval (NO LABEL to avoid duplicate legends)
            start_min = interval['start_time'] / 60
            end_min = interval['end_time'] / 60
            
            for ax in axes:
                ax.axvspan(start_min, end_min, alpha=0.3, color=color)  # Removed label=zone
        
        # --- PLOT 1: Power with target power ---
        power_lines = []
        if 'power' in df_smooth.columns and 'power_smooth' in df_smooth.columns:
            # Actual power
            power_line = axes[0].plot(time_minutes, df_smooth['power_smooth'], 
                                    color='red', linewidth=1.5, label='Actual Power')[0]
            power_lines.append(power_line)
            
            # Target power if available
            if 'target_power' in df_enhanced.columns and df_enhanced['target_power'].notna().any():
                target_line = axes[0].plot(time_minutes, df_enhanced['target_power'], 
                                         color='black', linestyle='--', linewidth=1.5, 
                                         alpha=0.8, label='Target Power')[0]
                power_lines.append(target_line)
        
        axes[0].set_ylabel('Power (W)')
        axes[0].set_title(f"Training with {interval_source} Intervals: {data['training']} - {metadata.get('training_type', '')}")
        axes[0].grid(True, alpha=0.3)
        
        # Add power threshold lines if FTP is available
        ftp_lines = []
        if ftp:
            power_zones = {
                'Active Recovery': ftp * 0.55,
                'Endurance': ftp * 0.75,
                'Tempo': ftp * 0.90,
                'Threshold': ftp * 1.00,
                'VO2 Max': ftp * 1.20,
            }
            
            threshold_colors = {
                'Active Recovery': 'lightblue',
                'Endurance': 'blue', 
                'Tempo': 'green',
                'Threshold': 'orange',
                'VO2 Max': 'red'
            }
            
            for zone_name, power_value in power_zones.items():
                color = threshold_colors.get(zone_name, 'gray')
                line = axes[0].axhline(y=power_value, color=color, linestyle=':', alpha=0.7, 
                               linewidth=1, label=f'{zone_name} ({power_value:.0f}W)')
                ftp_lines.append(line)
        
        # Add local legend for power plot
        if power_lines or ftp_lines:
            axes[0].legend(loc='upper right', frameon=True, fontsize=9)
        
        # --- PLOT 2: Power Delta (Actual - Target) ---
        delta_lines = []
        if 'power' in df_enhanced.columns and 'target_power' in df_enhanced.columns:
            power_delta = df_enhanced['power'] - df_enhanced['target_power']
            power_delta_smooth = self.smooth_series(power_delta, smooth_params['power'])
            delta_line = axes[1].plot(time_minutes, power_delta_smooth, 
                                    color='purple', linewidth=1.5, 
                                    label='Power Delta (Actual - Target)')[0]
            delta_lines.append(delta_line)
            
            # Fill areas
            fill_above = axes[1].fill_between(time_minutes, 0, power_delta_smooth, where=power_delta_smooth >= 0, 
                               alpha=0.3, color='red', label='Above Target')
            fill_below = axes[1].fill_between(time_minutes, 0, power_delta_smooth, where=power_delta_smooth < 0, 
                               alpha=0.3, color='blue', label='Below Target')
            zero_line = axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Target Line')
            
            delta_lines.extend([fill_above, fill_below, zero_line])
        
        axes[1].set_ylabel('Power Δ (W)')
        axes[1].grid(True, alpha=0.3)
        
        # Add local legend for delta plot
        if delta_lines:
            axes[1].legend(loc='upper right', frameon=True, fontsize=9)
        
        # --- PLOT 3: Heart rate ---
        hr_lines = []
        if 'heart_rate' in df_smooth.columns and 'hr_smooth' in df_smooth.columns:
            hr_line = axes[2].plot(time_minutes, df_smooth['hr_smooth'], 
                                 color='blue', linewidth=1.5, label='Heart Rate')[0]
            hr_lines.append(hr_line)
            axes[2].set_ylabel('Heart Rate (bpm)')
            axes[2].grid(True, alpha=0.3)
        
        # Add local legend for HR plot
        if hr_lines:
            axes[2].legend(loc='upper right', frameon=True, fontsize=9)
        
        # --- PLOT 4: Cadence with target ---
        cadence_lines = []
        if 'cadence' in df_smooth.columns and 'cadence_smooth' in df_smooth.columns:
            # Actual cadence
            cadence_line = axes[3].plot(time_minutes, df_smooth['cadence_smooth'], 
                                      color='green', linewidth=1.5, label='Actual Cadence')[0]
            cadence_lines.append(cadence_line)
            
            # Target cadence if available
            target_cadence_set = False
            for interval in intervals:
                if 'target_cadence' in interval and interval['target_cadence'] > 0:
                    start_min = interval['start_time'] / 60
                    end_min = interval['end_time'] / 60
                    target_cadence = interval['target_cadence']
                    target_line = axes[3].hlines(y=target_cadence, xmin=start_min, xmax=end_min, 
                                 color='orange', linestyle='-', linewidth=2, alpha=0.7,
                                 label='ZWO Target Cadence' if not target_cadence_set else "")
                    if not target_cadence_set:
                        cadence_lines.append(target_line)
                        target_cadence_set = True
        
        axes[3].set_ylabel('Cadence (rpm)')
        axes[3].set_xlabel('Time (minutes)')
        axes[3].grid(True, alpha=0.3)
        
        # Add local legend for cadence plot
        if cadence_lines:
            axes[3].legend(loc='upper right', frameon=True, fontsize=9)
        
        # Add HR drift annotations
        for i, (interval, metrics) in enumerate(zip(intervals, all_metrics)):
            if 'hr_drift_absolute' in metrics and abs(metrics['hr_drift_absolute']) > 2:
                drift_color = 'red' if metrics['hr_drift_absolute'] > 5 else 'orange' if metrics['hr_drift_absolute'] > 2 else 'green'
                trend_symbol = "↗" if metrics['hr_drift_absolute'] > 0 else "↘" if metrics['hr_drift_absolute'] < 0 else "→"
                start_min = interval['start_time'] / 60
                end_min = interval['end_time'] / 60
                axes[2].annotate(f"ΔHR: {metrics['hr_drift_absolute']:+.1f}{trend_symbol}", 
                               xy=((start_min + end_min) / 2, metrics.get('hr_mean', 0)),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', va='bottom', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=drift_color, alpha=0.7),
                               arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        
        # Debug: Print zone distribution
        print(f"📊 Zone distribution: {zone_count}")
        
        # CREATE COMMON LEGEND ONLY FOR INTERVAL COLORS (ZONES)
        self._create_common_zone_legend(fig, interval_colors, zone_count)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Interval plot saved: {save_path}")
        
        plt.close()
        
        # Create comprehensive metrics table with HR drift
        if all_metrics:
            self.create_metrics_table(all_metrics, interval_source, save_path)
        
        return all_metrics  # RETURN THE COMPREHENSIVE METRICS
    
    def _create_common_zone_legend(self, fig, interval_colors, zone_count):
        """
        Create a common legend only for interval colors (zones) that appears once for the entire figure.
        """
        from matplotlib.patches import Patch
        
        # Collect unique zones that actually appear in the plot
        unique_zones = []
        zone_elements = []
        
        for zone in zone_count.keys():
            if zone in interval_colors and zone not in unique_zones:
                unique_zones.append(zone)
                color = interval_colors[zone]
                zone_elements.append(
                    Patch(facecolor=color, alpha=0.3, label=zone, edgecolor='black')
                )
        
        # Create common zone legend at the top of the figure
        if zone_elements:
            fig.legend(
                handles=zone_elements,
                loc='upper left',
                bbox_to_anchor=(0.02, 0.98),  # Position at top center
                ncol=min(6, len(zone_elements)),  # Adjust columns based on number of zones
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize='medium',
                title='Power Zones',
                title_fontsize='medium'
            )
        
#
#
#    def plot_with_intervals(self, data, interval_type, save_path=None):
#        """Plot training data with intervals highlighted"""
#        df = data['time_series']
#        metadata = data.get('metadata', {})
#        
#        # Get intervals based on type
#        if interval_type == 'zwo':
#            intervals = data.get('zwo_parsed', []) or data.get('zwo_intervals', [])
#            interval_source = "ZWO File"
#        elif interval_type == 'auto':
#            intervals = data.get('auto_intervals', [])
#            interval_source = "Auto-detected"
#        else:  # combined
#            intervals = (data.get('zwo_parsed', []) or data.get('zwo_intervals', [])) + data.get('auto_intervals', [])
#            interval_source = "Combined (ZWO + Auto)"
#        
#        print(f"📊 Plotting with {len(intervals)} intervals from {interval_source}")
#        expected = self.test_expected_zwo_structure()        
#        # COMPREHENSIVE DEBUG
#        self.debug_interval_data(intervals)
#        # After getting intervals, add this:
#        zone_classes, interval_types = self.debug_zone_classes(intervals)
#        # Add ramp target power to dataframe
#        ftp = metadata.get('ftp', 250)  # Default FTP if not provided
#        print(f"🔧 Using FTP: {ftp}W")
#        
#        df_enhanced = self.add_ramp_target_power(df, intervals, ftp)
#        
#        # Check if target power was actually set
#        target_power_non_nan = df_enhanced['target_power'].notna().sum()
#        print(f"✅ Target power set for {target_power_non_nan} data points")
#        
#        # Calculate metrics for all intervals with HR drift
#        all_metrics = []
#        total_duration = self.safe_float_convert(df['timestamp'].max(), 3600)
#        
#        for interval in intervals:
#            metrics = self.calculate_interval_metrics(interval, df_enhanced, total_duration)
#            if metrics:
#                all_metrics.append(metrics)
#        
#            
#            # Apply smoothing
#            smooth_params = self.get_smoothing_params('combined', total_duration)
#            df_smooth = df_enhanced.copy()
#            if 'power' in df_enhanced.columns and df_enhanced['power'].notna().any():
#                df_smooth['power_smooth'] = self.smooth_series(df_enhanced['power'], smooth_params['power'])
#            if 'heart_rate' in df_enhanced.columns and df_enhanced['heart_rate'].notna().any():
#                df_smooth['hr_smooth'] = self.smooth_series(df_enhanced['heart_rate'], smooth_params['hr'])
#            if 'cadence' in df_enhanced.columns and df_enhanced['cadence'].notna().any():
#                df_smooth['cadence_smooth'] = self.smooth_series(df_enhanced['cadence'], smooth_params['cadence'])
#            
#            # Create plot with 4 subplots to include target power
#            fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
#            time_minutes = df_enhanced['timestamp'] / 60
#        
#        # Define colors for different zone types based on your ZWO parser
#        interval_colors = {
#            # Power Zones
#            'Z1': 'lightblue', 
#            'Z2': 'blue', 
#            'Z3': 'green',
#            'Z4': 'yellow', 
#            'Z5': 'orange', 
#            'Z6': 'red',
#            
#            # Interval Types (fallbacks)
#            'warmup': 'lightgray', 
#            'cooldown': 'lightgray',
#            'steadystate': 'purple', 
#            'ramp': 'brown',
#            'interval_on': 'coral',
#            'interval_off': 'lightgreen',
#            
#            # Add any other zone classes you might have
#            'unknown': 'gray'
#        }     
#        print(f"🎨 Applying colors to {len(intervals)} intervals...")
#        for i, (interval, metrics) in enumerate(zip(intervals, all_metrics)):
#            zone_class = interval.get('zone_class', 'unknown')
#            interval_type = interval.get('type', 'unknown')
#            
#            # Use zone_class for color, fallback to interval_type
#            color_key = zone_class if zone_class != 'unknown' else interval_type
#            color = interval_colors.get(color_key, 'gray')
#            
#            start_min = self.safe_float_convert(interval.get('start_time', 0)) / 60
#            end_min = self.safe_float_convert(interval.get('end_time', 0)) / 60
#            
#            print(f"   Interval {i}: {interval_type} (zone: {zone_class}) -> color: {color}")
#            print(f"      Time: {start_min:.1f} to {end_min:.1f} min")
#
#            # Plot 1: Power with target power
#            if 'power' in df_smooth.columns and 'power_smooth' in df_smooth.columns:
#                # Actual power
#                axes[0].plot(time_minutes, df_smooth['power_smooth'], label='Actual Power', color='red', linewidth=1.5)
#                
#                # Target power if available
#                if 'target_power' in df_enhanced.columns and df_enhanced['target_power'].notna().any():
#                    axes[0].plot(time_minutes, df_enhanced['target_power'], label='Target Power', 
#                               color='black', linestyle='--', linewidth=1.5, alpha=0.8)
#            
#            axes[0].set_ylabel('Power (W)')
#            axes[0].set_title(f"Training with {interval_source} Intervals: {data['training']} - {metadata.get('training_type', '')}")
#            axes[0].grid(True, alpha=0.3)
#            axes[0].legend()
#
#        
#        # Add power threshold lines if FTP is available
#        ftp = metadata.get('ftp')
#        if ftp:
#            print(f"📊 Adding power threshold lines for FTP: {ftp}W")
#            
#            # Define common power zones (as percentages of FTP)
#            power_zones = {
#                'Active Recovery': ftp * 0.55,
#                'Endurance': ftp * 0.75,
#                'Tempo': ftp * 0.90,
#                'Threshold': ftp * 1.00,
#                'VO2 Max': ftp * 1.20,
#            }
#            
#            # Colors for threshold lines
#            threshold_colors = {
#                'Active Recovery': 'lightblue',
#                'Endurance': 'blue', 
#                'Tempo': 'green',
#                'Threshold': 'orange',
#                'VO2 Max': 'red'
#            }
#            
#            # Plot each threshold line
#            for zone_name, power_value in power_zones.items():
#                color = threshold_colors.get(zone_name, 'gray')
#                axes[0].axhline(y=power_value, color=color, linestyle=':', alpha=0.7, 
#                               linewidth=1, label=f'{zone_name} ({power_value:.0f}W)')
#        
#
# 
#            # Plot 2: Power Delta (Actual - Target)
#            if 'power' in df_enhanced.columns and 'target_power' in df_enhanced.columns:
#                power_delta = df_enhanced['power'] - df_enhanced['target_power']
#                power_delta_smooth = self.smooth_series(power_delta, smooth_params['power'])
#                axes[1].plot(time_minutes, power_delta_smooth, label='Power Delta (Actual - Target)', 
#                           color='purple', linewidth=1.5)
#                axes[1].fill_between(time_minutes, 0, power_delta_smooth, where=power_delta_smooth >= 0, 
#                                   alpha=0.3, color='red', label='Above Target')
#                axes[1].fill_between(time_minutes, 0, power_delta_smooth, where=power_delta_smooth < 0, 
#                                   alpha=0.3, color='blue', label='Below Target')
#                axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
#            
#            axes[1].set_ylabel('Power Δ (W)')
#            axes[1].grid(True, alpha=0.3)
#            axes[1].legend()
#            
#            # Plot 3: Heart rate
#            if 'heart_rate' in df_smooth.columns and 'hr_smooth' in df_smooth.columns:
#                axes[2].plot(time_minutes, df_smooth['hr_smooth'], label='Heart Rate', color='blue', linewidth=1.5)
#                axes[2].set_ylabel('Heart Rate (bpm)')
#                axes[2].grid(True, alpha=0.3)
#                axes[2].legend()
#            
#            # Plot 4: Cadence with target
#            if 'cadence' in df_smooth.columns and 'cadence_smooth' in df_smooth.columns:
#                # Actual cadence
#                axes[3].plot(time_minutes, df_smooth['cadence_smooth'], label='Actual Cadence', color='green', linewidth=1.5)
#                
#                # Target cadence if available
#                target_cadence_set = False
#                for interval in intervals:
#                    if 'target_cadence' in interval and interval['target_cadence'] > 0:
#                        start_min = self.safe_float_convert(interval.get('start_time', 0)) / 60
#                        end_min = self.safe_float_convert(interval.get('end_time', 0)) / 60
#                        target_cadence = interval['target_cadence']
#                        axes[3].hlines(y=target_cadence, xmin=start_min, xmax=end_min, 
#                                     color='orange', linestyle='-', linewidth=2, alpha=0.7,
#                                     label='ZWO Target Cadence' if not target_cadence_set else "")
#                        target_cadence_set = True
#            
#            axes[3].set_ylabel('Cadence (rpm)')
#            axes[3].set_xlabel('Time (minutes)')
#            axes[3].grid(True, alpha=0.3)
#            axes[3].legend()
#
#
#        # Highlight intervals and mark HR drift
#        legend_added = set()
#        zone_count = {}
#        print(f"🎨 Coloring {len(intervals)} intervals...")
#        
#        for i, (interval, metrics) in enumerate(zip(intervals, all_metrics)):
#            zone_class = interval.get('zone_class', 'unknown')
#            interval_type = interval.get('type', 'unknown')
#            
#            # Track zone counts for debug
#            zone_count[zone_class] = zone_count.get(zone_class, 0) + 1
#            
#            # Use zone_class for color, fallback to interval_type
#            color_key = zone_class if zone_class != 'unknown' else interval_type
#            color = interval_colors.get(color_key, 'gray')
#            
#            start_min = self.safe_float_convert(interval.get('start_time', 0)) / 60
#            end_min = self.safe_float_convert(interval.get('end_time', 0)) / 60
#            
#            print(f"   Interval {i}: {interval_type} (zone: {zone_class}) -> color: {color}")
#            print(f"      Time: {start_min:.1f} to {end_min:.1f} min")
#            
#            # Highlight interval area on ALL subplots
#            for ax_idx, ax in enumerate(axes):
#                # Only add to legend once per zone class
#                label = f"{zone_class}" if (zone_class not in legend_added and ax_idx == 0) else ""
#                ax.axvspan(start_min, end_min, alpha=0.3, color=color, label=label)
#                
#                # Add zone label text in the middle of the interval on the power plot
#                if ax_idx == 0:  # Only on power plot
#                    mid_time = (start_min + end_min) / 2
#                    ax.text(mid_time, ax.get_ylim()[1] * 0.95, zone_class, 
#                           ha='center', va='top', fontsize=9, fontweight='bold',
#                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
#            
#            if zone_class not in legend_added:
#                legend_added.add(zone_class)
#            
#            # Mark HR drift with text annotation on HR plot
#            if 'hr_drift_absolute' in metrics and abs(metrics['hr_drift_absolute']) > 2:
#                drift_color = 'red' if metrics['hr_drift_absolute'] > 5 else 'orange' if metrics['hr_drift_absolute'] > 2 else 'green'
#                trend_symbol = "↗" if metrics['hr_drift_absolute'] > 0 else "↘" if metrics['hr_drift_absolute'] < 0 else "→"
#                axes[2].annotate(f"ΔHR: {metrics['hr_drift_absolute']:+.1f}{trend_symbol}", 
#                               xy=((start_min + end_min) / 2, metrics.get('hr_mean', 0)),
#                               xytext=(0, 10), textcoords='offset points',
#                               ha='center', va='bottom', fontsize=8,
#                               bbox=dict(boxstyle='round,pad=0.2', facecolor=drift_color, alpha=0.7),
#                               arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
#       
#
#        # Debug: Print zone distribution
#        print(f"📊 Zone distribution: {zone_count}")
#        
#        # Create clean legend for essential elements only
#        print("🎨 Creating clean legend...")
#        
#        try:
#            # Get current handles and labels from power plot
#            handles, labels = axes[0].get_legend_handles_labels()
#            
#            # Keep only essential lines and add zone colors
#            essential_handles = []
#            essential_labels = []
#            
#            # Add the main data lines
#            for handle, label in zip(handles, labels):
#                if any(keyword in label for keyword in ['Actual', 'Target', 'FTP', 'Endurance', 'Tempo', 'Threshold']):
#                    essential_handles.append(handle)
#                    essential_labels.append(label)
#            
#            # Add zone colors
#            unique_zones = set(interval.get('zone_class', 'unknown') for interval in intervals)
#            for zone_class in sorted(unique_zones):
#                color = interval_colors.get(zone_class, 'gray')
#                proxy_artist = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=f'Zone {zone_class}')
#                essential_handles.append(proxy_artist)
#                essential_labels.append(f'Zone {zone_class}')
#            
#            # Create clean legend
#            axes[0].legend(handles=essential_handles, labels=essential_labels, 
#                          title="Training Guide", loc='upper right',
#                          framealpha=0.9, fontsize=8)
#            
#            # Remove legends from other subplots
#            for ax in axes[1:]:
#                if ax.get_legend() is not None:
#                    ax.get_legend().remove()
#                    
#        except Exception as e:
#            print(f"⚠️ Legend creation failed: {e}")
#            # Basic fallback
#            for ax in axes:
#                ax.legend()
#        
#        plt.tight_layout()
#        
#        if save_path:
#            plt.savefig(save_path, dpi=150, bbox_inches='tight')
#            print(f"💾 Interval plot saved: {save_path}")
#        
#        plt.close()
#        
#        # Create comprehensive metrics table with HR drift
#        if all_metrics:
#            self.create_metrics_table(all_metrics, interval_source, save_path)
#        
#        return all_metrics
#
    def calculate_interval_metrics(self, interval, df, total_duration):
        """Calculate comprehensive interval metrics with HR drift analysis"""
        # Implementation remains the same as your original
        start_time = self.safe_float_convert(interval.get('start_time', 0))
        end_time = self.safe_float_convert(interval.get('end_time', 0))
        
        interval_data = df[
            (df['timestamp'] >= start_time) & 
            (df['timestamp'] <= end_time)
        ].copy()
        
        if interval_data.empty:
            print(f"⚠️ No data found for interval {start_time}-{end_time}")
            return None
        
        # Calculate HR drift metrics
        hr_drift_metrics = self.calculate_hr_drift(interval_data, interval)
        
        metrics = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'time_into_workout': start_time,
            'fatigue_index': start_time / total_duration if total_duration > 0 else 0,
            'zone_type': interval.get('zone_class', 'unknown'),
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

    def calculate_hr_drift(self, interval_data, interval):
        """Calculate detailed heart rate drift metrics"""
        # Implementation remains the same as your original
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
            
        return hr_metrics

    def create_metrics_table(self, all_metrics, interval_source, save_path=None):
        """Create a comprehensive metrics table with HR drift analysis"""
        # Implementation remains mostly the same as your original
        # Basic metrics table
        fig_basic, ax_basic = plt.subplots(figsize=(16, len(all_metrics) * 0.4 + 2))
        ax_basic.axis('tight')
        ax_basic.axis('off')
        
        basic_headers = ['#', 'Type', 'Zone', 'Start', 'Dur(s)', 'P_avg', 'P_std', 'P_trend', 'Fatigue']
        basic_data = []
        
        for i, metrics in enumerate(all_metrics):
            trend_symbol = {
                'increasing': '↗', 'decreasing': '↘', 'constant': '→'
            }.get(metrics.get('power_trend', 'constant'), '→')
            
            row = [
                i+1,
                metrics.get('type', 'unknown')[:10],
                metrics.get('zone_type', 'unknown'),
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
        
        ax_basic.set_title(f'Basic Interval Metrics - {interval_source}')
        plt.tight_layout()
        
        if save_path:
            basic_metrics_path = save_path.parent / f"{save_path.stem}_basic_metrics.png"
            plt.savefig(basic_metrics_path, dpi=150, bbox_inches='tight')
            print(f"📊 Basic metrics table saved: {basic_metrics_path}")
        
        plt.close()
        
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
            drift_color = 'red' if hr_drift_abs > 5 else 'orange' if hr_drift_abs > 2 else 'green'
            
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
        hr_table.auto_set_font_size(False)
        hr_table.set_fontsize(8)
        hr_table.scale(1, 1.8)
        
        # Color the HR drift cells
        for i in range(1, len(hr_data) + 1):
            hr_drift_abs = all_metrics[i-1].get('hr_drift_absolute', 0)
            cell_color = 'red' if hr_drift_abs > 5 else 'orange' if hr_drift_abs > 2 else 'lightgreen'
            hr_table[(i, 4)].set_facecolor(cell_color)
            hr_table[(i, 5)].set_facecolor(cell_color)
        
        ax_hr.set_title(f'Heart Rate Analysis - {interval_source}')
        plt.tight_layout()
        
        if save_path:
            hr_metrics_path = save_path.parent / f"{save_path.stem}_hr_metrics.png"
            plt.savefig(hr_metrics_path, dpi=150, bbox_inches='tight')
            print(f"❤️ HR metrics table saved: {hr_metrics_path}")
        
        plt.close()
        
        # Save detailed metrics to JSON
        if save_path:
            json_path = save_path.parent / f"{save_path.stem}_detailed_metrics.json"
            with open(json_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"📋 Detailed metrics saved: {json_path}")
        
        return all_metrics

    # Keep the rest of your statistical plot methods as they are
    def plot_interval_compliance(self, output_dir: Path) -> None:
        """Plot compliance metrics by interval"""
        # Your existing implementation...
        pass

    def plot_interval_performance(self, output_dir: Path) -> None:
        """Plot various interval performance metrics"""
        # Your existing implementation...
        pass

    def plot_compliance_heatmap(self, output_dir: Path) -> None:
        """Create heatmap of compliance across intervals"""
        # Your existing implementation...
        pass

#### DEBUGGERS 
    def debug_zone_classes(self, intervals):
        """Debug the actual zone classes in the intervals"""
        print("🎯 DEBUG: Zone Class Analysis")
        print("=" * 50)


        interval_colors = {
            # Power Zones
            'Z1': 'lightblue', 
            'Z2': 'blue', 
            'Z3': 'green',
            'Z4': 'yellow', 
            'Z5': 'orange', 
            'Z6': 'red',
            
            # Interval Types (fallbacks)
            'warmup': 'lightgray', 
            'cooldown': 'lightgray',
            'steadystate': 'purple', 
            'ramp': 'brown',
            'interval_on': 'coral',
            'interval_off': 'lightgreen',
            
            # Add any other zone classes you might have
            'unknown': 'gray'
        }     
 
        zone_classes = set()
        interval_types = set()
        
        for i, interval in enumerate(intervals):
            zone_class = interval.get('zone_class', 'unknown')
            interval_type = interval.get('type', 'unknown')
            
            zone_classes.add(zone_class)
            interval_types.add(interval_type)
            
            print(f"   Interval {i:2d}: {interval_type:12} -> zone: {zone_class}")
        
        print(f"\n📊 Found {len(zone_classes)} unique zone classes: {sorted(zone_classes)}")
        print(f"📊 Found {len(interval_types)} unique interval types: {sorted(interval_types)}")
        
        # Check which zone classes are in our color dictionary
        missing_colors = []
        for zone in zone_classes:
            if zone not in interval_colors:
                missing_colors.append(zone)
        
        if missing_colors:
            print(f"⚠️  Missing colors for zone classes: {missing_colors}")
            print("   They will be displayed in gray")
        
        return zone_classes, interval_types


