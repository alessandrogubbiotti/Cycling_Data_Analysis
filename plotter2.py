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
    """Add linear increasing/decreasing target power for ramps, warmups, and cooldowns"""
    df_enhanced = df.copy()
    
    for interval in intervals:
        start_time = self.safe_float_convert(interval.get('start_time', 0))
        end_time = self.safe_float_convert(interval.get('end_time', 0))
        power_type = interval.get('power_type', 'steady')
        
        # Determine the target power percentage to use
        power_pct = self.safe_float_convert(interval.get('target_power_pct', 0))
        power_low_pct = self.safe_float_convert(interval.get('target_power_low_pct', 0))
        power_high_pct = self.safe_float_convert(interval.get('target_power_high_pct', 0))
        
        # Logic to determine which power values to use
        if power_type == 'steady':
            # For steady state, prioritize the main power value
            if power_pct > 0:
                effective_power_pct = power_pct
            elif power_low_pct > 0 or power_high_pct > 0:
                # Use average if low/high are set but main is zero
                effective_power_pct = (power_low_pct + power_high_pct) / 2
            else:
                effective_power_pct = 0
                
            power_low_pct = effective_power_pct
            power_high_pct = effective_power_pct
            
        else:
            # For ramps, ensure we have valid values
            if power_low_pct == 0 and power_pct > 0:
                power_low_pct = power_pct
            if power_high_pct == 0 and power_pct > 0:
                power_high_pct = power_pct
        
        # Convert to absolute power
        power_low = power_low_pct * ftp / 100
        power_high = power_high_pct * ftp / 100
        
        # Find data points within this interval
        mask = (df_enhanced['timestamp'] >= start_time) & (df_enhanced['timestamp'] <= end_time)
        interval_points = df_enhanced[mask]
        
        if len(interval_points) > 0 and power_low > 0:  # Only set if we have valid power
            timestamps = interval_points['timestamp'].values
            
            if power_type != 'steady' and power_low != power_high:
                # Create linear progression for ramps
                progress = (timestamps - start_time) / (end_time - start_time)
                
                if power_type == 'ramp_down':
                    target_powers = power_high - (power_high - power_low) * progress
                else:
                    target_powers = power_low + (power_high - power_low) * progress
            else:
                # For steady intervals or when low == high, use constant power
                target_powers = (power_low + power_high) / 2
            
            df_enhanced.loc[mask, 'target_power'] = target_powers
            df_enhanced.loc[mask, 'interval_type'] = interval.get('type')
            df_enhanced.loc[mask, 'zone_class'] = interval.get('zone_class', 'unknown')
        # Add this debug code in your add_ramp_target_power method
        print(f"🔍 Interval: {interval.get('type')}, "
              f"Power: {power_pct}%, "
              f"Low: {power_low_pct}%, "
              f"High: {power_high_pct}%, "
              f"Type: {power_type}")        
    return df_enhanced
   


    def plot_with_intervals(self, data, interval_type, save_path=None):
        """Plot training data with intervals highlighted"""
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
        
        # Add ramp target power to dataframe
        ftp = metadata.get('ftp', 250)  # Default FTP if not provided
        df_enhanced = self.add_ramp_target_power(df, intervals, ftp)
        
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
        
        # Define colors for different zone types based on your ZWO parser
        interval_colors = {
            'Z1': 'lightblue', 'Z2': 'blue', 'Z3': 'green',
            'Z4': 'yellow', 'Z5': 'orange', 'Z6': 'red',
            'warmup': 'lightgray', 'cooldown': 'lightgray'
        }
        
        # Plot 1: Power with target power
        if 'power' in df_smooth.columns and 'power_smooth' in df_smooth.columns:
            # Actual power
            axes[0].plot(time_minutes, df_smooth['power_smooth'], label='Actual Power', color='red', linewidth=1.5)
            
            # Target power if available
            if 'target_power' in df_enhanced.columns and df_enhanced['target_power'].notna().any():
                axes[0].plot(time_minutes, df_enhanced['target_power'], label='Target Power', 
                           color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        
        axes[0].set_ylabel('Power (W)')
        axes[0].set_title(f"Training with {interval_source} Intervals: {data['training']} - {metadata.get('training_type', '')}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Power Delta (Actual - Target)
        if 'power' in df_enhanced.columns and 'target_power' in df_enhanced.columns:
            power_delta = df_enhanced['power'] - df_enhanced['target_power']
            power_delta_smooth = self.smooth_series(power_delta, smooth_params['power'])
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
        
        # Plot 4: Cadence with target
        if 'cadence' in df_smooth.columns and 'cadence_smooth' in df_smooth.columns:
            # Actual cadence
            axes[3].plot(time_minutes, df_smooth['cadence_smooth'], label='Actual Cadence', color='green', linewidth=1.5)
            
            # Target cadence if available
            target_cadence_set = False
            for interval in intervals:
                if 'target_cadence' in interval and interval['target_cadence'] > 0:
                    start_min = self.safe_float_convert(interval.get('start_time', 0)) / 60
                    end_min = self.safe_float_convert(interval.get('end_time', 0)) / 60
                    target_cadence = interval['target_cadence']
                    axes[3].hlines(y=target_cadence, xmin=start_min, xmax=end_min, 
                                 color='orange', linestyle='-', linewidth=2, alpha=0.7,
                                 label='ZWO Target Cadence' if not target_cadence_set else "")
                    target_cadence_set = True
        
        axes[3].set_ylabel('Cadence (rpm)')
        axes[3].set_xlabel('Time (minutes)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        # Highlight intervals and mark HR drift
        legend_added = set()
        for i, (interval, metrics) in enumerate(zip(intervals, all_metrics)):
            zone_class = interval.get('zone_class', 'unknown')
            color = interval_colors.get(zone_class, 'gray')
            start_min = self.safe_float_convert(interval.get('start_time', 0)) / 60
            end_min = self.safe_float_convert(interval.get('end_time', 0)) / 60
            
            # Highlight interval area
            for ax in axes:
                label = zone_class if zone_class not in legend_added else ""
                ax.axvspan(start_min, end_min, alpha=0.2, color=color, label=label)
                if label:
                    legend_added.add(zone_class)
            
            # Mark HR drift with text annotation on HR plot
            if 'hr_drift_absolute' in metrics and abs(metrics['hr_drift_absolute']) > 2:
                drift_color = 'red' if metrics['hr_drift_absolute'] > 5 else 'orange' if metrics['hr_drift_absolute'] > 2 else 'green'
                trend_symbol = "↗" if metrics['hr_drift_absolute'] > 0 else "↘" if metrics['hr_drift_absolute'] < 0 else "→"
                axes[2].annotate(f"ΔHR: {metrics['hr_drift_absolute']:+.1f}{trend_symbol}", 
                               xy=((start_min + end_min) / 2, metrics.get('hr_mean', 0)),
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
        
        plt.close()
        
        # Create comprehensive metrics table with HR drift
        if all_metrics:
            self.create_metrics_table(all_metrics, interval_source, save_path)
        
        return all_metrics

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
