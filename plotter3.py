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
    
    print(f"📊 Plotting with {len(intervals)} intervals")
    
    # DEBUG: Print interval information
    for i, interval in enumerate(intervals):
        print(f"   Interval {i}: {interval.get('type')}, "
              f"zone: {interval.get('zone_class')}, "
              f"power_pct: {interval.get('target_power_pct')}, "
              f"start: {interval.get('start_time')}, end: {interval.get('end_time')}")
    
    # Add ramp target power to dataframe
    ftp = metadata.get('ftp', 250)  # Default FTP if not provided
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
    
