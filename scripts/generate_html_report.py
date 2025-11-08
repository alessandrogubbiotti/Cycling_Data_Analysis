#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def create_an_html_report():
    """
    Creates an interactive training dashboard with multiple visualization dimensions
    """
    
    # Create output directory
    output_dir = Path("_build/html")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. COLLECT ALL TRAININGS
    trainings_data = collect_all_trainings()
    
    # 2. GENERATE INTERACTIVE DASHBOARD
    html_content = generate_interactive_dashboard(trainings_data)
    
    # 3. SAVE THE REPORT
    output_file = output_dir / "index.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 4. SAVE TRAININGS DATA AS JSON FOR JAVASCRIPT
    save_trainings_json(trainings_data, output_dir)
    
    print(f"✅ Interactive dashboard generated at: {output_file}")
    return output_file

def collect_all_trainings():
    """
    Collects all trainings organized in the 3 different ways you mentioned.
    This is a placeholder - you'll need to implement based on your data structure.
    """
    # TODO: Implement based on your actual data structure
    # This should return a list of training dictionaries with metrics
    trainings = [
        {
            'id': 'training_001',
            'date': '2024-01-15',
            'workout_type': 'Threshold Intervals',
            'duration_minutes': 105,
            'training_load': 145,  # TSS
            'avg_power': 215,
            'avg_heart_rate': 148,
            'intensity_factor': 0.85,
            'intervals': [
                {'type': 'warmup', 'duration': 15, 'avg_power': 150},
                {'type': 'threshold', 'duration': 20, 'avg_power': 240},
                # ... more intervals
            ],
            'plot_path': 'plots/training_2024-01-15.png'
        },
        {
            'id': 'training_002',
            'date': '2024-01-12',
            'workout_type': 'VO2 Max',
            'duration_minutes': 80,
            'training_load': 120,
            'avg_power': 235,
            'avg_heart_rate': 152,
            'intensity_factor': 0.92,
            'intervals': [
                {'type': 'warmup', 'duration': 10, 'avg_power': 160},
                {'type': 'vo2_max', 'duration': 5, 'avg_power': 320},
                # ... more intervals
            ],
            'plot_path': 'plots/training_2024-01-12.png'
        },
        # ... more trainings
    ]
    
    return trainings

def generate_interactive_dashboard(trainings_data):
    """Generate the HTML with interactive scatter plots"""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert to JSON for JavaScript
    trainings_json = json.dumps(trainings_data)
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cycling Training Dashboard</title>
    
    <!-- Plotly.js for interactive charts -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .dashboard-section {{
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .plot-container {{
            width: 100%;
            height: 500px;
            margin: 20px 0;
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        select, button {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: white;
        }}
        
        .training-details {{
            display: none;
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        
        .training-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .info-card {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .plot-gallery {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .plot-item {{
            text-align: center;
        }}
        
        .plot-item img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🚴 Interactive Cycling Training Dashboard</h1>
            <p>Explore your training data across multiple dimensions</p>
        </div>
        
        <!-- Visualization Controls -->
        <div class="dashboard-section">
            <h2>Training Overview</h2>
            <p>Click on any point to view detailed analysis for that training session</p>
            
            <div class="controls">
                <div class="control-group">
                    <label>X-Axis:</label>
                    <select id="x-axis">
                        <option value="date">Date</option>
                        <option value="workout_type">Workout Type</option>
                        <option value="duration_minutes">Duration</option>
                        <option value="intensity_factor">Intensity Factor</option>
                        <option value="avg_power">Average Power</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Y-Axis:</label>
                    <select id="y-axis">
                        <option value="training_load">Training Load (TSS)</option>
                        <option value="duration_minutes">Duration</option>
                        <option value="avg_power">Average Power</option>
                        <option value="avg_heart_rate">Average HR</option>
                        <option value="intensity_factor">Intensity Factor</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Color By:</label>
                    <select id="color-by">
                        <option value="workout_type">Workout Type</option>
                        <option value="intensity_factor">Intensity</option>
                        <option value="duration_minutes">Duration</option>
                    </select>
                </div>
                
                <button onclick="updatePlot()">Update Plot</button>
            </div>
            
            <!-- Interactive Scatter Plot -->
            <div id="training-plot" class="plot-container"></div>
        </div>
        
        <!-- Training Details Section (hidden by default) -->
        <div id="training-details" class="training-details">
            <h2 id="detail-title">Training Details</h2>
            
            <div class="training-info" id="training-info">
                <!-- Will be populated by JavaScript -->
            </div>
            
            <h3>Interval Structure</h3>
            <div id="interval-details">
                <!-- Will be populated by JavaScript -->
            </div>
            
            <div class="plot-gallery">
                <div class="plot-item">
                    <h4>Power Distribution</h4>
                    <img id="detail-plot" src="" alt="Training Plot">
                </div>
                <div class="plot-item">
                    <h4>Heart Rate Zones</h4>
                    <img id="hr-plot" src="" alt="HR Analysis">
                </div>
            </div>
        </div>
        
        <!-- Future Feature Notice -->
        <div class="dashboard-section">
            <h3>Future Features</h3>
            <p><strong>Interval Search:</strong> Coming soon - select an interval type and find all trainings containing it to track evolution over time.</p>
        </div>
        
        <div style="text-align: center; color: #666; margin-top: 40px;">
            Generated on: {current_time}
        </div>
    </div>

    <script>
        // Training data loaded from Python
        const trainings = {trainings_json};
        
        // Initialize the plot
        function initializePlot() {{
            updatePlot();
        }}
        
        // Update plot based on control selections
        function updatePlot() {{
            const xAxis = document.getElementById('x-axis').value;
            const yAxis = document.getElementById('y-axis').value;
            const colorBy = document.getElementById('color-by').value;
            
            const xData = trainings.map(t => t[xAxis]);
            const yData = trainings.map(t => t[yAxis]);
            const colors = trainings.map(t => getColor(t[colorBy], colorBy));
            const texts = trainings.map(t => 
                `Date: ${{t.date}}<br>Type: ${{t.workout_type}}<br>Load: ${{t.training_load}}<br>Duration: ${{t.duration_minutes}}min`
            );
            const ids = trainings.map(t => t.id);
            
            const trace = {{
                x: xData,
                y: yData,
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 12,
                    color: colors,
                    line: {{ width: 2, color: 'white' }}
                }},
                text: texts,
                customdata: ids,
                hovertemplate: '<b>%{{text}}</b><extra></extra>'
            }};
            
            const layout = {{
                title: `Training Overview: ${{getAxisLabel(xAxis)}} vs ${{getAxisLabel(yAxis)}}`,
                xaxis: {{ title: getAxisLabel(xAxis) }},
                yaxis: {{ title: getAxisLabel(yAxis) }},
                hovermode: 'closest',
                clickmode: 'event+select'
            }};
            
            Plotly.newPlot('training-plot', [trace], layout);
            
            // Add click event
            document.getElementById('training-plot').on('plotly_click', function(data) {{
                const point = data.points[0];
                const trainingId = point.customdata;
                showTrainingDetails(trainingId);
            }});
        }}
        
        // Show details for a specific training
        function showTrainingDetails(trainingId) {{
            const training = trainings.find(t => t.id === trainingId);
            if (!training) return;
            
            // Update basic info
            document.getElementById('detail-title').textContent = 
                `Training: ${{training.date}} - ${{training.workout_type}}`;
            
            // Update training info cards
            const infoHtml = `
                <div class="info-card">
                    <strong>Date</strong><br>{training.date}
                </div>
                <div class="info-card">
                    <strong>Workout Type</strong><br>{training.workout_type}
                </div>
                <div class="info-card">
                    <strong>Duration</strong><br>{training.duration_minutes} min
                </div>
                <div class="info-card">
                    <strong>Training Load</strong><br>{training.training_load} TSS
                </div>
                <div class="info-card">
                    <strong>Avg Power</strong><br>{training.avg_power}W
                </div>
                <div class="info-card">
                    <strong>Avg HR</strong><br>{training.avg_heart_rate} bpm
                </div>
            `;
            document.getElementById('training-info').innerHTML = infoHtml;
            
            // Update interval details
            if (training.intervals) {{
                let intervalsHtml = '<table style="width: 100%; border-collapse: collapse;">';
                intervalsHtml += '<tr><th>Type</th><th>Duration (min)</th><th>Avg Power</th></tr>';
                training.intervals.forEach(interval => {{
                    intervalsHtml += `<tr>
                        <td>${{interval.type}}</td>
                        <td>${{interval.duration}}</td>
                        <td>${{interval.avg_power || 'N/A'}}W</td>
                    </tr>`;
                }});
                intervalsHtml += '</table>';
                document.getElementById('interval-details').innerHTML = intervalsHtml;
            }}
            
            // Update plots
            document.getElementById('detail-plot').src = training.plot_path;
            document.getElementById('hr-plot').src = training.plot_path.replace('.png', '_hr.png');
            
            // Show details section
            document.getElementById('training-details').style.display = 'block';
        }}
        
        // Helper functions
        function getAxisLabel(field) {{
            const labels = {{
                'date': 'Date',
                'workout_type': 'Workout Type',
                'duration_minutes': 'Duration (minutes)',
                'training_load': 'Training Load (TSS)',
                'avg_power': 'Average Power (W)',
                'avg_heart_rate': 'Average Heart Rate (bpm)',
                'intensity_factor': 'Intensity Factor'
            }};
            return labels[field] || field;
        }}
        
        function getColor(value, field) {{
            // Simple color mapping - you can enhance this
            if (field === 'workout_type') {{
                const colors = {{
                    'Threshold Intervals': '#ff6b6b',
                    'VO2 Max': '#4ecdc4',
                    'Endurance': '#45b7d1',
                    'Recovery': '#96ceb4',
                    'Tempo': '#feca57'
                }};
                return colors[value] || '#778ca3';
            }}
            return '#007acc';
        }}
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializePlot);
    </script>
</body>
</html>
"""

def save_trainings_json(trainings_data, output_dir):
    """Save trainings data as JSON for external use"""
    json_file = output_dir / "trainings.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(trainings_data, f, indent=2)

if __name__ == "__main__":
    create_an_html_report()
