#!/usr/bin/env python3

import os
import json
from pathlib import Path
from datetime import datetime

def create_an_html_report():
    """
    Creates an HTML report focusing on the cycling data analysis pipeline.
    Other sections are commented out for future implementation.
    """
    
    # Create output directory
    output_dir = Path("_build/html")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML content
    html_content = generate_html_content()
    
    # Write the HTML file
    output_file = output_dir / "index.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Pipeline HTML report generated at: {output_file}")
    return output_file

def generate_html_content():
    """Generate the HTML content with only pipeline section implemented"""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cycling Data Analysis Pipeline</title>
    
    <!-- Mermaid.js for pipeline diagrams -->
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
    
    <style>
        /* Base styles */
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}
        
        .header h1 {{
            color: #333;
            margin: 0;
            font-size: 2.5em;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.2em;
            margin: 10px 0 0 0;
        }}
        
        /* Pipeline section */
        .pipeline-section {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        
        .pipeline-section h2 {{
            color: #333;
            margin-top: 0;
        }}
        
        .mermaid-container {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        
        /* Commented sections */
        .commented-section {{
            margin: 40px 0;
            padding: 25px;
            background: #f0f0f0;
            border-radius: 8px;
            border-left: 5px solid #ccc;
            color: #666;
        }}
        
        .commented-section h2 {{
            color: #888;
            font-style: italic;
        }}
        
        .comment-note {{
            background: #ffeb3b;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #ffc107;
        }}
        
        .timestamp {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Section -->
        <div class="header">
            <h1>🚴 Cycling Data Analysis Pipeline</h1>
            <p>Visualizing the complete data flow from .fit files to insights</p>
        </div>
        
        <!-- PIPELINE VISUALIZATION SECTION -->
        <div class="pipeline-section">
            <h2>Data Processing Pipeline</h2>
            <p>This diagram shows the complete flow of data through our analysis system:</p>
            
            <div class="mermaid-container">
                <pre class="mermaid">
graph TD
    A[📁 Raw .fit Files] --> B[🔄 Data Parser]
    B --> C[🏗️ Data Organizer]
    C --> D[📊 Metadata Creation]
    C --> E[💪 Workout Structure]
    D --> F[📈 Interval Extraction]
    E --> F
    F --> G[🔍 Subinterval Analysis]
    G --> H[📐 Statistics Computation]
    H --> I[📊 Interval Statistics]
    H --> J[🌍 Global Statistics]
    I --> K[🖼️ Visualization Generation]
    J --> K
    K --> L[🌐 HTML Report]
    L --> M[🚀 GitHub Pages Deployment]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#e0f2f1
    style G fill:#fff8e1
    style H fill:#f1f8e9
    style I fill:#e8eaf6
    style J fill:#fff3e0
    style K fill:#f3e5f5
    style L fill:#e8f5e8
    style M fill:#ffebee,stroke:#f44336,stroke-width:2px
                </pre>
            </div>
            
            <div class="comment-note">
                <strong>Pipeline Explanation:</strong> This automated workflow transforms raw cycling data into actionable insights through multiple processing stages.
            </div>
        </div>
        
        <!-- COMMENTED: INTERVAL ANALYSIS SECTION -->
        <!--
        <div class="commented-section">
            <h2>Interval Analysis</h2>
            <p>This section will display hierarchical interval statistics and analysis.</p>
            <div class="comment-note">
                TODO: Implement interval statistics tables and hierarchical visualization
            </div>
        </div>
        -->
        
        <!-- COMMENTED: TRAINING OVERVIEW SECTION -->
        <!--
        <div class="commented-section">
            <h2>Training Overview</h2>
            <p>This section will show training session metrics and summaries.</p>
            <div class="comment-note">
                TODO: Add training metrics cards, session tables, and performance trends
            </div>
        </div>
        -->
        
        <!-- COMMENTED: WORKOUT ANALYSIS SECTION -->
        <!--
        <div class="commented-section">
            <h2>Workout Analysis</h2>
            <p>This section will analyze workout structures and tracking.</p>
            <div class="comment-note">
                TODO: Implement workout creation tracking and inverse procedure analysis
            </div>
        </div>
        -->
        
        <!-- COMMENTED: VISUALIZATION GALLERY -->
        <!--
        <div class="commented-section">
            <h2>Performance Visualizations</h2>
            <p>This section will display all generated plots and charts.</p>
            <div class="comment-note">
                TODO: Add plot gallery with power distribution, heart rate zones, interval analysis, etc.
            </div>
        </div>
        -->
        
        <div class="timestamp">
            Report generated on: {current_time}
        </div>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    create_an_html_report()
