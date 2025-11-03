# Cycling Data Analysis System

A comprehensive system for organizing, analyzing, and managing Zwift cycling training data with automated parsing and structured storage.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Project Structure

```
Cycling_Data_Analysis/
├── Athlete/
│   ├── Athlete_Name_1/
│   │   ├── AllData/
│   │   ├── ParsedData/
│   │   ├── by_month/
│   │   └── by_training/
│   └── Athlete_Name_2/
│       ├── AllData/
│       ├── ParsedData/
│       ├── by_month/
│       └── by_training/
├── Trainings/
├── organizer.py
├── remove_training.py
└── README.md
```

<details>
<summary>Quick Start</summary>

### Prerequisites

* Python 3.7+
* Install dependencies:

  ```bash
  pip install fitparse
  ```

### Adding Training Data

```bash
python organizer.py
```

### Removing Training Data

```bash
python remove_training.py
```

</details>

<details>
<summary>Scripts</summary>

### `import_and_parse.py`

Main script for importing new training sessions. It:

* Parses FIT files into CSV and JSON formats
* Creates organized directory structure
* Collects training metadata
* Handles duplicate training names

### `remove_training.py`

Safe training removal with:

* Month-based browsing interface
* Bulk removal capabilities
* Training type display
* Safety confirmations

### 'plotter.py'
* Contains the plotting class 

### 'zwo_parser.py'
* From the guideline of the workout it extracts the intervlas

### 'interval_finder' 
* Defines the interval class
* The interval finder function in the class using the parsed data finds the intervals in the actual training
* This last feature is by now implemented in a naivce and non-flexible way: it assumes that the guidelined of trainer are followed from the beginning to the end

### 'statistics.py'

* The relevant statistics are computed
</details>

<details>
<summary>Data Outputs</summary>
- `session_metrics.csv`: Time-independent metrics (power, heart rate, etc.)
- `time_series.csv`: Record-by-record data (timestamp, power, cadence, etc.)
- `metadata.json`: Training context (nutrition, hydration, effort, notes)
</details>

<details>
<summary>Training Types</summary>
- Z2 (Endurance)
- Z3 (Tempo)
- Z4 (Threshold)
- Climb
- Sprint
</details>

<details>
<summary>GoldenCheetah Integration</summary>
Point GoldenCheetah to the `AllData/` directory or use the helper script to open files directly.
</details>

<details>
<summary>Workflow</summary>
1. Complete Zwift training → Get `.fit` file
2. Run `organizer.py` → Import and parse data
3. Add training metadata → Nutrition, effort, notes
4. Analyze → Use GoldenCheetah or custom scripts
5. Organize → Automatic symlinks by month and type
6. Remove if needed → Use `remove_training.py`
</details>


<details>
<summary>Some future improvements of the code</summary>
1. Create the intervals class as a list of intervals. It should contain the read abstract training, statistics--
2. Create the global statistics class 
3. Refine the plot class
4. Define a class with many trainings 
5. Introduce some algorithm to recognize the intervals to classify non guided trainings
6. Add some metrics (Blood sugar, CO2 emitted (I should buy the Calibre gadget))
</details>

