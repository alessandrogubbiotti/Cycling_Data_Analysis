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

### `organizer.py`

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

