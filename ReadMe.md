# 🚴 Cycling Data Analysis

> 🌐 **Live Dashboard:** [View Interactive Dashboard](https://alessandrogubbiotti.github.io/Cycling_Data_Analysis/)

A system vor visualizing and analyzing cycling data taken (for now) from zwift. The work is under construction and for now only some plotting tools and some basic statistics are available

![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Project Structure

```
Cycling_Data_Analysis/
├── Athlete/
│   ├── Athlete_Name_1/
│   │   ├── athlete_info.json
│   │   ├── AllData/
│   │   ├── ParsedData/
│   │   ├── by_month/
│   │   ├── by_zone/
│   │   └── by_training/
│   └── Athlete_Name_2/
│       ├── AllData/
│       ├── ParsedData/
│       ├── by_month/
│       ├── by_zone/
│       └── by_training/
├── Trainings/
│   └── Z2_High_Cadence/
│       ├── Z2_High_Cadence.zwo 
│       ├── Z2_High_Cadence.xml
│       ├── Z2_High_Cadence.json 
│       └── Z2_High_Cadence.txt
├── import_zwo.py
├── remove_zwo.py
├── import_and_parse.py
├── remove_training.py
├── create_athlete.py
├── training_class.py
├── interval_class.py
├── plotter_class.py
└── README.md
```

<details>
<summary>Explaination</summary>
### Trainings
Contains the different training templates, for now only the ones extracted from zwift
They can (and should) be imported and removed using the import\_zwo.py and the remove\_zwo.py, respecticely

### Athlete

Contains the different athletes.  new athlete should be added using create\_athlete.py. 
A new training should be loaded using import\_and\_parse.py and removed using remove\_trainind.py. Once imported, the analysis is done by calling the main  
<details>


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
python load_and_parse.py
```

### Removing Training Data

```bash
python remove_training.py
```

</details>

</details>
<summary>Structure of the code <summary>

### Organizer Tools

* import.py: Takes in input the .fit file and creates the ParseData folder were the traing data will be stored and the symlinks in the by\_month and by\_type folders to the .fit file and the folder of the training 

* remove.py: Removes the training

* add\_and\_parse\_workout.py The function that from a .zwo file outputs the parsed trainign as a sequence of intervals 

### Training Class: 
```
Training
├── metadata  	 	 # Contains the metadata
├── read_metadata(.json)  	 	 # A functions that reads the metadata from the metadata.json file 
├── data # A pandas object that contains the data from the .fit file and then enhanced with some interval data 
├── read_data()
├── intervals # An intervals object which is a defined latet. Basically it is a sequential list of intervals with information about them
├── interval_finder # It divides the traininf into intervals and marks the initinall and final times of the intervals, its kind and the zone. Now it simply reads the .json file parsed from the .zwo and we place them sequentially from time zero (we are assuming hte training is guided and we start the training at the same time of the guide without stops)
├── interval_statistics # a class containing the fucntions that fill the interval object with statistics using the dat 
├── intervals # An intervals object which is a defined latet
├── plotter # A plotter object that visualizes the training 
└── README.md
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
# 🚧 Work in Progress

## Current Status
Building interactive cycling data analysis dashboard with:

### ✅ Completed
- Data pipeline from .fit files
- Basic HTML report generation  
- GitHub Actions automation

### 🚧 In Progress
- Interactive training visualizations
- Drill-down training analysis
- Multi-dimensional plots

### 📋 Planned
- [ ] Interval search across sessions
- [ ] Progress tracking over time
- [ ] Workout comparison tools

