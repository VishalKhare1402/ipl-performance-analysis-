# 🏏 IPL Team & Player Performance Analysis

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Library-NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-11557C?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> A Python-based sports analytics project that digs into IPL (Indian Premier League) data to uncover team and player performance trends using statistical analysis and data visualization.

---

## 📌 Project Overview

This project analyzes IPL match and player datasets to identify performance patterns, consistency metrics, and trends across teams and players. Using Python's data science stack (Pandas, NumPy, Matplotlib), the project covers the full pipeline from raw data to visual insights.

---

## 🎯 Objectives

- Clean and preprocess IPL datasets using Pandas
- Calculate key statistical metrics for players and teams
- Analyze performance variability and consistency
- Visualize trends using Matplotlib charts

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core programming language |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Statistical calculations |
| Matplotlib | Data visualization |
| Jupyter Notebook | Interactive development environment |

---

## 📂 Project Structure

```
ipl-performance-analysis/
│
├── data/
│   ├── matches.csv              # Match-level data (teams, scores, results)
│   └── deliveries.csv           # Ball-by-ball delivery data
│
├── notebooks/
│   └── ipl_analysis.ipynb       # Full analysis notebook
│
├── scripts/
│   ├── data_cleaning.py         # Data preprocessing script
│   ├── player_analysis.py       # Player-level metrics
│   └── team_analysis.py         # Team-level metrics
│
├── outputs/
│   ├── top_batsmen.png
│   ├── team_win_rates.png
│   ├── run_rate_trends.png
│   └── player_consistency.png
│
└── README.md
```

---

## 🔍 Key Analysis

### 1. Data Cleaning
```python
import pandas as pd
import numpy as np

# Load datasets
matches = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

# Check for nulls
print(matches.isnull().sum())
print(deliveries.isnull().sum())

# Drop irrelevant columns
matches.drop(columns=['umpire3'], inplace=True)

# Standardize team names (handle name changes over the years)
team_name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
}
matches['team1'] = matches['team1'].replace(team_name_map)
matches['team2'] = matches['team2'].replace(team_name_map)
```

### 2. Statistical Metrics (NumPy)
```python
# Batting stats per player
batting_stats = deliveries.groupby('batsman').agg(
    total_runs=('batsman_runs', 'sum'),
    balls_faced=('ball', 'count'),
    innings=('match_id', 'nunique')
).reset_index()

# Strike Rate
batting_stats['strike_rate'] = (batting_stats['total_runs'] / batting_stats['balls_faced']) * 100

# Average
batting_stats['average'] = batting_stats['total_runs'] / batting_stats['innings']

# Consistency (Standard deviation of runs per match)
runs_per_match = deliveries.groupby(['batsman', 'match_id'])['batsman_runs'].sum().reset_index()
consistency = runs_per_match.groupby('batsman')['batsman_runs'].std().reset_index()
consistency.columns = ['batsman', 'std_dev_runs']

batting_stats = batting_stats.merge(consistency, on='batsman')
```

### 3. Team Win Rate Analysis
```python
# Win percentage by team
total_matches = pd.concat([matches['team1'], matches['team2']]).value_counts().reset_index()
total_matches.columns = ['team', 'total_played']

wins = matches['winner'].value_counts().reset_index()
wins.columns = ['team', 'wins']

team_performance = total_matches.merge(wins, on='team')
team_performance['win_rate'] = (team_performance['wins'] / team_performance['total_played']) * 100
team_performance = team_performance.sort_values('win_rate', ascending=False)
```

### 4. Visualization
```python
import matplotlib.pyplot as plt

# Top 10 run scorers
top_batsmen = batting_stats.nlargest(10, 'total_runs')

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top_batsmen['batsman'], top_batsmen['total_runs'], color='steelblue')
ax.set_xlabel('Total Runs')
ax.set_title('Top 10 Run Scorers in IPL')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/top_batsmen.png', dpi=150)
plt.show()
```

---

## 📈 Key Insights

- **Most Consistent Batsman:** Low std deviation in runs-per-match = high consistency
- **Best Win Rate:** Mumbai Indians and Chennai Super Kings historically lead in win rates
- **Strike Rate Leaders:** T20 format rewards high strike rates (>140) over averages
- **Toss Impact:** Toss winners chose to field ~58% of the time in recent seasons

---

## 💡 Skills Demonstrated

- Data loading and cleaning with Pandas
- Statistical analysis using NumPy (mean, std dev, correlation)
- Performance benchmarking across categories
- Clear data visualization with Matplotlib
- End-to-end Python analytics pipeline

---

## 🚀 How to Use

### Prerequisites
```bash
pip install pandas numpy matplotlib jupyter
```

### Run the Notebook
```bash
git clone https://github.com/VishalKhare1402/ipl-performance-analysis.git
cd ipl-performance-analysis
jupyter notebook notebooks/ipl_analysis.ipynb
```

### Dataset Source
- Download IPL dataset from [Kaggle – IPL Dataset](https://www.kaggle.com/datasets/nowke9/ipldata)
- Place `matches.csv` and `deliveries.csv` into the `data/` folder

---

## 📸 Sample Outputs

> *(Add your chart screenshots here after running the notebook)*

| Chart | Description |
|-------|-------------|
| `top_batsmen.png` | Top 10 run scorers horizontal bar chart |
| `team_win_rates.png` | Win rate comparison across all teams |
| `run_rate_trends.png` | Season-wise run rate trends |
| `player_consistency.png` | Strike rate vs consistency scatter plot |

---

## 👤 Author

**Vishal Khare**
📧 vishalkhare54@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/vishalkhare9bb31436b)
🐙 [GitHub](https://github.com/VishalKhare1402)
