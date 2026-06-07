# 🇦🇪 UAE Road Safety & Incident Analytics (2019–2024)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Level](https://img.shields.io/badge/Level-Intermediate-yellow)

> A comprehensive data science project analyzing road incident patterns, weather impacts, emirate-level safety performance, and machine learning-driven severity prediction across the UAE from 2019 to 2024.

---

## 📌 Project Summary

This project combines **exploratory data analysis**, **supervised ML classification**, and **unsupervised clustering** to surface actionable insights from UAE road safety data. It's designed to simulate the kind of work a data analyst or data scientist would do in a government, smart city, or transportation analytics context.

**Dataset:** Synthetic, UAE-inspired (5,000 road incident records across 7 emirates)  
**Period:** January 2019 – December 2024  
**Domain:** Public Safety | Smart Cities | Transportation Analytics

---

## 🎯 Objectives

- Identify high-risk time periods, locations, and weather conditions
- Build a multi-class ML model to predict incident severity (1–5 scale)
- Cluster incidents into meaningful risk profiles using K-Means
- Compare safety performance across all 7 UAE emirates
- Visualize trends, correlations, and patterns with publication-quality charts

---

## 📁 Project Structure

```
uae-safety-analytics/
│
├── data/
│   ├── uae_road_incidents_2019_2024.csv   # Main incidents dataset (5,000 rows)
│   ├── uae_emirate_safety_index.csv       # Safety index by emirate & year
│   └── uae_weather_road_impact.csv        # Weather × road type impact matrix
│
├── src/
│   ├── generate_data.py                   # Synthetic data generation
│   └── analysis.py                        # Full analysis pipeline (EDA + ML)
│
├── reports/
│   ├── 01_executive_dashboard.png         # KPIs, trends, heatmaps
│   ├── 02_weather_impact_analysis.png     # Weather × severity analysis
│   ├── 03_ml_severity_prediction.png      # RF model results
│   ├── 04_emirate_safety_index.png        # Emirate comparison
│   ├── 05_temporal_patterns.png           # Time-based analysis
│   └── 06_risk_clustering.png             # K-Means clusters
│
├── dashboard/
│   └── index.html                         # Interactive HTML dashboard
│
├── requirements.txt
└── README.md
```

---

## 📊 Analyses Performed

### 1. Executive Overview Dashboard
- Annual incident and fatality trends (2019–2024)
- Incident breakdown by emirate and incident type
- Severity heatmap by day-of-week × time-of-day

### 2. Weather & Environmental Impact
- Average severity by weather condition (Clear, Fog, Sandstorm, Rain, Heatwave)
- Monthly × weather severity matrix
- Road type × weather severity cross-analysis

### 3. ML: Severity Classification (Random Forest)
- 5-class severity prediction (1 = minor → 5 = fatal)
- Feature importance analysis
- Confusion matrix + 5-fold cross-validation

### 4. Emirate Safety Index Analysis
- Safety index trends per emirate (2019–2024)
- Incidents per 100,000 population
- Emergency response time improvements
- CCTV density vs safety index correlation (Pearson r ≈ 0.60)

### 5. Temporal Pattern Analysis
- Hourly incident distribution by severity level
- Monthly fatality rate with seasonal patterns
- Year-over-year monthly comparison
- Vehicle type severity boxplots

### 6. K-Means Risk Profiling
- 4 distinct risk clusters identified
- Cluster profiles: Low / Moderate / Elevated / High Risk
- Cluster distribution by emirate and time of day

---

## 🤖 Machine Learning

| Model | Task | Accuracy (CV) |
|-------|------|--------------|
| Random Forest | Severity Classification (5-class) | 44.54% ± ~1% |
| K-Means | Risk Profile Clustering | 4 clusters (Silhouette optimized) |

**Features used:**
- Emirate, incident type, road type, vehicle type, weather, time slot (all label-encoded)
- Hour, month, fatalities, injuries, response time, fine amount

---

## 🔑 Key Findings

| Finding | Detail |
|---------|--------|
| 🌪️ Most dangerous weather | **Sandstorm** conditions → highest avg severity |
| 🌙 Most dangerous time | **Early morning (0–6 AM)** — fatigue & low visibility |
| 📈 Safety trend | All emirates improving; **Abu Dhabi** leads (85+ index) |
| 📡 Infrastructure | Higher CCTV density correlates with better safety index |
| 🚗 Vehicle risk | **Motorcycles & Trucks** show highest severity variance |
| 📅 Seasonal spike | **Winter months** (Dec–Feb) show elevated fatality rates (fog) |

---

## 🚀 Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Run Data Generation

```bash
python src/generate_data.py
```

### Run Full Analysis

```bash
python src/analysis.py
```

All charts will be saved to `reports/`.  
Open `dashboard/index.html` in a browser to view the full interactive dashboard.

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.11
```

---

## 🗂️ Dataset Overview

### `uae_road_incidents_2019_2024.csv` (5,000 rows)

| Column | Type | Description |
|--------|------|-------------|
| `incident_id` | str | Unique ID (INC-00001 format) |
| `date` | date | Incident date |
| `year`, `month`, `hour` | int | Temporal breakdown |
| `emirate` | str | One of 7 UAE emirates |
| `incident_type` | str | Speeding, DUI, Collision, etc. |
| `road_type` | str | Highway, Urban, Residential, etc. |
| `vehicle_type` | str | Sedan, SUV, Truck, Motorcycle, etc. |
| `weather_condition` | str | Clear, Fog, Sandstorm, Rain, Heatwave |
| `severity_score` | int | 1 (minor) – 5 (fatal), target variable |
| `fatalities` | int | Number of deaths |
| `injuries` | int | Number of injuries |
| `response_time_min` | float | Emergency response time (minutes) |
| `fine_aed` | int | Traffic fine in AED |

---

## 💡 Potential Extensions

- [ ] Deploy an ML severity prediction API (FastAPI + Uvicorn)
- [ ] Add Folium/Plotly geo-mapping of incident hotspots
- [ ] Build a Streamlit or Dash interactive app
- [ ] Add ARIMA/Prophet time series forecasting for monthly incidents
- [ ] Hyperparameter tuning with GridSearchCV

---

## 👩‍💻 Author

**Hasna Hussain (Hasu)**  
MSc Data Science & AI — Middlesex University Dubai  
📍 UAE | [LinkedIn](https://www.linkedin.com/in/hasna-hussain-29a631221) | [GitHub](https://github.com/hasnahussain)

---

## 📜 License

This project is for educational and portfolio purposes. The dataset is synthetically generated and does not represent real incident data.
