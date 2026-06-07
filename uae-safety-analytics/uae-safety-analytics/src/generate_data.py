"""
UAE Public Safety & Road Incident Analytics
Data Generator — creates realistic synthetic datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

EMIRATES = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"]
EMIRATE_POP = {
    "Dubai": 3600000, "Abu Dhabi": 2900000, "Sharjah": 1800000,
    "Ajman": 540000, "Ras Al Khaimah": 380000, "Fujairah": 270000, "Umm Al Quwain": 90000
}

INCIDENT_TYPES = [
    "Speeding", "Reckless Driving", "Tailgating", "Signal Violation",
    "Parking Violation", "Pedestrian Accident", "Vehicle Collision",
    "DUI", "Wrong Way Driving", "Distracted Driving"
]

ROAD_TYPES = ["Highway", "Urban Street", "Residential", "Industrial Zone", "Coastal Road"]
TIME_SLOTS = ["Early Morning (0-6)", "Morning Rush (6-9)", "Midday (9-15)",
              "Evening Rush (15-19)", "Night (19-24)"]
WEATHER = ["Clear", "Sandstorm", "Fog", "Rain", "Heatwave"]
VEHICLE_TYPES = ["Sedan", "SUV", "Truck", "Motorcycle", "Bus", "Van"]

def generate_road_incidents(n=5000):
    start = datetime(2019, 1, 1)
    rows = []
    for _ in range(n):
        emirate = np.random.choice(EMIRATES, p=[0.35, 0.28, 0.17, 0.07, 0.06, 0.04, 0.03])
        date = start + timedelta(days=random.randint(0, 365*5))
        month = date.month
        hour_probs = np.array([
            0.02,0.01,0.01,0.01,0.02,0.03,0.05,0.07,0.06,0.05,0.05,0.05,
            0.05,0.05,0.05,0.06,0.07,0.07,0.06,0.05,0.04,0.04,0.03,0.02
        ])
        hour_probs = hour_probs / hour_probs.sum()
        hour = np.random.choice(range(24), p=hour_probs)
        # Fog more in winter, sandstorms in spring
        if month in [12, 1, 2]:
            weather_probs = [0.5, 0.1, 0.25, 0.1, 0.05]
        elif month in [3, 4, 5]:
            weather_probs = [0.45, 0.3, 0.05, 0.05, 0.15]
        else:
            weather_probs = [0.6, 0.15, 0.05, 0.02, 0.18]
        
        weather = np.random.choice(WEATHER, p=weather_probs)
        severity_base = random.uniform(1, 5)
        if weather in ["Sandstorm", "Fog"]: severity_base += 0.8
        if hour in [0, 1, 2, 3]: severity_base += 0.5
        severity = min(5, max(1, round(severity_base)))
        
        fatalities = 0
        if severity == 5: fatalities = np.random.choice([0,1,2,3], p=[0.6,0.25,0.1,0.05])
        elif severity == 4: fatalities = np.random.choice([0,1], p=[0.85,0.15])

        rows.append({
            "incident_id": f"INC-{_+1:05d}",
            "date": date.strftime("%Y-%m-%d"),
            "year": date.year,
            "month": date.month,
            "month_name": date.strftime("%B"),
            "hour": hour,
            "time_slot": TIME_SLOTS[0 if hour<6 else 1 if hour<9 else 2 if hour<15 else 3 if hour<19 else 4],
            "emirate": emirate,
            "incident_type": np.random.choice(INCIDENT_TYPES),
            "road_type": np.random.choice(ROAD_TYPES, p=[0.35,0.3,0.15,0.1,0.1]),
            "vehicle_type": np.random.choice(VEHICLE_TYPES, p=[0.35,0.3,0.15,0.1,0.05,0.05]),
            "weather_condition": weather,
            "severity_score": severity,
            "fatalities": fatalities,
            "injuries": random.randint(0, severity * 2),
            "response_time_min": round(np.random.normal(8 if emirate=="Dubai" else 12, 3), 1),
            "fine_aed": random.choice([400, 600, 800, 1000, 2000, 3000]) if severity < 4 else 0,
        })
    return pd.DataFrame(rows)

def generate_safety_index(n=7*5):
    rows = []
    for emirate in EMIRATES:
        for year in range(2019, 2025):
            pop = EMIRATE_POP[emirate]
            base_score = {"Dubai": 82, "Abu Dhabi": 85, "Sharjah": 78,
                         "Ajman": 72, "Ras Al Khaimah": 74, "Fujairah": 76, "Umm Al Quwain": 73}[emirate]
            trend = (year - 2019) * 1.2
            rows.append({
                "emirate": emirate,
                "year": year,
                "population": pop + random.randint(-10000, 50000),
                "safety_index": round(min(100, base_score + trend + random.uniform(-2, 2)), 1),
                "incidents_per_100k": round(random.uniform(80, 200) - trend * 3, 1),
                "avg_response_time_min": round(random.uniform(6, 15) - trend * 0.1, 1),
                "cctv_cameras": int(pop * random.uniform(0.005, 0.015)),
                "police_stations": random.randint(3, 45),
                "emergency_calls": int(pop * random.uniform(0.02, 0.06)),
            })
    return pd.DataFrame(rows)

def generate_weather_impact():
    rows = []
    for weather in WEATHER:
        for road in ROAD_TYPES:
            rows.append({
                "weather": weather,
                "road_type": road,
                "avg_severity": round(random.uniform(1.5, 4.5) + (1.2 if weather in ["Fog","Sandstorm"] else 0), 2),
                "incident_count": random.randint(50, 800),
                "fatality_rate_pct": round(random.uniform(0.5, 8.0), 2),
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Generating datasets...")
    incidents = generate_road_incidents(5000)
    incidents.to_csv("/home/claude/uae-safety-analytics/data/uae_road_incidents_2019_2024.csv", index=False)
    print(f"✓ Road incidents: {len(incidents)} rows")

    safety = generate_safety_index()
    safety.to_csv("/home/claude/uae-safety-analytics/data/uae_emirate_safety_index.csv", index=False)
    print(f"✓ Safety index: {len(safety)} rows")

    weather = generate_weather_impact()
    weather.to_csv("/home/claude/uae-safety-analytics/data/uae_weather_road_impact.csv", index=False)
    print(f"✓ Weather impact: {len(weather)} rows")
    print("All datasets saved.")
