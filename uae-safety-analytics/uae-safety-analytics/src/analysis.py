"""
UAE Public Safety & Road Incident Analytics
Full Analysis Pipeline: EDA + ML + Insights
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              mean_absolute_error, r2_score)
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("/home/claude/uae-safety-analytics/reports", exist_ok=True)

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
PALETTE = {
    "bg":       "#0D1117",
    "surface":  "#161B22",
    "border":   "#30363D",
    "accent1":  "#58A6FF",
    "accent2":  "#F78166",
    "accent3":  "#3FB950",
    "accent4":  "#D2A8FF",
    "accent5":  "#FFA657",
    "text":     "#E6EDF3",
    "muted":    "#8B949E",
}

EMIRATE_COLORS = {
    "Dubai":           "#58A6FF",
    "Abu Dhabi":       "#3FB950",
    "Sharjah":         "#FFA657",
    "Ajman":           "#F78166",
    "Ras Al Khaimah":  "#D2A8FF",
    "Fujairah":        "#79C0FF",
    "Umm Al Quwain":   "#56D364",
}

def apply_dark_theme(fig, axes=None):
    fig.patch.set_facecolor(PALETTE["bg"])
    if axes is None:
        return
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["muted"])
        ax.yaxis.label.set_color(PALETTE["muted"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])

def save_fig(name):
    path = f"/home/claude/uae-safety-analytics/reports/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved: {name}.png")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("\n📥 Loading data...")
df = pd.read_csv("/home/claude/uae-safety-analytics/data/uae_road_incidents_2019_2024.csv")
si = pd.read_csv("/home/claude/uae-safety-analytics/data/uae_emirate_safety_index.csv")
wi = pd.read_csv("/home/claude/uae-safety-analytics/data/uae_weather_road_impact.csv")
df["date"] = pd.to_datetime(df["date"])
print(f"  Incidents: {len(df):,} rows | Safety Index: {len(si)} rows | Weather: {len(wi)} rows")

# ─────────────────────────────────────────────
# 1. OVERVIEW DASHBOARD
# ─────────────────────────────────────────────
print("\n📊 Chart 1: Executive Overview Dashboard")
fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor(PALETTE["bg"])
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

# KPI tiles
kpis = [
    ("Total Incidents", f"{len(df):,}", PALETTE["accent1"]),
    ("Total Fatalities", f"{df['fatalities'].sum():,}", PALETTE["accent2"]),
    ("Avg Severity", f"{df['severity_score'].mean():.2f}/5", PALETTE["accent5"]),
    ("Avg Response", f"{df['response_time_min'].mean():.1f} min", PALETTE["accent3"]),
]
for i, (label, val, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(PALETTE["surface"])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(plt.Rectangle((0,0),1,1, color=PALETTE["surface"]))
    ax.text(0.5, 0.62, val, ha="center", va="center", fontsize=22, fontweight="bold",
            color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=9,
            color=PALETTE["muted"], transform=ax.transAxes)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)

# Yearly trend
ax2 = fig.add_subplot(gs[1, :2])
yearly = df.groupby("year").agg(incidents=("incident_id","count"), fatalities=("fatalities","sum")).reset_index()
ax2.fill_between(yearly["year"], yearly["incidents"], alpha=0.25, color=PALETTE["accent1"])
ax2.plot(yearly["year"], yearly["incidents"], "o-", color=PALETTE["accent1"], linewidth=2.5, markersize=7, label="Incidents")
ax2b = ax2.twinx()
ax2b.plot(yearly["year"], yearly["fatalities"], "s--", color=PALETTE["accent2"], linewidth=2, markersize=6, label="Fatalities")
ax2b.tick_params(colors=PALETTE["muted"])
ax2b.yaxis.label.set_color(PALETTE["muted"])
for spine in ax2b.spines.values(): spine.set_edgecolor(PALETTE["border"])
ax2b.set_facecolor(PALETTE["surface"])
ax2.set_title("Annual Incidents & Fatalities", color=PALETTE["text"], fontsize=11, pad=8)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=8,
           facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"], edgecolor=PALETTE["border"])
apply_dark_theme(fig, ax2)

# By Emirate
ax3 = fig.add_subplot(gs[1, 2:])
em_counts = df.groupby("emirate")["incident_id"].count().sort_values(ascending=True)
colors = [EMIRATE_COLORS.get(e, PALETTE["accent1"]) for e in em_counts.index]
bars = ax3.barh(em_counts.index, em_counts.values, color=colors, height=0.65)
for bar, val in zip(bars, em_counts.values):
    ax3.text(val + 15, bar.get_y() + bar.get_height()/2, f"{val:,}",
             va="center", color=PALETTE["muted"], fontsize=8)
ax3.set_title("Incidents by Emirate", color=PALETTE["text"], fontsize=11, pad=8)
apply_dark_theme(fig, ax3)

# Incident type donut
ax4 = fig.add_subplot(gs[2, :2])
type_counts = df["incident_type"].value_counts()
wedge_colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
                PALETTE["accent4"], PALETTE["accent5"],
                "#79C0FF", "#56D364", "#FFA657", "#F78166", "#A5D6FF"]
wedges, texts, autotexts = ax4.pie(
    type_counts.values, labels=None,
    autopct=lambda p: f"{p:.0f}%" if p > 6 else "",
    colors=wedge_colors[:len(type_counts)],
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2)
)
for at in autotexts: at.set_color(PALETTE["bg"]); at.set_fontsize(8); at.set_fontweight("bold")
ax4.set_title("Incident Type Distribution", color=PALETTE["text"], fontsize=11, pad=8)
ax4.legend(type_counts.index, loc="lower center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, fontsize=7, facecolor=PALETTE["surface"],
           labelcolor=PALETTE["muted"], edgecolor=PALETTE["border"])
ax4.set_facecolor(PALETTE["surface"])

# Severity heatmap by hour & day
ax5 = fig.add_subplot(gs[2, 2:])
df["weekday"] = df["date"].dt.day_name()
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
hour_bins = [0,6,9,15,19,24]
hour_labels = ["0-6","6-9","9-15","15-19","19-24"]
df["hour_bin"] = pd.cut(df["hour"], bins=hour_bins, labels=hour_labels, right=False, include_lowest=True)
pivot = df.pivot_table(values="severity_score", index="weekday", columns="hour_bin", aggfunc="mean")
pivot = pivot.reindex(weekday_order)
sns.heatmap(pivot, ax=ax5, cmap="YlOrRd", linewidths=0.3, linecolor=PALETTE["bg"],
            cbar_kws={"shrink":0.8}, annot=True, fmt=".1f", annot_kws={"size":8})
ax5.set_title("Avg Severity: Day × Time", color=PALETTE["text"], fontsize=11, pad=8)
ax5.tick_params(colors=PALETTE["muted"], labelsize=8)
ax5.set_xlabel("Time Slot", color=PALETTE["muted"])
ax5.set_ylabel("")
ax5.set_facecolor(PALETTE["surface"])
cbar = ax5.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["muted"])

fig.suptitle("🇦🇪  UAE Road Safety Analytics  |  2019–2024",
             fontsize=16, fontweight="bold", color=PALETTE["text"], y=0.98)
save_fig("01_executive_dashboard")

# ─────────────────────────────────────────────
# 2. WEATHER IMPACT ANALYSIS
# ─────────────────────────────────────────────
print("\n📊 Chart 2: Weather & Environmental Impact")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
apply_dark_theme(fig, axes)

# Weather severity
weather_sev = df.groupby("weather_condition").agg(
    avg_severity=("severity_score","mean"),
    count=("incident_id","count"),
    fatality_rate=("fatalities", lambda x: x.sum()/len(x)*100)
).reset_index().sort_values("avg_severity", ascending=False)

colors_w = [PALETTE["accent2"] if w in ["Sandstorm","Fog"] else PALETTE["accent1"] for w in weather_sev["weather_condition"]]
bars = axes[0].bar(weather_sev["weather_condition"], weather_sev["avg_severity"], color=colors_w, width=0.6)
axes[0].axhline(weather_sev["avg_severity"].mean(), color=PALETTE["accent3"], linestyle="--", alpha=0.7, linewidth=1.5, label="Mean")
axes[0].set_title("Avg Severity by Weather", fontsize=11)
axes[0].set_ylabel("Severity Score (1-5)", color=PALETTE["muted"])
axes[0].legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"], edgecolor=PALETTE["border"])
axes[0].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, weather_sev["avg_severity"]):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04, f"{val:.2f}",
                 ha="center", fontsize=8, color=PALETTE["muted"])

# Monthly heatmap
monthly_weather = df.groupby(["month_name","weather_condition"])["severity_score"].mean().unstack(fill_value=0)
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
monthly_weather = monthly_weather.reindex([m for m in month_order if m in monthly_weather.index])
sns.heatmap(monthly_weather, ax=axes[1], cmap="plasma", linewidths=0.2,
            linecolor=PALETTE["bg"], annot=True, fmt=".1f", annot_kws={"size":7})
axes[1].set_title("Severity: Month × Weather", fontsize=11)
axes[1].tick_params(colors=PALETTE["muted"], labelsize=8)
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_facecolor(PALETTE["surface"])

# Weather vs road type
wrt = wi.pivot(index="road_type", columns="weather", values="avg_severity")
sns.heatmap(wrt, ax=axes[2], cmap="coolwarm", linewidths=0.3,
            linecolor=PALETTE["bg"], annot=True, fmt=".1f", annot_kws={"size":8},
            center=wrt.values.mean())
axes[2].set_title("Severity: Road Type × Weather", fontsize=11)
axes[2].tick_params(colors=PALETTE["muted"], labelsize=8)
axes[2].set_xlabel("")
axes[2].set_ylabel("")
axes[2].set_facecolor(PALETTE["surface"])

fig.suptitle("Weather & Environmental Impact on Road Incidents",
             fontsize=14, color=PALETTE["text"], y=1.02)
save_fig("02_weather_impact_analysis")

# ─────────────────────────────────────────────
# 3. ML: SEVERITY PREDICTION (Random Forest)
# ─────────────────────────────────────────────
print("\n🤖 Chart 3: ML — Severity Classification")
le = LabelEncoder()
df_ml = df.copy()
cat_cols = ["emirate","incident_type","road_type","vehicle_type","weather_condition","time_slot"]
for col in cat_cols:
    df_ml[col+"_enc"] = le.fit_transform(df_ml[col])

features = [c+"_enc" for c in cat_cols] + ["hour","month","fatalities","injuries","response_time_min"]
X = df_ml[features]
y = df_ml["severity_score"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
acc = (y_pred == y_test).mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
apply_dark_theme(fig, axes)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
imp_colors = [PALETTE["accent2"] if v > 0.1 else PALETTE["accent1"] for v in importances.values]
axes[0].barh(importances.index, importances.values, color=imp_colors)
axes[0].set_title("Feature Importance (RF)", fontsize=11)
axes[0].set_xlabel("Importance", color=PALETTE["muted"])

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, ax=axes[1], annot=True, fmt="d", cmap="Blues",
            linewidths=0.3, linecolor=PALETTE["bg"],
            xticklabels=[f"Sev {i}" for i in sorted(y.unique())],
            yticklabels=[f"Sev {i}" for i in sorted(y.unique())])
axes[1].set_title(f"Confusion Matrix\nAccuracy: {acc:.2%} | CV: {cv_scores.mean():.2%}±{cv_scores.std():.2%}", fontsize=11)
axes[1].set_xlabel("Predicted", color=PALETTE["muted"])
axes[1].set_ylabel("Actual", color=PALETTE["muted"])
axes[1].set_facecolor(PALETTE["surface"])

# CV score distribution
axes[2].bar(range(1,6), cv_scores, color=PALETTE["accent3"], alpha=0.85, width=0.5)
axes[2].axhline(cv_scores.mean(), color=PALETTE["accent2"], linestyle="--", linewidth=2, label=f"Mean: {cv_scores.mean():.2%}")
axes[2].set_title("5-Fold Cross Validation Accuracy", fontsize=11)
axes[2].set_xlabel("Fold", color=PALETTE["muted"])
axes[2].set_ylabel("Accuracy", color=PALETTE["muted"])
axes[2].set_ylim(0.5, 1.0)
axes[2].legend(fontsize=9, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"], edgecolor=PALETTE["border"])
for i, v in enumerate(cv_scores):
    axes[2].text(i+1, v+0.01, f"{v:.2%}", ha="center", fontsize=8, color=PALETTE["muted"])

fig.suptitle("ML Model: Road Incident Severity Prediction (Random Forest)",
             fontsize=14, color=PALETTE["text"], y=1.02)
save_fig("03_ml_severity_prediction")

# ─────────────────────────────────────────────
# 4. SAFETY INDEX & EMIRATE COMPARISON
# ─────────────────────────────────────────────
print("\n📊 Chart 4: Emirate Safety Index Trends")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
apply_dark_theme(fig, axes.flatten())

# Safety index over time
ax = axes[0,0]
for emirate in EMIRATE_COLORS:
    data = si[si["emirate"] == emirate]
    ax.plot(data["year"], data["safety_index"], "o-", color=EMIRATE_COLORS[emirate],
            linewidth=2, markersize=5, label=emirate, alpha=0.9)
ax.set_title("Safety Index Trend by Emirate", fontsize=11)
ax.set_ylabel("Safety Index (0-100)", color=PALETTE["muted"])
ax.legend(fontsize=7, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"],
          edgecolor=PALETTE["border"], ncol=2)

# Incidents per 100k
ax = axes[0,1]
latest = si[si["year"] == si["year"].max()].sort_values("incidents_per_100k")
bars = ax.barh(latest["emirate"], latest["incidents_per_100k"],
               color=[EMIRATE_COLORS[e] for e in latest["emirate"]], height=0.6)
ax.set_title(f"Incidents per 100K Population ({si['year'].max()})", fontsize=11)
ax.set_xlabel("Rate", color=PALETTE["muted"])
for bar, val in zip(bars, latest["incidents_per_100k"]):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f"{val:.1f}",
            va="center", fontsize=8, color=PALETTE["muted"])

# Response time improvement
ax = axes[1,0]
for emirate in ["Dubai", "Abu Dhabi", "Sharjah", "Ajman"]:
    data = si[si["emirate"] == emirate]
    ax.plot(data["year"], data["avg_response_time_min"], "o-",
            color=EMIRATE_COLORS[emirate], linewidth=2, markersize=5, label=emirate)
ax.set_title("Emergency Response Time Trend (mins)", fontsize=11)
ax.set_ylabel("Avg Response (min)", color=PALETTE["muted"])
ax.legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"], edgecolor=PALETTE["border"])

# CCTV vs safety correlation
ax = axes[1,1]
latest2 = si[si["year"] == si["year"].max()].copy()
latest2["cctv_per_1000"] = latest2["cctv_cameras"] / latest2["population"] * 1000
scatter = ax.scatter(latest2["cctv_per_1000"], latest2["safety_index"],
                     c=[EMIRATE_COLORS[e] for e in latest2["emirate"]],
                     s=latest2["population"]/30000, alpha=0.8, edgecolors="white", linewidths=0.5)
for _, row in latest2.iterrows():
    ax.annotate(row["emirate"], (row["cctv_per_1000"], row["safety_index"]),
                textcoords="offset points", xytext=(5,4), fontsize=7, color=PALETTE["muted"])
m, b, r, p, se = stats.linregress(latest2["cctv_per_1000"], latest2["safety_index"])
x_line = np.linspace(latest2["cctv_per_1000"].min(), latest2["cctv_per_1000"].max(), 50)
ax.plot(x_line, m*x_line+b, "--", color=PALETTE["accent2"], alpha=0.7, linewidth=1.5, label=f"r={r:.2f}")
ax.set_title("CCTV Density vs Safety Index", fontsize=11)
ax.set_xlabel("CCTV per 1,000 people", color=PALETTE["muted"])
ax.set_ylabel("Safety Index", color=PALETTE["muted"])
ax.legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"], edgecolor=PALETTE["border"])

fig.suptitle("Emirate-Level Safety Performance & Infrastructure Analysis",
             fontsize=14, color=PALETTE["text"], y=1.01)
save_fig("04_emirate_safety_index")

# ─────────────────────────────────────────────
# 5. TEMPORAL PATTERN ANALYSIS
# ─────────────────────────────────────────────
print("\n📊 Chart 5: Temporal Patterns")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
apply_dark_theme(fig, axes.flatten())

# Hourly distribution by severity
ax = axes[0,0]
for sev, color in zip([1,3,5], [PALETTE["accent3"], PALETTE["accent5"], PALETTE["accent2"]]):
    hourly = df[df["severity_score"]==sev].groupby("hour").size()
    ax.fill_between(hourly.index, hourly.values, alpha=0.4, color=color, label=f"Sev {sev}")
    ax.plot(hourly.index, hourly.values, color=color, linewidth=1.5)
ax.set_title("Hourly Incident Distribution by Severity", fontsize=11)
ax.set_xlabel("Hour of Day", color=PALETTE["muted"])
ax.set_ylabel("Incident Count", color=PALETTE["muted"])
ax.legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"], edgecolor=PALETTE["border"])
ax.set_xticks(range(0,24,2))

# Monthly fatality rate
ax = axes[0,1]
monthly_fat = df.groupby("month").agg(
    total=("incident_id","count"), fatalities=("fatalities","sum")
).reset_index()
monthly_fat["rate"] = monthly_fat["fatalities"] / monthly_fat["total"] * 100
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax.bar(month_names, monthly_fat["rate"],
       color=[PALETTE["accent2"] if m in [1,2,3,12] else PALETTE["accent1"] for m in monthly_fat["month"]],
       width=0.7, alpha=0.85)
ax.axhline(monthly_fat["rate"].mean(), color=PALETTE["accent3"], linestyle="--",
           linewidth=1.5, label=f"Mean: {monthly_fat['rate'].mean():.2f}%")
ax.set_title("Monthly Fatality Rate (%)", fontsize=11)
ax.set_ylabel("Fatality Rate (%)", color=PALETTE["muted"])
ax.legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"], edgecolor=PALETTE["border"])

# YoY comparison
ax = axes[1,0]
yearly_monthly = df.groupby(["year","month"])["incident_id"].count().unstack()
year_colors = [PALETTE["accent4"], PALETTE["accent1"], PALETTE["accent3"],
               PALETTE["accent5"], PALETTE["accent2"]]
for i, year in enumerate(yearly_monthly.index):
    ax.plot(range(1,13), yearly_monthly.loc[year], "o-",
            color=year_colors[i % len(year_colors)], linewidth=2, markersize=5,
            label=str(year), alpha=0.85)
ax.set_xticks(range(1,13))
ax.set_xticklabels(month_names)
ax.set_title("Year-over-Year Monthly Incidents", fontsize=11)
ax.set_ylabel("Incident Count", color=PALETTE["muted"])
ax.legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"],
          edgecolor=PALETTE["border"], ncol=3)

# Vehicle type severity boxplot
ax = axes[1,1]
vehicle_types = df["vehicle_type"].unique()
data_by_vehicle = [df[df["vehicle_type"]==v]["severity_score"].values for v in vehicle_types]
bp = ax.boxplot(data_by_vehicle, labels=vehicle_types, patch_artist=True,
                medianprops=dict(color=PALETTE["accent2"], linewidth=2),
                whiskerprops=dict(color=PALETTE["border"]),
                capprops=dict(color=PALETTE["border"]),
                flierprops=dict(marker=".", color=PALETTE["muted"], markersize=3, alpha=0.4))
box_colors = [PALETTE["accent1"], PALETTE["accent3"], PALETTE["accent5"],
              PALETTE["accent4"], PALETTE["accent2"], PALETTE["muted"]]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Severity Distribution by Vehicle Type", fontsize=11)
ax.set_ylabel("Severity Score", color=PALETTE["muted"])
ax.tick_params(axis='x', rotation=20)

fig.suptitle("Temporal & Behavioral Patterns in Road Incidents",
             fontsize=14, color=PALETTE["text"], y=1.01)
save_fig("05_temporal_patterns")

# ─────────────────────────────────────────────
# 6. CLUSTERING — HIGH-RISK PROFILES
# ─────────────────────────────────────────────
print("\n🤖 Chart 6: K-Means Risk Profiling")
cluster_features = ["severity_score","fatalities","injuries","response_time_min","hour","month","fine_aed"]
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["risk_cluster"] = kmeans.fit_predict(X_cluster)

cluster_labels = {
    df.groupby("risk_cluster")["severity_score"].mean().idxmax(): "🔴 High Risk",
    df.groupby("risk_cluster")["severity_score"].mean().idxmin(): "🟢 Low Risk",
}
cluster_means = df.groupby("risk_cluster")["severity_score"].mean().sort_values()
mid_clusters = [c for c in range(4) if c not in cluster_labels]
cluster_labels[mid_clusters[0]] = "🟡 Moderate Risk"
cluster_labels[mid_clusters[1]] = "🟠 Elevated Risk"
df["risk_label"] = df["risk_cluster"].map(cluster_labels)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
apply_dark_theme(fig, axes.flatten())

cluster_colors_map = {
    "🔴 High Risk": PALETTE["accent2"],
    "🟢 Low Risk": PALETTE["accent3"],
    "🟡 Moderate Risk": PALETTE["accent5"],
    "🟠 Elevated Risk": "#FFA657"
}

# Cluster scatter (PCA-style: hour vs severity)
ax = axes[0,0]
for label, color in cluster_colors_map.items():
    mask = df["risk_label"] == label
    ax.scatter(df.loc[mask, "hour"], df.loc[mask, "severity_score"],
               c=color, alpha=0.3, s=12, label=label)
ax.set_title("Risk Clusters: Hour vs Severity", fontsize=11)
ax.set_xlabel("Hour of Day", color=PALETTE["muted"])
ax.set_ylabel("Severity Score", color=PALETTE["muted"])
ax.legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"],
          edgecolor=PALETTE["border"], markerscale=2)

# Cluster profiles
ax = axes[0,1]
profile = df.groupby("risk_label")[["severity_score","fatalities","injuries"]].mean()
x_pos = np.arange(len(profile))
width = 0.25
ax.bar(x_pos - width, profile["severity_score"], width=width, label="Avg Severity",
       color=PALETTE["accent1"], alpha=0.85)
ax.bar(x_pos, profile["fatalities"]*2, width=width, label="Fatalities ×2",
       color=PALETTE["accent2"], alpha=0.85)
ax.bar(x_pos + width, profile["injuries"]/2, width=width, label="Injuries ÷2",
       color=PALETTE["accent3"], alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(profile.index, rotation=15, fontsize=8)
ax.set_title("Risk Cluster Profiles", fontsize=11)
ax.legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"],
          edgecolor=PALETTE["border"])

# Cluster by emirate
ax = axes[1,0]
cluster_emirate = df.groupby(["emirate","risk_label"]).size().unstack(fill_value=0)
cluster_emirate.plot(kind="bar", ax=ax, stacked=True,
                     color=[cluster_colors_map.get(c, PALETTE["accent1"]) for c in cluster_emirate.columns],
                     alpha=0.85, width=0.7)
ax.set_title("Risk Cluster Distribution by Emirate", fontsize=11)
ax.set_ylabel("Incident Count", color=PALETTE["muted"])
ax.tick_params(axis='x', rotation=25, labelsize=8)
ax.legend(fontsize=7, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"],
          edgecolor=PALETTE["border"], ncol=2)

# Cluster by time slot
ax = axes[1,1]
cluster_time = df.groupby(["time_slot","risk_label"]).size().unstack(fill_value=0)
cluster_time.plot(kind="barh", ax=ax, stacked=True,
                  color=[cluster_colors_map.get(c, PALETTE["accent1"]) for c in cluster_time.columns],
                  alpha=0.85)
ax.set_title("Risk Clusters by Time of Day", fontsize=11)
ax.set_xlabel("Incident Count", color=PALETTE["muted"])
ax.legend(fontsize=7, facecolor=PALETTE["surface"], labelcolor=PALETTE["muted"],
          edgecolor=PALETTE["border"])

fig.suptitle("K-Means Risk Profiling: Identifying High-Risk Incident Patterns",
             fontsize=14, color=PALETTE["text"], y=1.01)
save_fig("06_risk_clustering")

# ─────────────────────────────────────────────
# 7. PRINT SUMMARY STATS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  UAE ROAD SAFETY ANALYTICS — KEY FINDINGS")
print("="*55)
print(f"  Period Analyzed     : 2019–2024")
print(f"  Total Incidents     : {len(df):,}")
print(f"  Total Fatalities    : {df['fatalities'].sum():,}")
print(f"  Avg Severity Score  : {df['severity_score'].mean():.2f}/5")
print(f"  Most Dangerous Time : {df.groupby('time_slot')['severity_score'].mean().idxmax()}")
print(f"  Most Dangerous Wx   : {df.groupby('weather_condition')['severity_score'].mean().idxmax()}")
print(f"  RF Model Accuracy   : {acc:.2%} (CV: {cv_scores.mean():.2%})")
print(f"  K-Means Clusters    : 4 risk profiles identified")
print(f"\n  Reports saved to: /home/claude/uae-safety-analytics/reports/")
print("="*55)
