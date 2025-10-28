import json
import pandas as pd
import plotly.express as px
import os
import moviepy
from moviepy import VideoFileClip
import imageio.v2 as imageio

# Load GeoJSON file
with open("data/GeoQuake.json") as f:
    data = json.load(f)

# Parse features
records = []
for feature in data["features"]:
    props = feature["properties"]
    coords = feature["geometry"]["coordinates"]
    try:
        records.append({
            "time": pd.to_datetime(props["time"], unit="ms"),
            "magnitude": props.get("mag", None),
            "depth": coords[2],
            "longitude": coords[0],
            "latitude": coords[1],
            "place": props.get("place", "Unknown")
        })
    except Exception:
        continue

# Create DataFrame
df = pd.DataFrame(records)
df = df.dropna(subset=["latitude", "longitude", "magnitude", "depth"])
df = df[df["depth"] >= 0]
df["date"] = df["time"].dt.date.astype(str)
df["depth_scaled"] = df["depth"] / df["depth"].max() * 10

# ----------------------------------------
# 1. Heatmap with Time Slider (HTML)
# ----------------------------------------
fig_map = px.scatter_geo(
    df,
    lat="latitude",
    lon="longitude",
    color="magnitude",
    size="depth_scaled",
    hover_name="place",
    animation_frame="date",
    projection="natural earth",
    title="Global Earthquake Events Over Time",
    color_continuous_scale="Turbo",
    size_max=10
)
fig_map.write_html("visualizations/heatmap_temporal.html")

# ----------------------------------------
# 1A. Save frames for video
# ----------------------------------------
frame_dir = "visualizations/frames"
os.makedirs(frame_dir, exist_ok=True)

for date in sorted(df["date"].unique()):
    df_day = df[df["date"] == date]
    fig_frame = px.scatter_geo(
        df_day,
        lat="latitude",
        lon="longitude",
        color="magnitude",
        size="depth_scaled",
        hover_name="place",
        projection="natural earth",
        title=f"Earthquakes on {date}",
        color_continuous_scale="Turbo",
        size_max=10
    )
    fig_frame.write_image(f"{frame_dir}/frame_{date}.png")

# ----------------------------------------
# 1B. Convert frames to video
# ----------------------------------------
    
images = [imageio.imread(f"visualizations/frames/{img}") for img in sorted(os.listdir("visualizations/frames")) if img.endswith(".png")]
imageio.mimsave("visualizations/heatmap_temporal.mp4", images, fps=2)

# ----------------------------------------
# 2. Magnitude vs Depth Scatter Plot
# ----------------------------------------
fig_scatter = px.scatter(
    df,
    x="magnitude",
    y="depth",
    color="magnitude",
    hover_name="place",
    title="Magnitude vs Depth of Earthquakes",
    labels={"magnitude": "Magnitude", "depth": "Depth (km)"},
    color_continuous_scale="Viridis"
)
fig_scatter.update_yaxes(autorange="reversed")
fig_scatter.write_image("visualizations/scatter_depth_magnitude.png")

# ----------------------------------------
# 3. Earthquake Frequency Over Time
# ----------------------------------------
df_freq = df.groupby("date").size().reset_index(name="count")
fig_line = px.line(
    df_freq,
    x="date",
    y="count",
    title="Earthquake Frequency Over Time",
    labels={"date": "Date", "count": "Number of Earthquakes"}
)
fig_line.write_image("visualizations/frequency_over_time.png")