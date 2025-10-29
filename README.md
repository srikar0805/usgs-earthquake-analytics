# usgs-geojson-earthquakes

data link: https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.geojson

## Project Goal
Visualize earthquake events using USGS GeoJSON Detail records to reveal spatial patterns, temporal trends, event severity distribution, and relationships between event metadata and secondary products such as shakemaps and focal mechanisms.

## Visualization Strategy

### Spatial Visualizations
- **Interactive map with 3D depth encoding**
  - Plot points at (longitude, latitude) with color by magnitude and vertical displacement or point size encoding depth or magnitude.
  - Enable tooltip showing place, time, mag, depth, alert, and a link to products.contents files.
  - Use cluster/heatmap overlays for dense regions to reveal hotspots.
- **Choropleth or regional aggregation**
  - Aggregate counts, average magnitude, and maximum intensity per administrative region or hex bins and display choropleth or hexbin maps for comparative spatial summaries.

### Temporal Visualizations
- **Time series dashboard**
  - Rolling counts and average magnitude per day/week with zoomable time axis.
  - Anomaly detection layer highlighting bursts, foreshock/aftershock sequences using sequence markers derived from ids and event times.
- **Small multiples by region or magnitude class**
  - Faceted line charts showing temporal trends for different geographic regions or magnitude bins to compare temporal behavior.

### Multidimensional and Nested Data Visualizations
- **Scatterplot matrix and parallel coordinates**
  - Compare mag, depth, rms, dmin, gap, and nst to find correlations and outliers.
  - Color/shape by magType or alert.
- **Product presence matrix**
  - Binary heatmap showing which product types (shakemap, moment-tensor, finite-fault) exist per event, enabling analysis of which events generate richer secondary data.
- **Drill-down panels**
  - From a selected event show nested products list with links to downloadable files, and render shakemap images or moment-tensor summaries inline when available.

### Graph and Sequence Visualizations
- **Aftershock network and linkage graph**
  - Create directed edges from mainshock to aftershocks using temporal and spatial proximity heuristics; visualize as a node-link graph with node size by magnitude and edge thickness by temporal closeness.
- **Event family clustering**
  - Cluster events by waveform-derived attributes when available in products and visualize cluster membership on map and timeline.

### Interaction and UX Considerations
- Provide filtering by date range, magnitude range, depth range, region, and product availability.
- Tooltips linking to original USGS detail URL and product content for validation.
- Lazy-loading of product content to avoid heavy initial payloads.
- Export options for filtered subsets (CSV/GeoJSON).

### Implementation Notes
- Parse GeoJSON safely: handle missing product keys and variable list lengths.
- Convert epoch ms to ISO 8601 for display.
- Precompute regional aggregations and time-window summaries for responsive dashboards.
- Recommended libraries: Leaflet or Mapbox GL for maps, D3 or Vega-Lite for charts, and a lightweight web framework (Flask or Node) to serve product previews securely.

## Expected Insights
- Spatial clusters of high-magnitude events and depth patterns that suggest tectonic features.
- Temporal bursts corresponding to seismic sequences.
- Relationship between magnitude/depth and product richness indicating which events trigger detailed analysis products.

---

## Current Visualizations
This project currently implements three core visualizations:
1. **Heatmap with Temporal Animation** – Animated world map showing earthquakes over time, with color by magnitude and size by depth.
2. **Magnitude vs Depth Scatter Plot** – Scatter plot comparing earthquake magnitude against depth, highlighting shallow vs deep event patterns.
3. **Earthquake Frequency Over Time** – Line chart showing daily counts of earthquakes, revealing bursts of seismic activity.

These outputs are saved in the `visualizations/` folder as `.png` images, with the temporal heatmap also exported as both `.html` (interactive) and `.mp4` (video).

---

## Broader Impact
In addition to the dataset description, this project emphasizes the importance of multidimensional visualization. By combining spatial, temporal, and scalar perspectives, the analysis provides a holistic view of earthquake activity. The interactive heatmap highlights where and when seismic events occur, while the scatter plot and frequency chart reveal deeper insights into the relationships between magnitude, depth, and time.

These visualizations are designed not only to meet academic requirements but also to serve as a foundation for real-world applications. The same techniques could be extended to build dashboards for disaster preparedness, population risk modeling, or infrastructure planning. By making complex seismic data more accessible and interpretable, this project demonstrates the power of visualization in bridging raw data and actionable knowledge.

---

## How to Run

1. Clone this repository and navigate into the project folder.
2. Ensure you have Python 3.10+ installed.
3. Install dependencies:
   ```bash
   pip install pandas plotly kaleido moviepy

Place your USGS GeoJSON file in the data/ directory as GeoQuake.json.

## Repository Structure

├── data/
│   └── GeoQuake.json
├── src/
│   └── visualize_earthquakes.py
|   |__ extension.ipynb
├── visualizations/
│   ├── heatmap_temporal.html
│   ├── heatmap_temporal.mp4
│   ├── scatter_depth_magnitude.png
│   ├── frequency_over_time.png
│   └── captions.md
├── README.md
└── DATA.md# usgs-earthquake-predictive
