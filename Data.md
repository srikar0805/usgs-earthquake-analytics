Dataset  
USGS GeoJSON Detail for individual earthquake events (GeoJSON Feature with extended products information).

Dimensionality

•  Spatial: Point geometry with longitude, latitude, and depth (3D spatial coordinates).
•  Temporal: time and updated fields as epoch timestamps (time series when aggregating events).
•  Scalar attributes: mag, felt, cdi, mmi, rms, gap, dmin, nst, sig, tsunami — numeric or integer measures per event.
•  Categorical attributes: place, status, alert, net, code, magType, type, ids, sources, types — descriptive labels and event identifiers.
•  Multidimensional nested products: products is a nested, variable-length dictionary of contributor reports where each product has its own metadata and content entries (e.g., shakemap, focal-mechanism, moment-tensor).
•  Document links and metadata: URLs, content types, sizes, and update times inside products.contents provide additional non-numeric fields for enrichment.

Granularity and Volume

•  Single-event granularity in each GeoJSON Detail file; aggregating many details yields event-level time series and spatial point clouds.
•  Typical volume: small per-file but scales to large datasets when collecting global feeds over time.

Quality and Missingness

•  Numeric fields like mag, depth, and time are usually present; some product fields may be absent for certain events.
•  Nested products are inconsistent across events, requiring defensive parsing.

Data Types Summary

•  Spatial coordinates: Float.
•  Time fields: Integer (epoch ms).
•  Measurements: Float/Integer.
•  Textual metadata and IDs: String.
•  Nested product objects: JSON objects mapping to lists of structured records.