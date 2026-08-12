# Terrain dynamic functions

ArcticDEM supplies WGS84 ellipsoidal elevations. The Terrain tools convert
them to orthometric EGM2008 elevations before persistence so that sea level is
approximately zero metres.

The conversion requires PROJ's open `us_nga_egm08_25.tif` grid. Install it in
the Python environment used to run Atlantis:

```bash
projsync --file us_nga_egm08_25.tif
```

Acquisition fails closed when the real grid-backed transformation is not
available. PROJ's ballpark zero-offset fallback must never be used because it
would store sea-level terrain roughly 28–49 metres too high around Greenland.
