"""Shared terrain runtime configuration constants."""

# Canonical EPSG:3413 root square used by every terrain tile address. The raw
# Greenland bounds are approximately x=[-627247, 849164],
# y=[-3326128, -666027]; this is the 2,700 km square padded around their
# midpoint for exact quadtree subdivision.
GREENLAND_BBOX = (
  -1239041.5,
  -3346077.5,
  1460958.5,
  -646077.5,
)

# Maximum terrain tile depth and subdivision ceiling.
# Depth 13 ≈ 330 m tiles, halving each level to depth 16 ≈ 41 m — walking
# scale. Past WMS_CONTRACT_DEPTH the provider imagery may be a plain blowup
# of the level above; fetched metatiles are inspected and fall back to the
# deterministic procedural upscaler when they carry no new detail (source
# "cooked_upscale"). Past WMS_TEXTURE_PROBE_MAX_DEPTH the fetch is not even
# attempted and cooks recurse on cooked parents. Render cost of the deep
# levels is bounded by the per-depth LOD cores in serve.py (~3 tile widths
# per level), not by this ceiling.
MAX_TILE_DEPTH = 16

# Deepest level where provider imagery is trusted at face value.
# Depth 12 ≈ 659 m tiles — Sentinel-2 z14 (~2.4 m/px) and ArcticDEM 10 m
# still have headroom there.
WMS_CONTRACT_DEPTH = 12

# Dataforsyningen is only trusted without a detail check through depth 10.
# Starting at depth 11, score every fetched metatile for evidence that the
# provider merely enlarged a coarser image and carved it into finer addresses.
# This is deliberately independent from the depth-12 DEM contract.
WMS_TEXTURE_INSPECT_MIN_DEPTH = 11

# Deepest level worth asking the WMS for at all. Depth 13 metatiles are
# fetched and blowup-inspected (measured: occasionally genuine); deeper
# requests always came back as upsamples, so past this depth the texture
# worker skips the provider round-trip and cooks directly.
WMS_TEXTURE_PROBE_MAX_DEPTH = 13

# Initial skeleton depth used on a brand-new DB at server startup.
# Keep this shallow so Flask becomes responsive quickly; deeper levels
# are created lazily by traversal as real data arrives.
BOOTSTRAP_SEED_DEPTH = 8

# Asiaq Teknisk Grundkort settlements the server keeps downloaded and
# ingested (folder names from kortforsyning.asiaq.gl/files/). The default
# camera starts over Nuuk, so a fresh clone gets its buildings/roads with
# no manual steps. Add folders here to cover more settlements.
GRUNDKORT_SETTLEMENTS = ["0600NUK_Nuuk"]

# Åbent Land GL50 vector blocks (100 km GR96/UTM-24N ids) fetched at first
# start so a fresh clone has exact fjord coastline around the default Nuuk
# camera immediately, rather than flying over WMS-derived shoreline while the
# first blocks download.
#
# This is a seed, NOT the set of blocks the server can use: anything else is
# acquired on demand by gtk50_demand as tiles ask for it, so there is no need
# to add ids here when flying somewhere new. Downloading needs the
# Dataforsyningen account login in .env; without it the server logs a loud
# startup error and coastline masks fall back to the rendered-WMS decoder.
GTK50_BLOCKS = ["71_-1", "71_-2"]
