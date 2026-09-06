# Southern Greenland white ocean wedge

Original cached JPEG bytes for tile `11-934-28`, source `dataforsyningen`,
updated `2026-09-04T16:44:27.695650+00:00`. Extracted read-only from the local
Terrain database for the user-reported diagonal white ocean patch.

SHA-256: `4c288129d624a5a503ae84358b6b65ad150a023a469e3f35754f7df788ff090c`.

The corresponding `gtk50_vector` version 2 coastline mask is 65x65 with all
4,225 vertices classified as ocean. The fixture test reconstructs that exact
boolean mask. JPEG pixels are north-first; coastline vertices are south-first.

This is a suspected imagery gap based on appearance and location, not a
provider-confirmed NoData mask. The white wedge cuts diagonally across the
256x256 texture. No tile-boundary assumption should be made.

Contains data from Klimadatastyrelsen, supplied by Dataforsyningen.
Provider terms: https://dataforsyningen.dk/vilkaar.

The companion `ocean_white_block_9-230-6.jpg` is an original cached texture
from the wider-view example, also entirely ocean in its 65x65 GTK50 version 2
mask. SHA-256:
`28de5bb6fb1f99c8da1597f67bfb83a71fad080c4872e5336562a6cd5eb0a202`.
It tests a large offshore blank area at a coarser LOD with the same criteria.

`ocean_ice_12-1858-52.jpg` is the user's real ice comparison case: smaller
objects with irregular outlines and varying brightness. Its GTK50 version 2
mask also classifies all 4,225 vertices as ocean, so the colour/size criteria
must protect it without relying on a land exclusion. SHA-256:
`2d353423ba508bb96ba9944f6fc255cded69bbe84ca6eff1edf90ff25911d05c`.
