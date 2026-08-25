"""One callable runner for every offline Terrain migration regression gate."""

from __future__ import annotations

import time

from dynamic_functions.Terrain.Test.arctic_dem_decode import arcticdem_decode
from dynamic_functions.Terrain.Test.arcticdem_failures import arcticdem_failures
from dynamic_functions.Terrain.Test.binary_batch import binary_batch_offline
from dynamic_functions.Terrain.Test.camera_lod import camera_lod_offline
from dynamic_functions.Terrain.Test.coastline import coastline_offline
from dynamic_functions.Terrain.Test.composition import composition_offline
from dynamic_functions.Terrain.Test.dataforsyningen_decode import (
    dataforsyningen_decode,
)
from dynamic_functions.Terrain.Test.dataforsyningen_failures import (
    dataforsyningen_failures,
)
from dynamic_functions.Terrain.Test.demand import demand_lanes_offline
from dynamic_functions.Terrain.Test.demand_priority import (
    demand_priority_offline,
)
from dynamic_functions.Terrain.Test.demand_retry import demand_retry_offline
from dynamic_functions.Terrain.Test.dem_persistence import dem_persistence
from dynamic_functions.Terrain.Test.effective_heightmap import (
    effective_heightmap_offline,
)
from dynamic_functions.Terrain.Test.hydrography import hydrography_offline
from dynamic_functions.Terrain.Test.http_adapter import http_adapter_offline
from dynamic_functions.Terrain.Test.parent_fallback import parent_fallback
from dynamic_functions.Terrain.Test.polling_convergence import (
    polling_convergence_offline,
)
from dynamic_functions.Terrain.Test.texture_persistence import (
    texture_persistence,
)
from dynamic_functions.Terrain.Test.tidal_connectivity import (
    tidal_connectivity_offline,
)
from dynamic_functions.Terrain.Test.viewer_server import viewer_server_offline


_DEM_TILE = "10-334-192"
_TEXTURE_TILE = "10-328-212"
# Keep persistence away from the live texture gate's known tile and from the
# atomic-failure fixture hardcoded in texture_persistence.
_TEXTURE_PERSISTENCE_TILE = "10-336-212"
_ARCTIC_FIXTURE_DIGEST = (
    "a281369d5aca9740d5a7805c39c168e0904475556aaa340c2064da050becc902"
)


def _require(checks: dict[str, bool], test_name: str) -> None:
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{test_name} failed: " + ", ".join(failed))


@visible
def terrain_regression() -> dict:
    """Run all deterministic offline Terrain gates and fail on any regression."""

    tests = []

    def run(name: str, function, checks) -> None:
        started = time.perf_counter()
        result = function()
        _require(checks(result), name)
        tests.append(
            {
                "name": name,
                "passed": True,
                "durationMs": round((time.perf_counter() - started) * 1000, 2),
            }
        )

    run(
        "arcticdem_decode",
        lambda: arcticdem_decode(_DEM_TILE),
        lambda result: {
            "shape": result["shape"] == [65, 65],
            "dtype": result["dtype"] == "float32",
            "verticalDatum": result["verticalDatum"] == "EGM2008",
            "digest": result["digest"] == _ARCTIC_FIXTURE_DIGEST,
        },
    )
    run(
        "arcticdem_failures",
        lambda: arcticdem_failures(_DEM_TILE),
        lambda result: result["checks"],
    )
    run(
        "dem_persistence",
        lambda: dem_persistence(_DEM_TILE),
        lambda result: {
            "firstWrite": result["firstWrite"],
            "duplicateWrite": result["duplicateWrite"] is False,
            "datumClobberBlocked": result["datumClobberBlocked"],
            "exactRoundTrip": result["exactRoundTrip"],
            "clobberBlocked": result["clobberBlocked"],
            "existingPreserved": result["existingPreserved"],
            "failedAcquisitionRejected": result["failedAcquisitionRejected"],
            "failedAcquisitionPreserved": result["failedAcquisitionPreserved"],
            "digest": result["digest"] == _ARCTIC_FIXTURE_DIGEST,
        },
    )
    run(
        "dataforsyningen_decode",
        lambda: dataforsyningen_decode(_TEXTURE_TILE),
        lambda result: {
            "fixture": result["fixtureStatus"] == "valid",
            "children": result["childCount"] == 16,
            "white": result["whiteStatus"] == "white_fill",
            "corrupt": result["corruptStatus"] == "corrupt",
        },
    )
    run(
        "dataforsyningen_failures",
        lambda: dataforsyningen_failures(_TEXTURE_TILE),
        lambda result: result["checks"],
    )
    run(
        "texture_persistence",
        lambda: texture_persistence(_TEXTURE_PERSISTENCE_TILE),
        lambda result: {
            key: bool(result[key])
            for key in (
                "firstWrite",
                "exactRoundTrip",
                "clobberBlocked",
                "existingPreserved",
                "partialRejected",
                "failedAcquisitionRejected",
                "failedAcquisitionPreserved",
                "atomicFailureRaised",
                "atomicFailureLeftNoSiblings",
            )
        }
        | {"duplicateWrite": result["duplicateWrite"] is False},
    )
    run(
        "parent_fallback",
        parent_fallback,
        lambda result: {
            key: bool(result[key])
            for key in (
                "demNearestAncestor",
                "demExactPrecedence",
                "demMiss",
                "textureNearestAncestor",
                "textureExactPrecedence",
                "textureMiss",
                "readsChangedNoRows",
            )
        },
    )
    run(
        "coastline_offline",
        lambda: coastline_offline(_DEM_TILE),
        lambda result: {
            "orientation": result["southFirstOrientation"],
            "firstWrite": result["firstWrite"],
            "duplicateWrite": result["duplicateWrite"] is False,
            "exactRoundTrip": result["exactRoundTrip"],
            "clobberBlocked": result["clobberBlocked"],
            "existingPreserved": result["existingPreserved"],
            "readOnlyReads": result["readOnlyReads"],
        },
    )
    run(
        "hydrography_offline",
        lambda: hydrography_offline(_DEM_TILE),
        lambda result: {
            key: bool(result[key])
            for key in (
                "southFirstOrientation",
                "requestReadOnly",
                "fetchCalledOnce",
                "acquisitionExact",
                "corruptRejected",
                "wrongSizeRejected",
                "firstWrite",
                "exactRoundTrip",
                "clobberBlocked",
                "existingPreserved",
                "readOnlyRead",
                "providerFailureRaised",
                "providerFailurePreserved",
            )
        }
        | {"duplicateWrite": result["duplicateWrite"] is False},
    )
    run(
        "tidal_connectivity_offline",
        tidal_connectivity_offline,
        lambda result: {
            key: bool(result[key])
            for key in (
                "sameTileCoastSeeds",
                "sharedEdgePropagates",
                "neighborCoastSeeds",
                "wholeLowTileSeedsAllComponents",
                "oneLowSampleDoesNotSeedTile",
                "disconnectedInlandRejected",
                "readOnlyDerivation",
            )
        },
    )
    run(
        "effective_heightmap_offline",
        effective_heightmap_offline,
        lambda result: {
            key: bool(result[key])
            for key in (
                "coastAndConnectedHydroUnion",
                "disconnectedInlandRejected",
                "waterFloorApplied",
                "staleWaterOnLandClipped",
                "measuredLandPreserved",
                "canonicalDemPreserved",
                "readOnlyDerivation",
                "responseMatchesArray",
                "noMaskUsesDemFallback",
                "allWaterWithoutDemSynthesized",
                "mixedWaterWithoutDemRejected",
                "shapeMismatchRejected",
            )
        },
    )
    run(
        "composition_offline",
        composition_offline,
        lambda result: {
            key: bool(result[key])
            for key in (
                "inputOrderPreserved",
                "demNearestAncestor",
                "textureNearestAncestor",
                "textureSurvivesMissingDem",
                "demSurvivesMissingTexture",
                "missingConnectivityIsPending",
                "readyConnectivityComposed",
                "domainErrorIsolated",
                "explicitMiss",
                "readOnlyComposition",
                "noNetworkOrScheduling",
            )
        }
        | {"tileCount": result["tileCount"] == 7},
    )
    run(
        "binary_batch_offline",
        binary_batch_offline,
        lambda result: {
            "format": result["format"] == "binary-v1",
            "aligned": result["aligned"],
            "browserFields": result["browserFields"],
            "knownDigestReused": result["knownDigestReused"],
            "unknownDigestTransferred": result["unknownDigestTransferred"],
            "missingCarriesNoBlock": result["missingCarriesNoBlock"],
            "corruptDomainIsolated": result["corruptDomainIsolated"],
            "embeddedBase64Removed": result["embeddedBase64Removed"],
            "compositionUnchanged": result["compositionUnchanged"],
            "exactOracleParity": result["exactOracleParity"],
            "noTrailingBytes": result["noTrailingBytes"],
            "invalidKnownRejected": result["invalidKnownRejected"],
            "stableDigest": result["stableDigest"],
        },
    )
    run(
        "camera_lod_offline",
        camera_lod_offline,
        lambda result: {
            key: bool(result[key])
            for key in (
                "radialBoundaryParity",
                "pastContractParity",
                "altitudeParity",
                "hysteresisHeld",
                "stableSelection",
                "twoToOneBalanced",
                "pureSelection",
                "coherentFallbackAntichain",
                "fallbackReported",
                "trueMissReported",
                "cameraGeometry",
                "missingViewerFields",
                "browserWireFields",
                "readOnly",
                "noNetworkOrScheduling",
                "invalidInputRejected",
                "stableBinary",
            )
        },
    )
    run(
        "demand_lanes_offline",
        demand_lanes_offline,
        lambda result: {
            key: bool(result[key])
            for key in (
                "immediateReturn",
                "slowStarted",
                "independentCapacity",
                "boundedActive",
                "deduplicated",
                "allWorkCompleted",
                "failureIsolated",
                "failedNotHotLooped",
                "unknownLaneRejected",
                "dependencyStaged",
                "candidateReadsAreReadOnly",
            )
        }
        | {"didNotWait": result["waitedForWorkers"] is False},
    )
    run(
        "demand_priority_offline",
        demand_priority_offline,
        lambda result: result,
    )
    run(
        "demand_retry_offline",
        demand_retry_offline,
        lambda result: {
            key: bool(result[key])
            for key in (
                "firstDeadline",
                "noEarlyRetry",
                "laterPassEligible",
                "boundedEventuallySucceeds",
                "refreshDoesNotSleep",
                "terminalNotRetried",
                "transientExhausted",
                "classifierBoundaries",
            )
        },
    )
    run(
        "polling_convergence_offline",
        polling_convergence_offline,
        lambda result: result,
    )
    run("http_adapter_offline", http_adapter_offline, lambda result: result)
    run("viewer_server_offline", viewer_server_offline, lambda result: result)
    return {
        "passed": True,
        "testCount": len(tests),
        "tests": tests,
        "offline": True,
    }
