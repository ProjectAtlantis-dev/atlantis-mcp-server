"""Offline regression gate for curing beneath fine camera demand."""

from unittest.mock import patch

from dynamic_functions.Terrain import demand


@visible
def camera_cure_offline() -> dict:
    near = "13-3688-588"
    sibling = "12-1845-294"
    ring = "11-930-147"
    parent = "11-922-147"
    targets = [near, sibling, ring, "8-115-18"]
    selection = {"tileIds": targets}
    ready = {"dem": set(targets), "textures": set(targets)}

    def present(_connection, table, ids):
        return set(ids) & ready.get(table, set())

    class Coordinator:
        def refresh(self, candidates):
            return {}

        def submit(self, candidates, **kwargs):
            return {}

        def status(self):
            return {}

    def candidates(origin="viewer"):
        return demand.submit_camera_demand_from_selection(
            None, selection, Coordinator(), demand_origin=origin
        )["candidates"]

    with patch.object(demand, "_present_ids", side_effect=present), patch.object(
        demand, "eligible_fjord_jobs", return_value=[]
    ):
        initial = candidates()
        ready["dem"].add(parent)
        staged = candidates()
        repeated = candidates()
        auxiliary = candidates("bathymetry")
        ready["coastline_masks"] = {parent}
        ready["textures"].add(parent)
        complete = candidates()

    return {
        "fineParentsDeduplicatedInCameraOrder": demand.camera_cure_ids(targets)
        == [parent, ring],
        "missingParentDemScheduled": initial["dem"] == [parent],
        "coastWaitsForParentDem": parent not in initial["coastline"],
        "underCameraCoastBeforeRing": staged["coastline"][0] == parent
        and ring in staged["coastline"],
        "fineRenderMasksRetained": "12-1844-294" in staged["coastline"]
        and sibling in staged["coastline"],
        "exactParentTextureScheduled": "11-920-144" in staged["texture"],
        "stationaryCameraRetainsCureDemand": repeated == staged,
        "readyEvidenceNotReacquired": parent not in complete["dem"]
        and parent not in complete["coastline"] and complete["texture"] == [],
        "bathymetryDoesNotAcquireCureParents": parent not in auxiliary["coastline"]
        and auxiliary["dem"] == [],
        "coarseDemandDoesNotExpand": demand.camera_cure_ids(["8-115-18"]) == [],
    }
