"""Coordinate-driven asset acquisition regression gate; no provider access."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import threading
from unittest.mock import patch
import zipfile

import numpy as np
from pyproj import Transformer
from starlette.requests import Request

from dynamic_functions.Terrain.Asset import acquisition as acq
from dynamic_functions.Terrain.Asset import rebuild, schema
from dynamic_functions.Terrain.demand import DemandLane
from dynamic_functions.Terrain import viewer_assets, viewer_server


def _fixture_archive(path, lat=60.15031, lon=-44.28503):
    transform = Transformer.from_crs(4326, 3183, always_xy=True)
    x, y = transform.transform(lon, lat)
    xy = [(x,y),(x+10,y),(x+10,y+10),(x,y+10),(x,y)]
    content = (struct.pack('<i4d2ii', 15, x,y,x+10,y+10,1,5,0)
               + b''.join(struct.pack('<2d', *point) for point in xy)
               + struct.pack('<2d5d', 15,15,*([15]*5)))
    header = bytearray(100)
    struct.pack_into('>i', header, 24, (108+len(content))//2)
    shp = bytes(header)+struct.pack('>2i',1,len(content)//2)+content
    dbf_header=bytearray(32)
    struct.pack_into('<IHH', dbf_header,4,1,65,13)
    descriptor=bytearray(32)
    descriptor[:8]=b'lokal_id'
    descriptor[11]=ord('C')
    descriptor[16]=12
    dbf=bytes(dbf_header)+bytes(descriptor)+b'\r'+b' '+b'building'.ljust(12)
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('BYGNING.SHP',shp)
        archive.writestr('BYGNING.DBF',dbf)
        archive.writestr('BYGNING.PRJ','UTM_Zone_23N')


def _metadata():
    return dict(schemaVersion=4,vehicleAssetType='vehicle',structureAssetType='structure',
        grundkortSettlements=['0600NUK'],
        vehicleDefinition=dict(url='/vehicle.glb',realLengthM=7.7,tireDiameterM=1.2,altOffsetM=0.1),
        structureDefinition=dict(url='/structure.glb'),
        seedVehicleInstances=[dict(id='vehicle',lat=64,lon=-51,headingDeg=0,z=3,headlightsOn=True)],
        seedStructureInstances=[])


@visible
async def asset_coordinate_loading_offline() -> dict:
    qx,qy=acq.to_stereo(60.1375,-44.3006)
    assert '0102APL_Aappilattoq' in acq.nearby_settlements(qx,qy,9000)
    assert '0600NUK_Nuuk' not in acq.nearby_settlements(qx,qy,9000)
    north=acq.to_stereo(77.46666,-69.23155)
    assert '1700QNQ_Qaanaaq' in acq.nearby_settlements(*north,9000)
    for args in [(float('nan'),qy,9000),(qx,qy,-1)]:
        try:
            acq.nearby_settlements(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid coordinate accepted')

    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        sources=root/'grundkort'
        sources.mkdir()
        archive_path=sources/'0102APL_TekniskGrundkort_SHP.zip'
        _fixture_archive(archive_path)
        (root/'metadata.json').write_text(json.dumps(_metadata()))
        (root/'building_ground_samples.json').write_text(json.dumps(dict(schemaVersion=1,samples={'seed':3})))
        connection=sqlite3.connect(root/'assets.db',check_same_thread=False)
        schema.create(connection)
        entered, release=threading.Event(),threading.Event()
        attempts=[]
        def acquire(folder):
            attempts.append(folder)
            entered.set()
            assert release.wait(5), 'test worker not released'
            return acq.acquire_settlement(folder)
        lane=DemandLane('test-assets',acquire,1)
        ground=np.full((65,65),7,dtype=np.float32)
        confidence=np.ones((65,65),dtype=np.uint8)
        try:
            with (
                patch.object(acq,'_HERE',root),
                patch.object(acq.assets,'db',return_value=connection),
                patch.object(acq,'_lane',return_value=lane),
                patch.object(acq,'_ground_tile',return_value=(ground,confidence)),
                patch.object(viewer_assets,'_LOCAL_ASSETS_DB',root/'assets.db'),
                patch.object(acq.urllib.request,'urlopen',side_effect=AssertionError('network in offline gate')),
            ):
                # Actual HTTP handler, worker held pending: missing data isn't ready.
                request=Request({'type':'http','method':'GET','path':'/api/buildings',
                    'query_string':b'lat=60.1375&lon=-44.3006','headers':[]})
                response=await viewer_server._buildings(request)
                assert response.status_code==200,response.body
                size=struct.unpack('<I',response.body[:4])[0]
                payload=json.loads(response.body[4:4+size])
                assert payload['count']==0 and payload['shouldPoll']
                assert payload['buildingsStatus']=='loading'
                assert entered.wait(1)
                again=acq.request_for_point(qx,qy,9000)
                assert again['status']=='loading' and len(attempts)==1
                release.set()
                assert lane.wait_for_idle(3),lane.status()
                assert not lane.status()['failures'],lane.status()
                ready=acq.request_for_point(qx,qy,9000)
                assert ready['status']=='ready',ready
                assert len(attempts)==1
                buildings,_=viewer_assets.query_buildings(qx,qy,9000,qx,qy)
                assert len(buildings)==1 and buildings[0]['groundZ']==7
                assert acq.settlement_loaded(connection,'0102APL') # no roads is valid
                meta=json.loads((root/'metadata.json').read_text())
                assert '0102APL' in meta['grundkortSettlements']
                registry=json.loads((root/'building_ground_samples.json').read_text())
                assert registry['samples']['0102APL_building']==7

                # Rebuild dynamically discovered package from its persisted inventory.
                # Seed Nuuk archive isn't a fixture, remove just that test seed.
                meta['grundkortSettlements'].remove('0600NUK')
                (root/'metadata.json').write_text(json.dumps(meta))
                rebuilt=root/'rebuilt.db'
                result=rebuild.build_catalog(rebuilt,sources,root/'metadata.json',root/'building_ground_samples.json')
                with sqlite3.connect(rebuilt) as db:
                    assert acq.settlement_loaded(db,'0102APL')
                    assert db.execute("SELECT count(*) FROM assets WHERE type='BYGNING'").fetchone()[0]==1

                # A failed refresh rolls back deletion and imported buildings.
                before=connection.execute('SELECT * FROM assets').fetchall()
                with patch.object(rebuild,'_ingest_roads',side_effect=ValueError('bad road payload')):
                    try:
                        acq.acquire_settlement('0102APL_Aappilattoq')
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('corrupt package accepted')
                assert connection.execute('SELECT * FROM assets').fetchall()==before
                connection.execute('UPDATE assets SET enabled=0')
                connection.commit()
                acq.acquire_settlement('0102APL_Aappilattoq')
                assert connection.execute('SELECT enabled FROM assets').fetchone()[0]==0

                # Missing measured samples cannot become fake zero-height grounds.
                with patch.object(acq,'_ground_tile',return_value=(ground,np.zeros_like(confidence))):
                    try:
                        acq.acquire_settlement('0102APL_Aappilattoq')
                    except ValueError as exc:
                        assert 'no measured terrain' in str(exc)
                    else:
                        raise AssertionError('zero-confidence ground accepted')

                connection.execute('DELETE FROM assets')
                connection.commit()
                assert not acq.settlement_loaded(connection,'0102APL')
                # A DB miss reopens completed work after a catalog purge.
                assert acq.request_for_point(qx,qy,9000)['status']=='loading'
                assert lane.wait_for_idle(3)
                assert acq.request_for_point(qx,qy,9000)['status']=='ready'
                assert len(attempts)==2
        finally:
            release.set()
            lane.close()
            connection.close()

    # A village whose DEM is absent uses the MCP provider and persistence path.
    measured = (np.full((65,65),8,dtype=np.float32), np.ones((65,65),dtype=np.uint8))
    with (
        patch.object(acq, '_read_ground_tile', side_effect=[None,None,measured]),
        patch.object(acq, 'fetch_best_dem', return_value=dict(
            heightmap=measured[0],source='arcticdem_10m',verticalDatum='EGM2008')) as fetch,
        patch.object(acq.terrain, 'db', return_value=object()),
        patch.object(acq, 'write_dem') as write,
    ):
        assert acq._ground_tile('12-1942-61') is measured
        fetch.assert_called_once_with('12-1942-61')
        assert write.call_count == 1

    with sqlite3.connect(':memory:') as connection:
        schema.create(connection)
        connection.execute(
            "INSERT INTO assets(id,type,lat,lon) VALUES ('0102APL_road','VEJMIDTE',60,-44)"
        )
        assert not acq.settlement_loaded(connection,'0102APL')

    now=[100.0]
    attempts=[]
    def transient(folder):
        attempts.append(folder)
        if len(attempts)<2:
            raise TimeoutError('provider timed out')
        return {}
    lane=DemandLane('test-asset-retry',transient,1,clock=lambda:now[0])
    try:
        lane.replace_pending(['0102APL_Aappilattoq'])
        assert lane.wait_for_idle(1)
        lane.replace_pending(['0102APL_Aappilattoq'])
        assert len(attempts)==1
        now[0]+=3
        lane.replace_pending(['0102APL_Aappilattoq'])
        assert lane.wait_for_idle(1)
        assert len(attempts)==2 and not lane.status()['failures']
    finally:
        lane.close()
    return {'ok':True,'coordinateSelection':True,'httpLoading':True,'deduplication':True,
        'groundSampling':True,'atomicImport':True,'preservesVisibility':True,
        'rebuildRetention':True,'purgeRecovery':True,'boundedRetry':True}
