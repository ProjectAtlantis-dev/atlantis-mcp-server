import runpy,unittest
from pathlib import Path
from unittest.mock import Mock

class RadarAlertsTests(unittest.TestCase):
    def test_alert_joins_current_choices_and_never_skips_events_seen_only_by_later_snapshot(self):
        namespace=runpy.run_path(str(Path(__file__).parents[1]/'main.py'),init_globals={'visible':lambda fn:fn})
        function=namespace['alerts'];g=function.__globals__
        g['events']=Mock(return_value={'events':[{'sequence':5,'type':'target-tracked','targetId':'test','kind':'drone','detectedBy':[{'sensorId':'radar'}]}]})
        track={'id':'test','kind':'drone','headingDeg':270,'state':'tracked','availableActions':[{'siteId':'site','layerId':'toy'}]}
        g['observe']=Mock(return_value={'tick':20,'lastEventSequence':9,'tracks':[track]})
        result=function(4);self.assertEqual(result['next_sequence'],5)
        alert=result['alerts'][0];self.assertEqual(alert['incoming_type'],'drone');self.assertEqual(alert['heading_deg'],270);self.assertTrue(alert['ready_for_action'])
        track.update(state='engaged',availableActions=[])
        self.assertFalse(function(4)['alerts'][0]['ready_for_action'])
        g['observe'].return_value['tracks']=[]
        self.assertEqual(function(4)['alerts'][0]['current_state'],'not-currently-tracked')
        with self.assertRaises(ValueError):function(-1)

class IncomingGatewayTests(unittest.TestCase):
    def test_coordinates_and_destination_elevation_reach_the_shared_server_gateway(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from dynamic_functions.Terrain.Defense import gateway
        principal=SimpleNamespace(caller='Tester',permissions=['simulation'],scenario='test-world')
        calls=[]
        def command(method,path,payload=None):
            calls.append((method,path,payload))
            return {'origin':{'latitude':64,'longitude':-51,'altitudeM':10}} if method=='GET' else payload
        with patch.object(gateway.atlantis,'get_owner_usernames',return_value=['Tester']),patch.object(gateway.simulation_host,'command',side_effect=command),patch.object(gateway.terrain_adapter,'configuration',return_value={}),patch.object(gateway.terrain_adapter,'elevation_grid',return_value={'heights':[25]*9}):
            result=gateway.spawn(principal,incoming_type='drone',request_id='t',latitude=64,longitude=-51,heading_deg=270)
            self.assertEqual(result['destination'],{'x':0,'y':0,'z':15})
            self.assertEqual(result['headingDeg'],270)
            self.assertEqual(calls[-1][1],'/games/test-world/scenario-incoming')
            with self.assertRaises(ValueError):gateway.spawn(principal,incoming_type='drone',request_id='t',latitude=float('nan'),longitude=-51)
            principal.caller='Other'
            with self.assertRaises(PermissionError):gateway.spawn(principal,incoming_type='drone',request_id='t',latitude=64,longitude=-51)
