import {waterShortcutClear} from './water-clearance.mjs';
import {planBoatManeuver} from './boat-maneuver.mjs';
import {startMission,blockMission} from './drive-mission.mjs';
import {groundHazard,groundSegmentClear} from './ground-surface.mjs';
import {updateDestinationProgress} from './route-guidance.mjs';
import {arriveMission} from './return-mission.mjs';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const angle=x=>Math.atan2(Math.sin(x),Math.cos(x));
export function attachBoat(controller,input){
 const {id,ownerAccountId,definitionId,terrainAssetId,pose,surface,definition}=input;
 const profile=definition?.boat;
 for(const key of ['maxSpeedMs','accelMs2','rudderTurnRadS2','yawDamping'])if(!Number.isFinite(profile?.[key])||profile[key]<=0)throw Error(`boat catalog requires ${key}`);
 if(groundHazard({...surface,navigationDomain:'water'},pose))throw Error('Boat must start on verified water');
 const current=controller.vehicles.get(id);
 if(current){if(current.definitionId!==definitionId||current.ownerAccountId!==ownerAccountId||current.terrainAssetId!==terrainAssetId)throw Error('boat binding conflict');return controller.observe(id);}
 if([...controller.vehicles.values()].some(v=>v.terrainAssetId===terrainAssetId))throw Error('terrain instance already attached');
 if(!Number.isFinite(input.sourcePose?.z))throw Error('boat water level required');
 controller.vehicles.set(id,{id,ownerAccountId,definitionId,terrainAssetId,boatProfile:structuredClone(profile),surface:structuredClone({...surface,navigationDomain:'water'}),
  waterLevelM:input.sourcePose.z,position:{x:pose.x,y:pose.y,z:input.sourcePose.z},headingRad:pose.headingRad,speedMps:0,distanceM:0,signedDistanceM:0,yawRate:0,revision:0,lease:null,input:null,controlStatus:'stopped'});
 return controller.observe(id);
}
export function startBoat(v,payload,now){if(!v.boatProfile)throw Error('sail_to requires a boat');return startMission(v,payload,now);}
/** Mission autopilot: bounded surge acceleration and catalog rudder/yaw response. */
export function stepBoat(v,dt,now){
 const m=v.mission,p=v.boatProfile;
 if(!m||!['queued','running'].includes(m.status)||!v.missionTerrainReady){v.speedMps=0;v.yawRate=0;v.controlStatus=m?`mission-${m.status}`:'stopped';v.revision++;return;}
 m.remainingM=Math.hypot(m.target.x-v.position.x,m.target.y-v.position.y);m.travelledM=v.distanceM-m.startDistanceM;
 if(m.remainingM<=m.arrivalToleranceM&&Math.abs(v.speedMps)<.1){arriveMission(m,'arrived',now);v.speedMps=0;return;}
 m.status='running';m.updatedAt=now;updateDestinationProgress(v);
 const nav=m.navigation;
 if(!nav?.points.length){blockMission(v,'water-route-unavailable',now);return;}
 while(nav.index<nav.points.length-1&&Math.hypot(nav.points[nav.index].x-v.position.x,nav.points[nav.index].y-v.position.y)<5&&groundSegmentClear(v.surface,v.position,nav.points[nav.index+1]))nav.index++;
 // Follow a water-checked lookahead on the route, not each tiny grid vertex.
 const radius=p.maxSpeedMs*p.yawDamping/p.rudderTurnRadS2;
 const lookahead=2*radius;
 while(nav.index<nav.points.length-1){
  const b=nav.points[nav.index],a=nav.points[Math.max(0,nav.index-1)];
  const dx=nav.index===0?nav.points[1].x-b.x:b.x-a.x,dy=nav.index===0?nav.points[1].y-b.y:b.y-a.y,length=dx*dx+dy*dy;
  const passed=length>0&&((v.position.x-b.x)*dx+(v.position.y-b.y)*dy)>=0;
  if(!passed||!groundSegmentClear(v.surface,v.position,nav.points[nav.index+1]))break;
  nav.index++;
 }
 if(!nav.complete&&nav.index===nav.points.length-1&&Math.hypot(nav.points[nav.index].x-v.position.x,nav.points[nav.index].y-v.position.y)<5){
  v.missionTerrainReady=false;v.speedMps=0;v.yawRate=0;m.reason='awaiting-water-route-extension';v.revision++;return;
 }
 let target=nav.points[nav.index],left=lookahead,a=v.position;
 for(let i=nav.index;i<nav.points.length;i++){
  const b=nav.points[i],length=Math.hypot(b.x-a.x,b.y-a.y),t=length?Math.min(1,left/length):1;
  const aim={x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t};
  if(!groundSegmentClear(v.surface,v.position,aim)||!waterShortcutClear(v.surface,v.position,aim))break;
  target=aim;left-=length;if(left<=0)break;a=b;
 }
 const dx=target.x-v.position.x,dy=target.y-v.position.y,distance=Math.hypot(dx,dy);
 const error=angle(Math.atan2(-dx,dy)-v.headingRad);
 const turnKey=`${target.x},${target.y}`;
 if(m.approachTurn?.target!==turnKey)delete m.approachTurn;
 if(m.approachTurn&&Math.abs(error)<.1)delete m.approachTurn;
 if(!m.approachTurn&&!m.boatManeuver&&!m.boatTurnTarget&&distance<2*radius&&Math.abs(error)>.25){
  const sign=Math.sign(error),cx=v.position.x-sign*Math.cos(v.headingRad)*radius,cy=v.position.y-sign*Math.sin(v.headingRad)*radius;
  // A point inside the current turning circle cannot be intercepted by
  // tightening toward it. Take the opposite arc until a tangent approach opens.
  if(Math.hypot(target.x-cx,target.y-cy)<radius+m.arrivalToleranceM){
   if(nav.complete&&nav.index===nav.points.length-1)m.approachTurn={target:turnKey,sign:-sign};
   else m.boatTurnTarget={...target};
  }
 }
 const steeringError=m.approachTurn?m.approachTurn.sign*Math.PI:error;
 let desiredSpeed=m.remainingM<=m.arrivalToleranceM?0:Math.min(p.maxSpeedMs,Math.sqrt(2*p.accelMs2*Math.max(0,m.remainingM-.5)),Math.sqrt(2*p.accelMs2*distance))*Math.max(.25,Math.cos(error));
 if(m.boatTurnTarget){
  desiredSpeed=0;
  if(Math.abs(v.speedMps)<.05){
   const segments=planBoatManeuver(v,m.boatTurnTarget);delete m.boatTurnTarget;
   if(segments===null){blockMission(v,'no-feasible-water-maneuver',now);v.speedMps=0;return;}
   if(segments.length){m.boatManeuver={segments,index:0,progress:0,lastDistance:v.distanceM};m.reason='maneuvering-in-verified-water';}
  }
 }
 let maneuver=m.boatManeuver;
 if(maneuver){
  maneuver.progress+=v.distanceM-maneuver.lastDistance; maneuver.lastDistance=v.distanceM;
  while(maneuver.index<maneuver.segments.length&&maneuver.progress>=maneuver.segments[maneuver.index].length){maneuver.progress-=maneuver.segments[maneuver.index].length;maneuver.index++;m.lastProgressAt=now;}
  if(maneuver.index>=maneuver.segments.length){delete m.boatManeuver;maneuver=null;if(m.reason==='maneuvering-in-verified-water')m.reason=null;}
 }
 const segment=maneuver?.segments[maneuver.index];
 if(segment)desiredSpeed=v.speedMps*segment.gear<-.05?0:segment.gear*Math.min(3,segment.gear<0?p.reverseMaxSpeedMs:p.maxSpeedMs);
 const speed=v.speedMps+clamp(desiredSpeed-v.speedMps,-p.accelMs2*dt,p.accelMs2*dt);
 const maxYaw=p.rudderTurnRadS2/p.yawDamping*Math.min(1,Math.abs(speed)/p.maxSpeedMs);
 v.yawRate+=clamp(clamp(segment?segment.steering*maxYaw:(m.approachTurn||(nav.complete&&nav.index===nav.points.length-1))?steeringError:2*speed*Math.sin(error)/Math.max(distance,1),-maxYaw,maxYaw)-v.yawRate,-p.rudderTurnRadS2*dt,p.rudderTurnRadS2*dt);
 const heading=v.headingRad+v.yawRate*dt;
 const next={x:v.position.x-Math.sin(heading)*speed*dt,y:v.position.y+Math.cos(heading)*speed*dt};
 if(!groundSegmentClear(v.surface,v.position,next)){
  v.speedMps=0;v.yawRate=0;delete m.boatManeuver;delete m.boatTurnTarget;
  const segments=planBoatManeuver(v,target);
  if(segments?.length){m.boatManeuver={segments,index:0,progress:0,lastDistance:v.distanceM};m.reason='maneuvering-in-verified-water';m.lastProgressAt=now;}
  else if(segments===null){blockMission(v,'no-feasible-water-maneuver',now);}
  else {v.missionTerrainReady=false;m.reason='awaiting-water-route-extension';}
 }else{
  v.position={...next,z:v.waterLevelM};v.headingRad=heading;v.speedMps=speed;v.distanceM+=Math.abs(speed)*dt;v.signedDistanceM+=speed*dt;
  if(distance<(m.bestWaypointRemainingM??Infinity)-.25||m.progressWaypoint!==nav.index){m.bestWaypointRemainingM=distance;m.progressWaypoint=nav.index;m.lastProgressAt=now;}
  if(now-m.lastProgressAt>60000){blockMission(v,'water-route-turn-unreachable',now);v.speedMps=0;}
 }
 v.controlStatus=`mission-${m.status}`;v.revision++;
}
