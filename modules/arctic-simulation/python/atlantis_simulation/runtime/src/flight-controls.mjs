import {returnPlan,arriveMission,resolveReturn} from './return-mission.mjs';
import {fixedWingProfile} from './fixed-wing-controls.mjs';
import {randomUUID} from 'node:crypto';
import {terminalMission,blockMission} from './drive-mission.mjs';
export const FLIGHT_PROFILES=Object.freeze({
 'black-hornet':{maxSpeedMps:8,climbMps:3,acceleration:3},
 'v22-osprey':{maxSpeedMps:25,climbMps:6,acceleration:4},
});
export const FLIGHT_MISSION_LIMIT_M=10000; // Simulation workload bound, not manufacturer range.
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export function attachFlight(controller,input,heightAt){
 const {id,ownerAccountId,definitionId,terrainAssetId,pose,surface,sourcePose}=input;
 const profile=definitionId==='rq180'?fixedWingProfile(input.definition):FLIGHT_PROFILES[definitionId];if(!profile)throw Error('aircraft controller profile unavailable');
 const ground=heightAt(surface,pose.x,pose.y),z=sourcePose?.z;
 if(ground===null||!Number.isFinite(z)||z<ground-.5)throw Error('saved aircraft pose is below verified terrain');
 const current=controller.vehicles.get(id);
 if(current){if(current.definitionId!==definitionId||current.ownerAccountId!==ownerAccountId||current.terrainAssetId!==terrainAssetId)throw Error('aircraft binding conflict');return controller.observe(id);}
 if([...controller.vehicles.values()].some(v=>v.terrainAssetId===terrainAssetId))throw Error('terrain instance already attached');
 controller.vehicles.set(id,{id,ownerAccountId,definitionId,terrainAssetId,surface:structuredClone(surface),flightProfile:profile,
  position:{x:pose.x,y:pose.y,z},headingRad:pose.headingRad,speedMps:0,distanceM:0,revision:0,lease:null,input:null,
  airborne:profile.kind==='fixed-wing'&&z>ground+5,controlStatus:'parked',flightPhase:'parked',rotorRpm:0,rotorAngleRad:0,pitchRad:0,rollRad:0});
 return controller.observe(id);
}
export function startFlight(v,{actor,ownerAccountId,destination,requestId,altitudeM,landing=false,returnDestination=null,waitForTask=false,autoReturn=false,takeoffHeadingDeg=null,waterLevelM=null},now){
 if(!v.flightProfile)throw Error('fly_to requires an aircraft');
 const fixed=v.flightProfile.kind==='fixed-wing';
 if(waterLevelM!==null&&!Number.isFinite(waterLevelM))throw Error('finite water level required');
 const maxMissionDistanceM=FLIGHT_MISSION_LIMIT_M;
 if(typeof landing!=='boolean')throw Error('landing must be boolean');
 if(fixed&&(landing||returnDestination?.landing))throw Error('fixed-wing landing is not implemented; use a flyover and loiter');
 if(takeoffHeadingDeg!==null&&(!fixed||!Number.isFinite(takeoffHeadingDeg)))throw Error('takeoff heading is only available for fixed-wing aircraft');
 if(fixed&&autoReturn&&!v.airborne)throw Error('auto_return requires an airborne departure for fixed-wing aircraft; provide return coordinates and flight altitude for a ground departure');
 if(v.ownerAccountId!==ownerAccountId)throw Error('aircraft ownership mismatch');
 if(typeof actor!=='string'||!actor||typeof requestId!=='string'||!requestId.trim()||requestId.length>128)throw Error('actor and requestId required');
 if(!destination||!Number.isFinite(destination.lat)||Math.abs(destination.lat)>85||!Number.isFinite(destination.lon)||Math.abs(destination.lon)>180||!Number.isFinite(altitudeM))throw Error('valid destination and altitude required');
 if(typeof waitForTask!=='boolean')throw Error('waitForTask must be boolean');
 if(requestId.startsWith('return:'))throw Error('return: request IDs are reserved');

 const previous=[...(v.missionHistory??[]),...(v.mission?[v.mission]:[])].find(m=>m.requestId===requestId);
 const plannedReturn=returnPlan(v,resolveReturn(v,autoReturn,returnDestination,previous,true),destination,maxMissionDistanceM,true);
 if(previous){if(previous.destination.lat!==destination.lat||previous.destination.lon!==destination.lon||previous.altitudeM!==altitudeM||(previous.waterLevelM??null)!==waterLevelM||previous.landing!==landing||(previous.takeoffHeadingDeg??null)!==takeoffHeadingDeg||JSON.stringify(previous.returnDestination??null)!==JSON.stringify(plannedReturn)||(previous.waitForTask??false)!==waitForTask)throw Error('mission requestId terms conflict');return structuredClone(previous);}
 if(!terminalMission(v.mission))throw Error('aircraft already has a mission');
 const o=v.surface.origin,target={x:(destination.lon-o.lon)*Math.PI/180*6378137*Math.cos(o.lat*Math.PI/180),y:(destination.lat-o.lat)*Math.PI/180*6378137};
 const distance=Math.hypot(target.x-v.position.x,target.y-v.position.y);
 if(distance>maxMissionDistanceM)throw Error('Flight exceeds the 10km simulation mission limit');
 if(v.mission){v.missionHistory??=[];v.missionHistory.push(v.mission);}
 if(fixed&&!v.airborne&&takeoffHeadingDeg!==null)v.headingRad=takeoffHeadingDeg*Math.PI/180;
 v.mission={takeoffHeadingDeg,waterLevelM,id:randomUUID(),type:'fly_to',actor,ownerAccountId,requestId,destination:{...destination},target,altitudeM,landing,waitForTask,autoReturn,leg:'outbound',returnDestination:plannedReturn,returnMissionId:null,status:'queued',reason:null,
  createdAt:now,updatedAt:now,remainingM:distance,arrivalToleranceM:fixed?25:1.5,travelledM:0,startDistanceM:v.distanceM,maxSpeedMps:v.flightProfile.maxSpeedMps};
 v.mission.journeyId=v.mission.id;
 v.missionTerrainReady=false;v.lease=null;v.input=null;return structuredClone(v.mission);
}
export function stepFlight(v,dt,now,heightAt){
 const m=v.mission,active=m&&['queued','running'].includes(m.status)&&v.missionTerrainReady;
 const h=heightAt(v.surface,v.position.x,v.position.y);
 let rpmTarget=v.position.z>(h??v.position.z)+2?900:0;
 if(active)rpmTarget=900;
 v.rotorRpm+=clamp(rpmTarget-v.rotorRpm,-300*dt,300*dt);
 v.rotorAngleRad=(v.rotorAngleRad+v.rotorRpm*Math.PI/30*dt)%(2*Math.PI);
 if(!active){v.speedMps=0;v.pitchRad=0;v.flightPhase=rpmTarget?'holding':'parked';v.controlStatus=m?`mission-${m.status}`:v.flightPhase;v.revision++;return;}
 const dx=m.target.x-v.position.x,dy=m.target.y-v.position.y,remaining=Math.hypot(dx,dy);
 m.remainingM=remaining;m.travelledM=v.distanceM-m.startDistanceM;m.status='running';m.updatedAt=now;
 const terminalApproach=remaining<=1.5;
 const highest=Math.max(...v.surface.heights.filter(Number.isFinite),...(v.surface.obstacles??[]).map(o=>o.maxZ??-Infinity));
 const cruiseZ=Math.max(m.altitudeM,highest+20),targetZ=terminalApproach?m.altitudeM:cruiseZ;
 const z=v.position.z+(v.rotorRpm>=850?clamp(targetZ-v.position.z,-v.flightProfile.climbMps*dt,v.flightProfile.climbMps*dt):0);
 const ready=v.rotorRpm>=850&&v.position.z>=cruiseZ-1;
 const targetSpeed=ready&&!terminalApproach?Math.min(v.flightProfile.maxSpeedMps,Math.sqrt(2*v.flightProfile.acceleration*remaining)):0;
 v.speedMps+=clamp(targetSpeed-v.speedMps,-v.flightProfile.acceleration*dt,v.flightProfile.acceleration*dt);
 const travel=Math.min(remaining,v.speedMps*dt),x=v.position.x+(remaining?dx/remaining*travel:0),y=v.position.y+(remaining?dy/remaining*travel:0);
 const ground=heightAt(v.surface,x,y);
 const obstacle=(v.surface.obstacles??[]).some(o=>x>=o.minX&&x<=o.maxX&&y>=o.minY&&y<=o.maxY&&z<o.maxZ+2);
 if(!Number.isFinite(ground)||z<ground-.1||obstacle){blockMission(v,!Number.isFinite(ground)?'flight-terrain-coverage-boundary':'flight-clearance',now);v.speedMps=0;return;}
 v.position={x,y,z};v.distanceM+=travel;
 if(travel>0){const desired=Math.atan2(-dx,dy),error=Math.atan2(Math.sin(desired-v.headingRad),Math.cos(desired-v.headingRad));v.headingRad+=clamp(error,-.8*dt,.8*dt);}
 v.pitchRad=-.12*v.speedMps/v.flightProfile.maxSpeedMps;
 v.flightPhase=!ready&&!terminalApproach?'takeoff':terminalApproach?(m.landing?'landing':'positioning'):'cruise';
 if(remaining<=1.5&&Math.abs(z-m.altitudeM)<.2&&v.speedMps<.1){arriveMission(m,m.landing?'landed':'arrived-hovering',now);v.flightPhase=m.landing?'parked':'holding';}
 v.controlStatus=`mission-${m.status}`;v.revision++;
}
