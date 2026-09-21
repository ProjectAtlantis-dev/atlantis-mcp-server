import {surfaceHeight} from './ground-surface.mjs';
import {randomUUID} from 'node:crypto';

// Capture in the authoritative dispatch transaction, not in a prior observe call.
export function resolveReturn(v,autoReturn,explicit,previous,flight=false){
 if(typeof autoReturn!=='boolean')throw Error('autoReturn must be boolean');
 if(autoReturn&&explicit!==null)throw Error('autoReturn cannot be combined with explicit return coordinates');
 if(previous&&(previous.autoReturn??false)!==autoReturn)throw Error('mission requestId terms conflict');
 if(!autoReturn)return explicit;
 if(previous)return previous.returnDestination;
 const o=v.surface.origin;
 const result={lat:o.lat+v.position.y/6378137*180/Math.PI,lon:o.lon+v.position.x/(6378137*Math.cos(o.lat*Math.PI/180))*180/Math.PI};
 if(flight){
  const ground=surfaceHeight(v.surface,v.position.x,v.position.y);
  if(!Number.isFinite(ground))throw Error('verified departure elevation required for autoReturn');
  result.landing=v.position.z<=ground+.5;
  result.altitudeM=result.landing?ground+.3:v.position.z;
 }
 return result;
}

// Return instructions are part of the original durable command, not a browser callback.
export function returnPlan(v, destination, outbound, maxDistanceM, flight=false) {
  if (destination == null) return null;
  const keys=flight?['lat','lon','altitudeM','landing']:['lat','lon'];
  if (typeof destination!=='object'||Object.keys(destination).some(k=>!keys.includes(k))||
      !Number.isFinite(destination.lat)||Math.abs(destination.lat)>85||
      !Number.isFinite(destination.lon)||Math.abs(destination.lon)>180) throw Error('invalid return destination');
  const result={lat:destination.lat,lon:destination.lon};
  if (flight) {
    if (!Number.isFinite(destination.altitudeM)||typeof destination.landing!=='boolean') throw Error('return flight altitude and landing required');
    result.altitudeM=destination.altitudeM;result.landing=destination.landing;
  }
  const o=v.surface.origin;
  const distance=Math.hypot((result.lon-outbound.lon)*Math.PI/180*6378137*Math.cos(o.lat*Math.PI/180),
    (result.lat-outbound.lat)*Math.PI/180*6378137);
  if(distance>maxDistanceM) throw Error('return destination exceeds mission distance limit');
  return result;
}

export function arriveMission(m, reason, now) {
  m.status=m.waitForTask?'awaiting_task':'completed';
  m.reason=m.waitForTask?'arrived-awaiting-task':reason;
  m.arrivedAt=now;m.updatedAt=now;
}

export function queueCompletedReturn(v, now) {
  const outbound=v.mission, destination=outbound?.returnDestination;
  if(outbound?.status!=='completed'||!destination||outbound.returnMissionId) return;
  const o=v.surface.origin;
  const target={x:(destination.lon-o.lon)*Math.PI/180*6378137*Math.cos(o.lat*Math.PI/180),
    y:(destination.lat-o.lat)*Math.PI/180*6378137};
  const distance=Math.hypot(target.x-v.position.x,target.y-v.position.y), id=randomUUID();
  outbound.returnMissionId=id;
  v.missionHistory??=[];v.missionHistory.push(outbound);
  v.mission={id,type:outbound.type,requestId:`return:${outbound.id}`,actor:outbound.actor,
    ownerAccountId:outbound.ownerAccountId,leg:'return',parentMissionId:outbound.id,journeyId:outbound.journeyId??outbound.id,
    status:'queued',reason:null,waitForTask:false,destination:{lat:destination.lat,lon:destination.lon},target,
    returnDestination:null,returnMissionId:null,createdAt:now,updatedAt:now,remainingM:distance,
    startDistanceM:v.distanceM,travelledM:0,arrivalToleranceM:outbound.arrivalToleranceM,maxSpeedMps:outbound.maxSpeedMps,
    ...(v.flightProfile?{altitudeM:destination.altitudeM,landing:destination.landing,waterLevelM:outbound.waterLevelM??null}:{lastProgressAt:now,bestRemainingM:distance})};
  v.missionTerrainReady=false;v.input=null;v.controlStatus='mission-queued';
}
