import {randomUUID} from 'node:crypto';

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
    ...(v.flightProfile?{altitudeM:destination.altitudeM,landing:destination.landing}:{lastProgressAt:now,bestRemainingM:distance})};
  v.missionTerrainReady=false;v.input=null;v.controlStatus='mission-queued';
}
