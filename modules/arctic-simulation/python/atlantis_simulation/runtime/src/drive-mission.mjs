import {returnPlan,arriveMission} from './return-mission.mjs';
import {randomUUID} from 'node:crypto';

export const terminalMission = mission => !mission || ['completed','cancelled','failed'].includes(mission.status);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const angle=x=>Math.atan2(Math.sin(x),Math.cos(x));
export function missionView(v){return v.mission ? structuredClone(v.mission) : null;}
export function startMission(v,{actor,ownerAccountId,destination,requestId,returnDestination=null,waitForTask=false},now){
  if(v.ownerAccountId!==ownerAccountId)throw Error('vehicle ownership mismatch');
  if(typeof actor!=='string'||!actor)throw Error('mission actor required');
  if(typeof requestId!=='string'||!requestId.trim()||requestId.length>128)throw Error('requestId required (1..128 characters)');
  if(!destination||!Number.isFinite(destination.lat)||Math.abs(destination.lat)>85||!Number.isFinite(destination.lon)||Math.abs(destination.lon)>180)throw Error('invalid destination');
  if(typeof waitForTask!=='boolean')throw Error('waitForTask must be boolean');
  if(requestId.startsWith('return:'))throw Error('return: request IDs are reserved');
  const plannedReturn=returnPlan(v,returnDestination,destination,20000);
  const previous=[...(v.missionHistory??[]),...(v.mission?[v.mission]:[])].find(m=>m.requestId===requestId);
  if(previous){
    if(previous.destination.lat!==destination.lat||previous.destination.lon!==destination.lon||JSON.stringify(previous.returnDestination??null)!==JSON.stringify(plannedReturn)||(previous.waitForTask??false)!==waitForTask)throw Error('mission requestId terms conflict');
    return structuredClone(previous);
  }
  if(!terminalMission(v.mission))throw Error('vehicle already has a mission; cancel it first');
  if(v.lease&&v.lease.expiresAt>now)throw Error('release manual control before dispatch');
  const origin=v.surface.origin;
  const target={x:(destination.lon-origin.lon)*Math.PI/180*6378137*Math.cos(origin.lat*Math.PI/180),y:(destination.lat-origin.lat)*Math.PI/180*6378137};
  const distance=Math.hypot(target.x-v.position.x,target.y-v.position.y);
  if(distance>20000)throw Error('destination exceeds the 20km mission limit');
  if(v.mission){v.missionHistory??=[];v.missionHistory.push(v.mission);}
  v.lease=null;v.input=null;
  v.mission={id:randomUUID(),type:'drive_to',requestId,actor,ownerAccountId,status:'queued',reason:null,
    destination:{...destination},target,waitForTask,leg:'outbound',returnDestination:plannedReturn,returnMissionId:null,createdAt:now,updatedAt:now,remainingM:distance,
    startDistanceM:v.distanceM,travelledM:0,arrivalToleranceM:1.5,maxSpeedMps:4,
    lastProgressAt:now,bestRemainingM:distance};
  v.mission.journeyId=v.mission.id;
  v.missionTerrainReady=false;
  return missionView(v);
}
export function changeMission(v,{missionId,action,ownerAccountId},now){
  if(v.ownerAccountId!==ownerAccountId)throw Error('vehicle ownership mismatch');
  if(action==='complete_task'){
    const previous=(v.missionHistory??[]).find(m=>m.id===missionId&&m.reason==='task-completed');
    if(previous)return structuredClone(previous);
  }
  const m=v.mission;if(!m||m.id!==missionId)throw Error('mission ID does not match current mission');
  if(action==='complete_task'){
    if(m.status==='completed'&&m.reason==='task-completed')return missionView(v);
    if(m.status!=='awaiting_task')throw Error('task completion requires awaiting_task state');
    m.status='completed';m.reason='task-completed';m.taskCompletedAt=now;m.updatedAt=now;
    return missionView(v);
  }
  if(!['pause','resume','cancel'].includes(action))throw Error('invalid mission action');
  if(terminalMission(m))throw Error('mission is already terminal');
  if(action==='resume'){
    if(!['paused','blocked'].includes(m.status))throw Error('only paused/blocked missions can resume');
    m.status=m.pausedFrom==='awaiting_task'?'awaiting_task':'queued';m.reason=m.status==='awaiting_task'?'arrived-awaiting-task':null;delete m.pausedFrom;v.missionTerrainReady=false;m.lastProgressAt=now;m.bestRemainingM=m.remainingM;
  }else{if(action==='pause'&&m.status!=='paused')m.pausedFrom=m.status;m.status=action==='pause'?'paused':'cancelled';m.reason=action==='pause'?'operator-paused':'operator-cancelled';}
  m.updatedAt=now;v.input=null;return missionView(v);
}
export function blockMission(v,reason,now){
  if(!v.mission||terminalMission(v.mission))return;
  v.mission.status='blocked';v.mission.reason=reason;v.mission.updatedAt=now;v.input=null;
}
export function missionInput(v,now){
  const m=v.mission;if(!m||terminalMission(m))return null;
  if(!['queued','running'].includes(m.status))return {throttle:0,steering:0,brake:1};
  if(!v.missionTerrainReady)return {throttle:0,steering:0,brake:1};
  m.remainingM=Math.hypot(m.target.x-v.position.x,m.target.y-v.position.y);
  m.travelledM=Math.max(0,v.distanceM-m.startDistanceM);
  if(m.remainingM<=m.arrivalToleranceM){
    if(Math.abs(v.speedMps)<.05){arriveMission(m,'arrived',now);}
    return {throttle:0,steering:0,brake:1};
  }
  m.status='running';m.updatedAt=now;
  if(m.remainingM<m.bestRemainingM-.25){m.bestRemainingM=m.remainingM;m.lastProgressAt=now;}
  const navigation=m.navigation;
  let target=m.target;
  if(navigation){
    while(navigation.index<navigation.points.length-1&&Math.hypot(navigation.points[navigation.index].x-v.position.x,navigation.points[navigation.index].y-v.position.y)<3)navigation.index++;
    target=navigation.points[navigation.index];
    if(!target){blockMission(v,'route-has-no-waypoints',now);return {throttle:0,steering:0,brake:1};}
    if(!navigation.complete&&navigation.index===navigation.points.length-1&&Math.hypot(target.x-v.position.x,target.y-v.position.y)<3){v.missionTerrainReady=false;return {throttle:0,steering:0,brake:1};}
  }
  const desired=Math.atan2(-(target.x-v.position.x),target.y-v.position.y);
  const error=angle(desired-v.headingRad);
  const waypointDistance=Math.hypot(target.x-v.position.x,target.y-v.position.y);
  // A detour can legitimately increase distance to the final destination.
  // Measure progress toward the active waypoint, including reverse travel.
  if(m.progressWaypoint!==`${target.x},${target.y}` || waypointDistance<(m.bestWaypointRemainingM??Infinity)-.25){
    m.progressWaypoint=`${target.x},${target.y}`;m.bestWaypointRemainingM=waypointDistance;m.lastProgressAt=now;
  }
  if(now-m.lastProgressAt>60000){blockMission(v,'no-route-progress',now);return {throttle:0,steering:0,brake:1};}
  const cornerSpeed=navigation&&navigation.index<navigation.points.length-1?Math.max(1,Math.min(m.maxSpeedMps,waypointDistance*.5)):m.maxSpeedMps;
  if(!m.backingRoute&&Math.abs(error)>Math.PI*.85&&waypointDistance>15)m.backingRoute=true;
  if(m.backingRoute&&Math.abs(error)<Math.PI*.6)m.backingRoute=false;
  if(m.backingRoute){
    if(v.speedMps>.05)return {throttle:0,steering:0,brake:1};
    return {throttle:clamp((-1.5-v.speedMps)*.7,-1,0),steering:clamp(-angle(error-Math.PI)*2,-1,1),brake:v.speedMps < -1.6?.4:0};
  }
  if(!m.reversing&&Math.abs(error)>1&&waypointDistance<15)m.reversing=true;
  if(m.reversing&&Math.abs(error)<.4)m.reversing=false;
  if(m.reversing){
    if(v.speedMps>.05)return {throttle:0,steering:0,brake:1};
    return {throttle:clamp((-1.5-v.speedMps)*.7,-1,0),steering:clamp(-error*2,-1,1),brake:v.speedMps < -1.6?.4:0};
  }
  const targetSpeed=Math.min(cornerSpeed,Math.sqrt(2*2*Math.max(0,m.remainingM-m.arrivalToleranceM)))*Math.max(.3,Math.cos(error));
  return {throttle:clamp((targetSpeed-v.speedMps)*.7,0,1),steering:clamp(error*2,-1,1),brake:v.speedMps>targetSpeed+.15?.5:0};
}
