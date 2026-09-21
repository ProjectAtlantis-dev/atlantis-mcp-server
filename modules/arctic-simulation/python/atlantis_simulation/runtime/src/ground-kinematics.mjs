import {maneuverInput} from './ground-maneuver.mjs';
import {groundSegmentClear} from './ground-surface.mjs';

import {groundProfile} from './vehicle-performance.mjs';
export const GROUND_PROFILE=Object.freeze(groundProfile('patria-amv'));
export function advanceGroundPose(v,input,dt){
 const p=groundProfile(v.definitionId);
 let speed=Math.max(-p.maxReverseMps,Math.min(p.maxForwardMps,v.speedMps+input.throttle*p.accelerationMps2*dt));
 speed=Math.sign(speed)*Math.max(0,Math.abs(speed)-(input.brake*p.brakingMps2+p.rollingDecelerationMps2)*dt);
 const distance=speed*dt,curvature=Math.tan(input.steering*p.maxSteeringRad)/p.wheelbaseM;
 const heading=v.headingRad+distance*curvature;
 // Integrate the bicycle arc analytically so planning and live tick sizes
 // produce the same trajectory rather than drifting into a checked obstacle.
 const dx=Math.abs(curvature)<1e-10?-Math.sin(v.headingRad)*distance:(Math.cos(heading)-Math.cos(v.headingRad))/curvature;
 const dy=Math.abs(curvature)<1e-10?Math.cos(v.headingRad)*distance:(Math.sin(heading)-Math.sin(v.headingRad))/curvature;
 return {definitionId:v.definitionId,position:{x:v.position.x+dx,y:v.position.y+dy},headingRad:heading,speedMps:speed};
}
function predict(v,input,dt){
 let pose=v;
 for(let t=0;t<groundProfile(v.definitionId).predictionSeconds;t+=dt){
  const next=advanceGroundPose(pose,input,dt);
  if(!groundSegmentClear(v.surface,pose.position,next.position,next.headingRad))return null;
  pose=next;
 }
 const result=pose;
 for(let t=0;Math.abs(pose.speedMps)>.01&&t<10;t+=dt){
  const next=advanceGroundPose(pose,{throttle:0,steering:0,brake:1},dt);
  if(!groundSegmentClear(v.surface,pose.position,next.position,next.headingRad))return null;
  pose=next;
 }
 return result;
}

// Choose controls using the same steering model and terrain checks as the tick.
// Route points remain guidance; neither prediction nor planning writes position.
export function terrainCheckedMissionInput(v,requested,dt){
 const m=v.mission,nav=m.navigation;
 if(m.remainingM<=m.arrivalToleranceM)return requested;
 let target=nav?.points[nav.index]??m.target,goalIndex=nav?.index;
 if(m.maneuverPlan){target=m.maneuverPlan.goal;goalIndex=m.maneuverPlan.goalIndex;}
 else if(nav&&nav.index<nav.points.length-1&&Math.hypot(target.x-v.position.x,target.y-v.position.y)<3){goalIndex=nav.index+1;target=nav.points[goalIndex];}
 if(m.maneuverPlan){
  if(Math.hypot(target.x-v.position.x,target.y-v.position.y)<(Math.hypot(target.x-m.target.x,target.y-m.target.y)<.01?m.arrivalToleranceM*.8:2)){
   if(nav)nav.index=Math.max(nav.index,goalIndex);delete m.maneuverPlan;
  }else{const input=maneuverInput(v,target,dt,goalIndex);if(input)return input;}
 }
 const bearing=Math.atan2(-(target.x-v.position.x),target.y-v.position.y);
 const error=Math.atan2(Math.sin(bearing-v.headingRad),Math.cos(bearing-v.headingRad));
 const needsTurn=Math.abs(error)>Math.PI/2&&m.remainingM>m.arrivalToleranceM;
 if(!needsTurn&&predict(v,requested,dt)){delete m.maneuver;return requested;}
 if(needsTurn){
  const profile=groundProfile(v.definitionId),radius=profile.wheelbaseM/Math.tan(profile.maxSteeringRad);
  const distance=Math.hypot(target.x-v.position.x,target.y-v.position.y);
  if(distance>radius*3){const t=radius*3/distance;target={x:v.position.x+(target.x-v.position.x)*t,y:v.position.y+(target.y-v.position.y)*t};}
 }
 const input=maneuverInput(v,target,dt,goalIndex);
 m.maneuver=input?'planned-turn':'no-feasible-turn';
 return input??{throttle:0,steering:0,brake:1};
}
