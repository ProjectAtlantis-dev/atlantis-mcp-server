import {waterShortcutClear} from './water-clearance.mjs';
import {groundProfile} from './vehicle-performance.mjs';
import {groundSegmentClear} from './ground-surface.mjs';
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
export function updateDestinationProgress(v){
 const route=v.mission.destinationRoute;if(!route)return;
 // Project onto the nearby forward corridor, rather than requiring the
 // steering path to hit every coarse-grid vertex exactly.
 let walked=0,best=route.index,bestDistance=Infinity;
 for(let i=route.index;i<route.points.length&&walked<=128;i++){
  const b=route.points[i],a=route.points[Math.max(0,i-1)];
  const dx=b.x-a.x,dy=b.y-a.y,length=dx*dx+dy*dy;
  const t=length?Math.max(0,Math.min(1,((v.position.x-a.x)*dx+(v.position.y-a.y)*dy)/length)):1;
  const p={x:a.x+t*dx,y:a.y+t*dy},d=distance(v.position,p);
  if(d<bestDistance&&d<=Math.max(16,route.stepM*2)&&groundSegmentClear(v.surface,v.position,p)){
   bestDistance=d;best=Math.min(route.points.length-1,i+(distance(v.position,b)<4?1:0));
  }
  walked+=Math.sqrt(length);
 }
 route.index=best;
}
export function localRouteTarget(v){
 const route=v.mission.destinationRoute;
 if(!route){
  const s=v.surface,t=v.mission.target;
  if(t.x<s.minX||t.y<s.minY||t.x>s.minX+(s.cols-1)*s.stepM||t.y>s.minY+(s.rows-1)*s.stepM)throw Error('complete destination route required beyond local terrain');
  return t;
 }
 updateDestinationProgress(v);
 const profile=v.boatProfile?{maxForwardMps:v.boatProfile.maxSpeedMs,planningDecelerationMps2:v.boatProfile.accelMs2,predictionSeconds:2}:groundProfile(v.definitionId);
 const horizon=profile.maxForwardMps**2/(2*profile.planningDecelerationMps2)+profile.maxForwardMps*profile.predictionSeconds;
 let i=route.index,length=distance(v.position,route.points[i]);
 for(;i<route.points.length-1;i++){
  const next=route.points[i+1],s=v.surface;
  if(next.x<s.minX+8||next.y<s.minY+8||next.x>s.minX+(s.cols-1)*s.stepM-8||next.y>s.minY+(s.rows-1)*s.stepM-8)break;
  const segment=distance(route.points[i],next);if(length+segment>horizon)break;length+=segment;
 }
 return route.points[i];
}
export function smoothLocalRoute(surface,start,points){
 const result=[];let anchor=start,i=0;
 while(i<points.length){
  let next=i;
  // Remove tiny grid stair-steps, keeping shortcuts in the planned corridor.
  for(let j=i+1;j<points.length&&distance(anchor,points[j])<=24;j++){
   if(groundSegmentClear(surface,anchor,points[j])&&waterShortcutClear(surface,anchor,points[j]))next=j;
  }
  result.push(points[next]);anchor=points[next];i=next+1;
 }
 return result;
}
