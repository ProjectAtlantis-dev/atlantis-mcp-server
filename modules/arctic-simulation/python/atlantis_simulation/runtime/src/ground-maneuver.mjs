import {advanceGroundPose} from './ground-kinematics.mjs';
import {groundProfile} from './vehicle-performance.mjs';
import {groundSegmentClear} from './ground-surface.mjs';
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
class Open {
 constructor(){this.a=[];}
 push(n){const a=this.a;let i=a.length;a.push(n);while(i){const p=(i-1)>>1;if(a[p].f<=n.f)break;a[i]=a[p];i=p;}a[i]=n;}
 pop(){const a=this.a,n=a[0],last=a.pop();if(a.length){let i=0;while(2*i+1<a.length){let c=2*i+1;if(c+1<a.length&&a[c+1].f<a[c].f)c++;if(a[c].f>=last.f)break;a[i]=a[c];i=c;}a[i]=last;}return n;}
}
function searchManeuver(v,target,gears){
 const profile=groundProfile(v.definitionId),speed=3,dt=1/6;
 const finalGoal=Math.hypot(target.x-v.mission.target.x,target.y-v.mission.target.y)<.01;
 const tolerance=finalGoal?v.mission.arrivalToleranceM*.8:2;
 const key=(p,gear)=>`${Math.round(p.position.x)},${Math.round(p.position.y)},${Math.round(wrap(p.headingRad)*18/Math.PI)},${gear}`;
 const distance=p=>Math.hypot(p.position.x-target.x,p.position.y-target.y);
 const open=new Open(),seen=new Map();const start={pose:v,g:0,f:distance(v),parent:null,gear:0};open.push(start);seen.set(key(v,0),0);
 let expanded=0;
 while(open.a.length&&expanded++<30000){
  const n=open.pop();if(n.g>seen.get(key(n.pose,n.gear)))continue;
  if(distance(n.pose)<tolerance){const path=[];for(let p=n;p.parent;p=p.parent)path.push({gear:p.gear,steering:p.steering,length:p.length});return path.reverse();}
  for(const gear of gears)for(const steering of [-1,-.5,0,.5,1]){
   let pose={...n.pose,speedMps:gear*speed},clear=true,travel=0;
   for(let i=0;i<6;i++){
    const next=advanceGroundPose(pose,{throttle:gear*profile.rollingDecelerationMps2/profile.accelerationMps2,steering,brake:0},dt);
    if(!groundSegmentClear(v.surface,pose.position,next.position,next.headingRad)){clear=false;break;}pose=next;travel+=speed*dt;
    if(distance(pose)<tolerance)break;
   }
   if(!clear)continue;
   const g=n.g+travel*(gear<0?4:1)+(n.gear&&gear!==n.gear?6:0)+Math.abs(steering)*.1;
   const k=key(pose,gear);if(g>=(seen.get(k)??Infinity))continue;
   seen.set(k,g);open.push({pose,g,f:g+distance(pose),parent:n,gear,steering,length:travel});
  }
 }
 return null;
}
export function planGroundManeuver(v,target){
 // Reverse distance costs more, so an ordinary forward turn beats backing
 // down a road, while a short parking correction can beat a full loop.
 return searchManeuver(v,target,[1,-1]);
}
export function maneuverInput(v,target,dt,goalIndex){
 const m=v.mission;let plan=m.maneuverPlan;
 const targetKey=`${target.x},${target.y}`;
 if(plan&&plan.target!==targetKey){delete m.maneuverPlan;plan=null;}
 if(!plan){
  // Stop before building a trajectory whose initial velocity is zero.
  if(Math.abs(v.speedMps)>.05)return {throttle:0,steering:0,brake:1};
  const segments=planGroundManeuver(v,target);
  if(!segments?.length){
   m.status='blocked';m.reason='no-feasible-turn-within-search-budget';
   return {throttle:0,steering:0,brake:1};
  }
  plan=m.maneuverPlan={target:targetKey,goal:{...target},goalIndex,segments,index:0,progress:0,lastDistance:v.distanceM};
 }
 plan.progress+=v.distanceM-plan.lastDistance;plan.lastDistance=v.distanceM;
 while(plan.index<plan.segments.length&&plan.progress>=plan.segments[plan.index].length){plan.progress-=plan.segments[plan.index].length;plan.index++;m.lastProgressAt=m.updatedAt;}
 const segment=plan.segments[plan.index];
 if(!segment){delete m.maneuverPlan;return {throttle:0,steering:0,brake:1};}
 if(v.speedMps*segment.gear<-.05)return {throttle:0,steering:0,brake:1};
 const targetSpeed=segment.gear*3;
 const input={throttle:Math.max(-1,Math.min(1,(targetSpeed-v.speedMps)*.7)),steering:segment.steering,brake:Math.abs(v.speedMps)>3.1?.5:0};
 const next=advanceGroundPose(v,input,dt);
 if(!groundSegmentClear(v.surface,v.position,next.position,next.headingRad)){delete m.maneuverPlan;return {throttle:0,steering:0,brake:1};}
 m.maneuver='planned-turn';return input;
}
