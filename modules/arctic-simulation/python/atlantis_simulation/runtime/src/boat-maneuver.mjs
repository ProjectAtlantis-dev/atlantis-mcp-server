import {groundSegmentClear} from './ground-surface.mjs';
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
class Frontier {
 constructor(){this.items=[];}
 push(n){const a=this.items;let i=a.length;a.push(n);while(i){const p=(i-1)>>1;if(a[p].f<=n.f)break;a[i]=a[p];i=p;}a[i]=n;}
 pop(){const a=this.items,n=a[0],last=a.pop();if(a.length){let i=0;while(i*2+1<a.length){let c=i*2+1;if(c+1<a.length&&a[c+1].f<a[c].f)c++;if(a[c].f>=last.f)break;a[i]=a[c];i=c;}a[i]=last;}return n;}
}
/** Bounded forward/reverse trajectory search through verified water. */
export function planBoatManeuver(v,target){
 const p=v.boatProfile,radius=p.maxSpeedMs*p.yawDamping/p.rudderTurnRadS2;
 const speed=Math.min(3,p.maxSpeedMs),gears=p.reverseMaxSpeedMs>0?[1,-1]:[1];
 const distance=q=>Math.hypot(q.x-target.x,q.y-target.y);
 const key=(q,gear)=>`${Math.round(q.x)},${Math.round(q.y)},${Math.round(wrap(q.heading)*18/Math.PI)},${gear}`;
 const pose={...v.position,heading:v.headingRad},start={pose,gear:0,g:0,f:distance(pose),parent:null};
 const open=new Frontier(),seen=new Map();open.push(start);seen.set(key(pose,0),0);let expanded=0;
 while(open.items.length&&expanded++<30000){
  const n=open.pop();if(n.g>seen.get(key(n.pose,n.gear)))continue;
  if(distance(n.pose)<2){const result=[];for(let q=n;q.parent;q=q.parent)result.push({gear:q.gear,steering:q.steering,length:q.length});return result.reverse();}
  for(const gear of gears)for(const steering of [-1,0,1]){
   let q={...n.pose},clear=true;
   const velocity=gear*Math.min(speed,gear<0?p.reverseMaxSpeedMs:speed),dt=.25;
   for(let i=0;i<4;i++){
    const heading=q.heading+steering*Math.abs(velocity)/radius*dt;
    const next={x:q.x-Math.sin(heading)*velocity*dt,y:q.y+Math.cos(heading)*velocity*dt,heading};
    if(!groundSegmentClear(v.surface,q,next)){clear=false;break;}q=next;
   }
   if(!clear)continue;
   const length=Math.abs(velocity),g=n.g+length*(gear<0?3:1)+(n.gear&&n.gear!==gear?8:0)+Math.abs(steering)*.1,k=key(q,gear);
   if(g>=(seen.get(k)??Infinity))continue;
   seen.set(k,g);open.push({pose:q,gear,steering,length,g,f:g+distance(q),parent:n});
  }
 }
 return null;
}
