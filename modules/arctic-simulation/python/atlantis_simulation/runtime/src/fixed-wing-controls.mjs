import {blockMission} from './drive-mission.mjs';
import {arriveMission} from './return-mission.mjs';
import {surfaceHeight} from './ground-surface.mjs';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const angle=x=>Math.atan2(Math.sin(x),Math.cos(x));
export function fixedWingProfile(definition){
 const p=definition?.flight;
 for(const key of ['maxSpeedMs','stallSpeedMs','takeoffSpeedMs','accelMs2','climbRateMs','descendRateMs','yawRateRad','rollRateRad'])if(!Number.isFinite(p?.[key])||p[key]<=0)throw Error(`fixed-wing catalog requires ${key}`);
 if(p.stallSpeedMs>=p.takeoffSpeedMs||p.takeoffSpeedMs>=p.maxSpeedMs)throw Error('fixed-wing speed envelope invalid');
 // Cruise and bank are explicit autopilot tuning, not manufacturer claims.
 return {...p,kind:'fixed-wing',maxSpeedMps:p.maxSpeedMs,cruiseMps:Math.min(p.maxSpeedMs,p.stallSpeedMs*1.5),maxBankRad:Math.PI/4};
}
function clearance(s,a,b,groundRoll=false){
 const n=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)/2));
 for(let i=0;i<=n;i++){
  const t=i/n,x=a.x+(b.x-a.x)*t,y=a.y+(b.y-a.y)*t,z=a.z+(b.z-a.z)*t,h=surfaceHeight(s,x,y);
  if(!Number.isFinite(h))return 'fixed-wing-terrain-coverage';
  if(!groundRoll&&z<h+2)return 'fixed-wing-terrain-clearance';
  if(groundRoll&&(Math.abs(h-a.z)>2||(s.water&&s.water[Math.round((y-s.minY)/s.stepM)*s.cols+Math.round((x-s.minX)/s.stepM)]!==false)))return 'takeoff-roll-not-clear-level-land';
  if((s.obstacles??[]).some(o=>x>=o.minX&&x<=o.maxX&&y>=o.minY&&y<=o.maxY&&z<o.maxZ+5))return 'fixed-wing-building-clearance';
 }
 return null;
}
export function stepFixedWing(v,dt,now){
 const m=v.mission,p=v.flightProfile;
 const active=m&&['queued','running'].includes(m.status)&&v.missionTerrainReady;
 const holding=m&&['completed','awaiting_task'].includes(m.status)&&v.missionTerrainReady&&v.airborne;
 if(!active&&!holding){v.speedMps=0;v.flightPhase=v.airborne?'simulation-paused':'parked';v.controlStatus=m?`mission-${m.status}`:v.flightPhase;v.revision++;return;}
 const h=surfaceHeight(v.surface,v.position.x,v.position.y);
 if(!Number.isFinite(h)){blockMission(v,'fixed-wing-terrain-coverage',now);v.speedMps=0;return;}
 if(!v.airborne){
  m.groundOffsetM??=Math.max(0,v.position.z-h);
  if(!m.takeoffValidated){
   const runwayM=p.takeoffSpeedMs**2/(2*p.accelMs2)+20;
   const end={x:v.position.x-Math.sin(v.headingRad)*runwayM,y:v.position.y+Math.cos(v.headingRad)*runwayM,z:h};
   const reason=clearance(v.surface,{...v.position,z:h},end,true);
   if(reason){blockMission(v,reason+'; reposition on a clear runway and set takeoff_heading_deg',now);v.speedMps=0;return;}
   m.takeoffValidated=true;
  }
  const speed=Math.min(p.takeoffSpeedMs,v.speedMps+p.accelMs2*dt);
  const next={x:v.position.x-Math.sin(v.headingRad)*speed*dt,y:v.position.y+Math.cos(v.headingRad)*speed*dt,z:h+m.groundOffsetM};
  const reason=clearance(v.surface,{...v.position,z:h},{...next,z:h},true);
  if(reason){blockMission(v,reason,now);v.speedMps=0;return;}
  v.position=next;v.speedMps=speed;v.distanceM+=speed*dt;
  v.airborne=speed>=p.takeoffSpeedMs;v.flightPhase='takeoff-roll';m.status='running';m.updatedAt=now;v.controlStatus='mission-running';v.revision++;return;
 }
 const speed=v.speedMps+clamp(p.cruiseMps-v.speedMps,-p.accelMs2*dt,p.accelMs2*dt);
 const radius=speed**2/(9.81*Math.tan(p.maxBankRad));
 let desired;
 if(holding){
  // Capture a tangent orbit at actual arrival; completion never freezes a flying aircraft.
  v.loiter??={x:v.position.x+Math.cos(v.headingRad)*radius,y:v.position.y+Math.sin(v.headingRad)*radius};
  const dx=v.loiter.x-v.position.x,dy=v.loiter.y-v.position.y,d=Math.hypot(dx,dy);
  desired=Math.atan2(-dx,dy)+Math.PI/2+Math.atan((d-radius)/radius);
 }else{delete v.loiter;desired=Math.atan2(-(m.target.x-v.position.x),m.target.y-v.position.y);}
 const error=angle(desired-v.headingRad),targetRoll=clamp(Math.atan(error*speed/9.81),-p.maxBankRad,p.maxBankRad);
 v.rollRad+=clamp(targetRoll-v.rollRad,-p.rollRateRad*dt,p.rollRateRad*dt);
 const yaw=clamp(9.81*Math.tan(v.rollRad)/Math.max(speed,p.stallSpeedMs),-p.yawRateRad,p.yawRateRad);
 const heading=v.headingRad+yaw*dt;
 const terrainCeiling=Math.max(v.surface.heights.reduce((h,z)=>Number.isFinite(z)?Math.max(h,z):h,-Infinity),...(v.surface.obstacles??[]).map(o=>o.maxZ));
 const targetZ=Math.max(m.altitudeM,terrainCeiling+30);
 const vertical=clamp(targetZ-v.position.z,-p.descendRateMs*dt,p.climbRateMs*dt);
 const next={x:v.position.x-Math.sin(heading)*speed*dt,y:v.position.y+Math.cos(heading)*speed*dt,z:v.position.z+vertical};
 // During initial rotation, grow clearance from the runway rather than requiring an instantaneous height jump.
 const reason=v.position.z<h+2&&vertical>0?clearance(v.surface,{...v.position,z:h+2},{...next,z:Math.max(next.z,h+2)}):clearance(v.surface,v.position,next);
 if(reason){if(holding&&m.status==='completed')m.status='running';blockMission(v,reason,now);v.speedMps=0;v.flightPhase='simulation-paused';return;}
 const old=v.position;v.position=next;v.headingRad=heading;v.speedMps=speed;v.distanceM+=speed*dt;
 v.pitchRad=Math.atan2(vertical/dt,speed);v.flightPhase=holding?'loiter':'cruise';
 if(active){
  m.status='running';m.updatedAt=now;m.remainingM=Math.hypot(m.target.x-next.x,m.target.y-next.y);m.travelledM=v.distanceM-m.startDistanceM;
  const dx=next.x-old.x,dy=next.y-old.y,l=dx*dx+dy*dy,t=l?clamp(((m.target.x-old.x)*dx+(m.target.y-old.y)*dy)/l,0,1):0;
  const miss=Math.hypot(old.x+t*dx-m.target.x,old.y+t*dy-m.target.y);
  if(miss<=m.arrivalToleranceM&&next.z>=m.altitudeM-2)arriveMission(m,'waypoint-reached-loitering',now);
 }
 v.controlStatus=`mission-${m.status}`;v.revision++;
}
