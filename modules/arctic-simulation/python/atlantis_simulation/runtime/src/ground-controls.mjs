import {queueCompletedReturn} from './return-mission.mjs';
import {FLIGHT_PROFILES,attachFlight,startFlight,stepFlight} from './flight-controls.mjs';
import {planTerrainRoute} from './terrain-route.mjs';
import {randomUUID} from 'node:crypto';
import {terminalMission,missionView,startMission,changeMission,blockMission,missionInput} from './drive-mission.mjs';

const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
function number(value,name,min=-Infinity,max=Infinity){
  if(typeof value!=='number'||!Number.isFinite(value)||value<min||value>max)throw Error(`invalid ${name}`);
  return value;
}
function text(value,name){if(typeof value!=='string'||!value.trim())throw Error(`missing ${name}`);return value;}
function surfaceHeight(surface,x,y){
  const gx=(x-surface.minX)/surface.stepM,gy=(y-surface.minY)/surface.stepM;
  if(gx<0||gy<0||gx>surface.cols-1||gy>surface.rows-1)return null;
  const ix=Math.min(surface.cols-2,Math.floor(gx)),iy=Math.min(surface.rows-2,Math.floor(gy));
  const tx=gx-ix,ty=gy-iy,h=(xx,yy)=>surface.heights[yy*surface.cols+xx];
  return (h(ix,iy)*(1-tx)+h(ix+1,iy)*tx)*(1-ty)+(h(ix,iy+1)*(1-tx)+h(ix+1,iy+1)*tx)*ty;
}
function validateSurface(s){
  if(!s||typeof s!=='object')throw Error('server elevation grid required');
  for(const key of ['minX','minY'])number(s[key],key);
  number(s.stepM,'stepM',.1,1000);
  for(const key of ['rows','cols'])if(!Number.isInteger(s[key])||s[key]<2||s[key]>1024)throw Error(`invalid ${key}`);
  if(!Array.isArray(s.heights)||s.heights.length!==s.rows*s.cols)throw Error('invalid elevation grid');
  s.heights.forEach(h=>number(h,'height'));
  number(s.origin?.lat,'origin.lat',-85,85);number(s.origin?.lon,'origin.lon',-180,180);
}
function surfaceNormal(s,p){
  const d=Math.min(1,s.stepM/2),x0=Math.max(s.minX,p.x-d),x1=Math.min(s.minX+(s.cols-1)*s.stepM,p.x+d);
  const y0=Math.max(s.minY,p.y-d),y1=Math.min(s.minY+(s.rows-1)*s.stepM,p.y+d);
  const x=-(surfaceHeight(s,x1,p.y)-surfaceHeight(s,x0,p.y))/(x1-x0);
  const y=-(surfaceHeight(s,p.x,y1)-surfaceHeight(s,p.x,y0))/(y1-y0);
  const length=Math.hypot(x,y,1);return {x:x/length,y:y/length,z:1/length};
}

/** Ground-only controller. All mutation is called by the authenticated server. */
export class GroundControls {
  constructor(saved=[],now=()=>Date.now()){
    this.now=now;this.vehicles=new Map();
    for(const item of saved){
      const v=structuredClone(item);validateSurface(v.surface);
      // A restart never revives a stale controller or held throttle.
      // Restart pauses the simulation at its durable pose. Never replay held
      // input or let residual velocity drift the vehicle before a fresh claim.
      v.signedDistanceM??=v.distanceM;
      v.lease=null;v.input=null;v.speedMps=0;v.controlStatus='restart-paused';
      if(v.mission&&['queued','running'].includes(v.mission.status)){v.mission.status='paused';v.mission.reason='server-restarted';}
      v.missionTerrainReady=false;
      this.vehicles.set(v.id,v);
    }
  }
  attach(input){
    const {id,terrainAssetId,definitionId,ownerAccountId,pose,surface}=input;
    if(!uuid.test(id)||!uuid.test(ownerAccountId))throw Error('canonical bank UUIDs required');
    text(terrainAssetId,'terrainAssetId');
    if(definitionId!=='patria-amv'&&!FLIGHT_PROFILES[definitionId])throw Error('controller unavailable for this model');
    validateSurface(surface);
    number(pose?.x,'pose.x');number(pose?.y,'pose.y');number(pose?.headingRad,'pose.headingRad');
    if(FLIGHT_PROFILES[definitionId])return attachFlight(this,input,surfaceHeight);
    const z=surfaceHeight(surface,pose.x,pose.y);
    if(z===null)throw Error('initial pose outside authoritative elevation grid');
    const current=this.vehicles.get(id);
    if(current){
      if(current.terrainAssetId!==terrainAssetId||current.definitionId!==definitionId||current.ownerAccountId!==ownerAccountId)throw Error('vehicle binding conflict');
      return this.observe(id);
    }
    if([...this.vehicles.values()].some(v=>v.terrainAssetId===terrainAssetId))throw Error('terrain instance already attached');
    this.vehicles.set(id,{id,terrainAssetId,definitionId,ownerAccountId,surface:structuredClone(surface),
      position:{x:pose.x,y:pose.y,z},headingRad:pose.headingRad,speedMps:0,distanceM:0,signedDistanceM:0,
      lease:null,input:null,revision:0,controlStatus:'stopped'});
    return this.observe(id);
  }
  get(id){const v=this.vehicles.get(id);if(!v)throw Error('controlled vehicle not found');return v;}
  fly_to(payload){return startFlight(this.get(payload.id),payload,this.now());}
  drive_to(payload){if(this.get(payload.id).flightProfile)throw Error('aircraft requires fly_to');return startMission(this.get(payload.id),payload,this.now());}
  mission_control(payload){const v=this.get(payload.id),result=changeMission(v,payload,this.now());queueCompletedReturn(v,this.now());return result;}
  mission_status({id}){const v=this.get(id);return {mission:missionView(v),history:structuredClone(v.missionHistory??[])};}
  capabilities({id}){const v=this.get(id);if(v.flightProfile)return {actions:[{id:'fly_to',label:'Fly here (60 m above terrain)',destination:'latlon'}],missionActions:['pause','resume','cancel','complete_task'],returnDestination:true,waitForTask:true,limits:{maxDistanceM:2000,maxSpeedMps:v.flightProfile.maxSpeedMps},controller:'simulated-vtol-v1'};return {actions:[{id:'drive_to',label:'Drive here',destination:'latlon'}],missionActions:['pause','resume','cancel','complete_task'],returnDestination:true,waitForTask:true,limits:{maxDistanceM:20000,maxSpeedMps:4},routing:'surveyed-road preference with local offroad obstacle detours; bounded by verified terrain'};}
  mission_surface({id,missionId,surface,error}){
    const v=this.get(id);if(!v.mission||v.mission.id!==missionId)throw Error('stale terrain update');
    if(!['queued','running'].includes(v.mission.status))return missionView(v);
    if(error){blockMission(v,error,this.now());return missionView(v);}
    validateSurface(surface);
    if(surface.origin.lat!==v.surface.origin.lat||surface.origin.lon!==v.surface.origin.lon)throw Error('terrain coordinate frame changed');
    const height=surfaceHeight(surface,v.position.x,v.position.y);
    if(height===null||(!v.flightProfile&&Math.abs(height-v.position.z)>2))throw Error('new terrain does not agree with current vehicle pose');
    if(v.flightProfile){v.surface=structuredClone(surface);v.missionTerrainReady=true;return missionView(v);}
    let navigation;
    try {navigation=planTerrainRoute(surface,v.position,v.mission.target);}
    catch(error){blockMission(v,'route-unavailable: '+error.message,this.now());return missionView(v);}
    v.surface=structuredClone(surface);v.mission.navigation={...navigation,index:0};
    v.missionTerrainReady=true;return missionView(v);
  }
  claim({id,actor,ownerAccountId}){
    const v=this.get(id);text(actor,'actor');
    if(v.flightProfile)throw Error('VTOL controller uses fly_to missions; manual flight control is not implemented');
    if(v.ownerAccountId!==ownerAccountId)throw Error('vehicle ownership mismatch');
    if(!terminalMission(v.mission))throw Error('cancel mission before taking manual control');
    if(v.lease&&v.lease.expiresAt>this.now()){
      if(v.lease.actor!==actor)throw Error('vehicle controlled by another controller');
      return structuredClone(v.lease);
    }
    v.lease={id:randomUUID(),actor,expiresAt:this.now()+15000,sequence:0};v.input=null;
    v.revision++;return structuredClone(v.lease);
  }
  drive({id,actor,leaseId,sequence,throttle,steering,brake,durationMs}){
    const v=this.get(id),lease=v.lease;
    if(!lease||lease.id!==leaseId||lease.actor!==actor||lease.expiresAt<=this.now())throw Error('control lease invalid or expired');
    if(!Number.isSafeInteger(sequence)||sequence<=lease.sequence)throw Error('control sequence must increase');
    number(throttle,'throttle',-1,1);number(steering,'steering',-1,1);number(brake,'brake',0,1);
    number(durationMs,'durationMs',50,2000);
    lease.sequence=sequence;lease.expiresAt=this.now()+15000;
    v.input={throttle,steering,brake,remaining:durationMs/1000,expiresAt:this.now()+durationMs};
    v.controlStatus='executing';v.revision++;
    return {accepted:true,sequence,revision:v.revision};
  }
  release({id,actor,leaseId}){
    const v=this.get(id);
    if(!v.lease||v.lease.actor!==actor||v.lease.id!==leaseId)throw Error('control lease mismatch');
    v.lease=null;v.input=null;v.controlStatus='released-braking';v.revision++;
    return {released:true};
  }
  step(dt){
    for(const v of this.vehicles.values()){
      if(v.flightProfile){stepFlight(v,dt,this.now(),surfaceHeight);queueCompletedReturn(v,this.now());continue;}
      if(v.lease&&v.lease.expiresAt<=this.now()){v.lease=null;v.input=null;v.controlStatus='lease-expired-braking';}
      if(v.input&&(v.input.remaining<=0||v.input.expiresAt<=this.now())){v.input=null;v.controlStatus='input-expired-braking';}
      const input=missionInput(v,this.now())??v.input??{throttle:0,steering:0,brake:1};
      // Explicit kinematic commissioning profile, not manufacturer-rated physics.
      let speed=Math.max(-6,Math.min(20,v.speedMps+input.throttle*3*dt));
      const decel=(input.brake*7+.15)*dt;
      speed=Math.sign(speed)*Math.max(0,Math.abs(speed)-decel);
      const heading=v.headingRad+speed/4.5*Math.tan(input.steering*.5)*dt;
      const x=v.position.x-Math.sin(heading)*speed*dt,y=v.position.y+Math.cos(heading)*speed*dt;
      const z=surfaceHeight(v.surface,x,y);
      let hazard=null;
      if(v.mission&&!terminalMission(v.mission)){
        if(z===null)hazard='terrain-coverage-boundary';
        else if(z<=.25)hazard='water-or-sea-level-terrain';
        else if(v.surface.water&&v.surface.water[Math.min(v.surface.rows-1,Math.max(0,Math.round((y-v.surface.minY)/v.surface.stepM)))*v.surface.cols+Math.min(v.surface.cols-1,Math.max(0,Math.round((x-v.surface.minX)/v.surface.stepM)))]!==false)hazard='water-or-unknown-surface';
        else if(Math.hypot(surfaceNormal(v.surface,{x,y}).x,surfaceNormal(v.surface,{x,y}).y)>.5)hazard='terrain-slope-exceeds-profile';
        else if((v.surface.obstacles??[]).some(o=>x>=o.minX&&x<=o.maxX&&y>=o.minY&&y<=o.maxY))hazard='building-clearance';
      }
      if(hazard){blockMission(v,hazard,this.now());v.speedMps=0;v.controlStatus='mission-blocked';}
      else if(z===null){v.speedMps=0;v.input=null;v.controlStatus='surface-boundary-stopped';}
      else {v.position={x,y,z};v.headingRad=heading;v.speedMps=speed;v.distanceM+=Math.abs(speed)*dt;v.signedDistanceM+=speed*dt;}
      if(v.input)v.input.remaining-=dt;
      if(v.mission&&!v.lease&&!v.input)v.controlStatus=`mission-${v.mission.status}`;
      queueCompletedReturn(v,this.now());
      v.revision++;
    }
  }
  observe(id){
    const v=this.get(id),s=v.surface;
    return {id:v.id,terrainAssetId:v.terrainAssetId,definitionId:v.definitionId,authority:v.flightProfile?'server-vtol-v1':'server-ground-v1',
      ...(v.flightProfile?{flightPhase:v.flightPhase,rotorRpm:v.rotorRpm,rotorAngleRad:v.rotorAngleRad,pitchRad:v.pitchRad,rollRad:v.rollRad}:{}),
      lat:s.origin.lat+v.position.y/6378137*180/Math.PI,
      lon:s.origin.lon+v.position.x/(6378137*Math.cos(s.origin.lat*Math.PI/180))*180/Math.PI,
      position:{...v.position},groundNormal:surfaceNormal(s,v.position),headingRad:v.headingRad,speedMps:v.speedMps,distanceM:v.distanceM,signedDistanceM:v.signedDistanceM??v.distanceM,
      revision:v.revision,controlStatus:v.controlStatus,controlled:!!v.lease,
      surfaceId:s.id??null,ownerAccountId:v.ownerAccountId,mission:missionView(v),
      navigationFrame:{origin:{...s.origin},minX:s.minX,minY:s.minY,maxX:s.minX+(s.cols-1)*s.stepM,maxY:s.minY+(s.rows-1)*s.stepM}};
  }
  snapshot(){return [...this.vehicles.keys()].map(id=>this.observe(id));}
  exportState(){return structuredClone([...this.vehicles.values()]);}
}
