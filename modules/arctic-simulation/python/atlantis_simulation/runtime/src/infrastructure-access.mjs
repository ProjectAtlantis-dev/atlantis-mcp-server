const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export function accessPolicy(value){
 if(!value||value.version!==1||!uuid.test(value.ownerAccountId)||!Array.isArray(value.allowedAccountIds)||value.allowedAccountIds.some(id=>!uuid.test(id)))throw Error('Invalid infrastructure access identities');
 if(!Number.isFinite(value.interactionRadiusM)||value.interactionRadiusM<=0||value.interactionRadiusM>100)throw Error('Interaction radius must be greater than zero and at most 100m');
 const b=value.bounds;
 if(!b||!['minX','maxX','minY','maxY','minZ','maxZ'].every(k=>Number.isFinite(b[k]))||b.minX>=b.maxX||b.minY>=b.maxY||b.minZ>=b.maxZ)throw Error('Explicit nonempty protection volume required');
 return structuredClone({...value,allowedAccountIds:[...new Set([value.ownerAccountId,...value.allowedAccountIds])]});
}
export function localPoint(entity,position){
 const a=entity.headingDeg*Math.PI/180,dx=position.x-entity.position.x,dy=position.y-entity.position.y;
 return {x:dx*Math.cos(a)-dy*Math.sin(a),y:dx*Math.sin(a)+dy*Math.cos(a),z:position.z-entity.position.z};
}
export function distanceToVolume(entity,position){
 const p=localPoint(entity,position),b=entity.accessPolicy.bounds;
 return Math.hypot(Math.max(b.minX-p.x,0,p.x-b.maxX),Math.max(b.minY-p.y,0,p.y-b.maxY),Math.max(b.minZ-p.z,0,p.z-b.maxZ));
}
export function accountAllowed(entity,accountId){return !!entity.accessPolicy?.allowedAccountIds.includes(accountId);}
export function interactionDecision(entity,subject,accountId){
 if(!entity?.accessPolicy)return {allowed:false,reason:'Access policy has not been commissioned'};
 if(!subject||subject.ownerAccountId!==accountId)return {allowed:false,reason:'Authenticated physical subject required'};
 if(!accountAllowed(entity,accountId))return {allowed:false,reason:'Not authorized for this structure'};
 if(subject.kind==='player'&&!subject.fresh)return {allowed:false,reason:'Player presence is stale'};
 if(!subject.position||!['x','y','z'].every(k=>Number.isFinite(subject.position[k])))throw Error('Physical subject has no valid authoritative position');
 const distanceM=distanceToVolume(entity,subject.position);
 return {allowed:distanceM<=entity.accessPolicy.interactionRadiusM,distanceM,radiusM:entity.accessPolicy.interactionRadiusM,
  reason:distanceM<=entity.accessPolicy.interactionRadiusM?null:'Physical subject is outside interaction range'};
}
function segmentHitsBox(a,b,box){
 let lo=0,hi=1;
 for(const [axis,min,max] of [['x','minX','maxX'],['y','minY','maxY'],['z','minZ','maxZ']]){
  const d=b[axis]-a[axis];
  if(Math.abs(d)<1e-12){if(a[axis]<box[min]||a[axis]>box[max])return false;continue;}
  let t0=(box[min]-a[axis])/d,t1=(box[max]-a[axis])/d;if(t0>t1)[t0,t1]=[t1,t0];lo=Math.max(lo,t0);hi=Math.min(hi,t1);if(lo>hi)return false;
 }
 return true;
}
function depth(p,b){return Math.min(p.x-b.minX,b.maxX-p.x,p.y-b.minY,b.maxY-p.y,p.z-b.minZ,b.maxZ-p.z);}
export function movementDenial(entities,accountId,from,to){
 if(from.x===to.x&&from.y===to.y&&from.z===to.z)return null;
 for(const entity of entities){
  if(!entity.accessPolicy||accountAllowed(entity,accountId))continue;
  const a=localPoint(entity,from),b=localPoint(entity,to),box=entity.accessPolicy.bounds;
  // Revocation never traps a subject already inside: allow movement toward an exit.
  if(depth(a,box)>=0&&(depth(b,box)<depth(a,box)||distanceToVolume(entity,to)>0))continue;
  if(segmentHitsBox(a,b,box))return `access-denied:${entity.id}`;
 }
 return null;
}
export function vehicleWorldPosition(vehicle,position,origin){
 const frame=vehicle.surface.origin,lat=frame.lat+position.y/6378137*180/Math.PI;
 const lon=frame.lon+position.x/(6378137*Math.cos(frame.lat*Math.PI/180))*180/Math.PI;
 return {x:(lon-origin.longitude)*Math.PI/180*6378137*Math.cos(origin.latitude*Math.PI/180),y:(lat-origin.latitude)*Math.PI/180*6378137,z:position.z-origin.altitudeM};
}
export function physicalSubject(room,kind,id){
 if(kind==='vehicle'){
  const v=room.groundControls.get(id);
  return {kind,id,ownerAccountId:v.ownerAccountId,position:vehicleWorldPosition(v,v.position,room.config.origin)};
 }
 if(kind==='player')return {kind,...room.playerPresence.observe(id)};
 throw Error('Subject must be a physical player or vehicle');
}
