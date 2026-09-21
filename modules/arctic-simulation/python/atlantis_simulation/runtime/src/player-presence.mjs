import {randomUUID} from 'node:crypto';
import fs from 'node:fs';

export function playerPolicy(world,required=true){
  const path=process.env.ATLANTIS_PLAYER_DEPLOYMENTS;
  if(!path)throw Error('ATLANTIS_PLAYER_DEPLOYMENTS required');
  const config=JSON.parse(fs.readFileSync(path,'utf8'));
  if(config.version!==1||!config.worlds)throw Error('Invalid player configuration');
  if(!config.worlds[world]&&required)throw Error('Unknown player world configuration');
  return config.worlds[world]??{};
}
export function protectedEntrance(world,id){
  if(!process.env.ATLANTIS_PLAYER_DEPLOYMENTS)return false;
  return !!playerPolicy(world,false).airlocks?.[id];
}
const finite=(v)=>typeof v==='number'&&Number.isFinite(v);
const inside=(p,z)=>p.x>=z.minX&&p.x<=z.maxX&&p.y>=z.minY&&p.y<=z.maxY&&p.z===z.floorZ;

/** Authoritative walking on explicitly commissioned planar walkable zones. */
export class PlayerPresence {
  constructor(world,saved=[],now=()=>Date.now()){
    this.world=world;this.now=now;this.players=new Map(saved.map(p=>[p.id,{...structuredClone(p),lease:null,input:null,lastSeenAt:0,
      entryAction:p.entryAction?.status==='running'?{...p.entryAction,status:'interrupted',reason:'restart requires reconciliation'}:p.entryAction}]));
  }
  binding(id){
    const policy=playerPolicy(this.world),binding=policy.players?.[id],zone=policy.zones?.[binding?.zoneId];
    if(!binding||!zone)throw Error('Player has no commissioned walking zone');
    if(!['minX','maxX','minY','maxY','floorZ'].every(k=>finite(zone[k]))||zone.minX>=zone.maxX||zone.minY>=zone.maxY)throw Error('Invalid walking zone');
    return {binding,zone};
  }
  attach({id,ownerAccountId}){
    const {binding,zone}=this.binding(id);
    if(!this.players.has(id)){
      if(!binding.spawn||!['x','y','z'].every(k=>finite(binding.spawn[k]))||!inside(binding.spawn,zone))throw Error('Spawn outside walking zone');
      this.players.set(id,{id,position:{...binding.spawn},zoneId:binding.zoneId,revision:0,lease:null,input:null,lastSeenAt:0});
    }
    const p=this.get(id);
    if(ownerAccountId!==undefined){if(p.ownerAccountId&&p.ownerAccountId!==ownerAccountId)throw Error('Player account binding conflict');p.ownerAccountId=ownerAccountId;}
    return this.observe(id);
  }
  get(id){const p=this.players.get(id);if(!p)throw Error('Attach player first');return p;}
  claim({id,actor}){
    this.binding(id);const p=this.get(id);
    if(p.lease&&p.lease.expiresAt>this.now()&&p.lease.actor!==actor)throw Error('Player controlled by another session');
    if(!p.lease||p.lease.expiresAt<=this.now())p.lease={id:randomUUID(),actor,sequence:0,expiresAt:this.now()+15000};
    p.controlError=null;p.lastSeenAt=this.now();return {...p.lease};
  }
  lease(p,{actor,leaseId}){
    if(!p.lease||p.lease.id!==leaseId||p.lease.actor!==actor||p.lease.expiresAt<=this.now())throw Error('Player control lease expired or invalid');
  }
  move(args){
    const p=this.get(args.id);this.binding(args.id);this.lease(p,args);
    const {east,north,durationMs,sequence}=args;
    if(!finite(east)||!finite(north)||Math.hypot(east,north)>1.000001||!finite(durationMs)||durationMs<50||durationMs>1000)throw Error('Invalid bounded walking input');
    if(!Number.isSafeInteger(sequence)||sequence<=p.lease.sequence)throw Error('Walking sequence must increase');
    p.lease.sequence=sequence;p.lease.expiresAt=this.now()+15000;p.lastSeenAt=this.now();
    p.input={east,north,remaining:durationMs/1000,expiresAt:this.now()+durationMs};
    return {accepted:true,sequence};
  }
  release(args){const p=this.get(args.id);this.lease(p,args);p.lease=null;p.input=null;p.lastSeenAt=0;return {released:true};}
  step(dt,movementGuard=()=>null){
    for(const p of this.players.values()){
      if(!p.input)continue;
      if(!p.lease||p.lease.expiresAt<=this.now()||p.input.expiresAt<=this.now()){p.input=null;continue;}
      let zone;
      try { zone=this.binding(p.id).zone; }
      catch(error){
        p.input=null;p.lease=null;p.lastSeenAt=0;p.revision++;
        p.controlError={code:'walking-zone-unavailable',message:error.message};
        console.error('Player walking configuration failed', {playerId:p.id,error});
        continue;
      }
      const elapsed=Math.min(dt,p.input.remaining),speed=2;
      const next={x:p.position.x+p.input.east*speed*elapsed,y:p.position.y+p.input.north*speed*elapsed,z:p.position.z};
      const denied=movementGuard(p,p.position,next);
      if(denied){p.input=null;p.controlError={code:'access-denied',message:denied};p.revision++;continue;}
      if(inside(next,zone)){p.position=next;p.revision++;}
      else p.input.remaining=0;
      p.input.remaining-=elapsed;if(p.input.remaining<=0)p.input=null;
    }
  }
  observe(id){const p=this.get(id);return {id:p.id,ownerAccountId:p.ownerAccountId,position:{...p.position},zoneId:p.zoneId,revision:p.revision,
    controlError:p.controlError??null,lastSeenAt:p.lastSeenAt,fresh:!!p.lease&&p.lease.expiresAt>this.now()&&this.now()-p.lastSeenAt<3000,
    moving:!!p.input&&p.input.expiresAt>this.now()&&Math.hypot(p.input.east,p.input.north)>0};}
  authorizeEntrance(id,entrance,entity){
    const {binding,zone}=this.binding(id);const p=this.observe(id);
    if(binding.zoneId!==p.zoneId||!inside(p.position,zone))throw Error('Player requires zone reconciliation');
    if(!entrance.allowedPlayerIds?.includes(id))throw Error('Player not authorized for entrance');
    if(!p.fresh)throw Error('Player position presence is stale');
    const offset=entrance.outerOffset,radius=entrance.radiusM;
    if(!offset||!['x','y','z'].every(k=>finite(offset[k]))||!finite(radius)||radius<=0||radius>3)throw Error('Invalid entrance geometry');
    // Offset is world ENU at heading zero; heading is clockwise from north.
    const angle=-entity.headingDeg*Math.PI/180;
    const target={x:entity.position.x+offset.x*Math.cos(angle)-offset.y*Math.sin(angle),
      y:entity.position.y+offset.x*Math.sin(angle)+offset.y*Math.cos(angle),z:entity.position.z+offset.z};
    if(Math.hypot(...['x','y','z'].map(k=>p.position[k]-target[k]))>radius)throw Error('Player is not near this entrance');
  }
  requestEntry(args,infrastructure){
    const p=this.get(args.id);this.lease(p,args);
    const entrance=playerPolicy(this.world).airlocks?.[args.airlockId],entity=infrastructure.entities.get(args.airlockId);
    if(!entrance||!entity||entity.sourceVehicleId)throw Error('Configured stationary airlock required');
    this.authorizeEntrance(args.id,entrance,entity);
    if(p.entryAction?.status==='running')throw Error('Entry action already running');
    infrastructure.command({id:args.airlockId,action:'airlock_open_outer',expectedRevision:args.expectedRevision},{accountId:p.ownerAccountId,subject:{kind:'player',...this.observe(p.id)}});
    entity.componentState.paused=false;
    p.entryAction={id:randomUUID(),airlockId:args.airlockId,status:'running',target:1,revision:entity.componentState.revision};
    return structuredClone(p.entryAction);
  }
  reconcileActions(infrastructure){
    for(const p of this.players.values()){
      const a=p.entryAction;if(a?.status!=='running')continue;
      const s=infrastructure.entities.get(a.airlockId)?.componentState;
      if(!s||s.revision!==a.revision||s.target.outer!==1){a.status='failed';a.reason='mechanism changed';}
      else if(s.outer===1)a.status='succeeded';
    }
  }
  action(id){return structuredClone(this.get(id).entryAction??null);}
  snapshot(){return [...this.players.keys()].map(id=>this.observe(id));}
  exportState(){return structuredClone([...this.players.values()]);}
}
