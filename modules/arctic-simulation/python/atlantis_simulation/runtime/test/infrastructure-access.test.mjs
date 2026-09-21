import test from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {InfrastructureState} from '../src/infrastructure-state.mjs';
import {SimulationRoom} from '../src/engine.mjs';
import {interactionDecision,movementDenial,physicalSubject} from '../src/infrastructure-access.mjs';
const owner=randomUUID(),guest=randomUUID();
const policy={version:1,ownerAccountId:owner,allowedAccountIds:[],interactionRadiusM:5,bounds:{minX:-2,maxX:2,minY:-2,maxY:2,minZ:0,maxZ:4}};
function fixture(){const s=new InfrastructureState();s.place({id:'gate',modelId:'future-airlock-pedestrian',position:{x:0,y:0,z:10},accessPolicy:policy});return s;}
const player=(account=owner,position={x:0,y:-4,z:11},fresh=true)=>({kind:'player',id:randomUUID(),ownerAccountId:account,position,fresh});
test('proximity and identity are both required; a stale player is never a nearby actor',()=>{
 const e=fixture().entities.get('gate');
 assert.equal(interactionDecision(e,player(),owner).allowed,true);
 assert.equal(interactionDecision(e,player(guest),guest).allowed,false);
 assert.equal(interactionDecision(e,player(),guest).allowed,false);
 assert.equal(interactionDecision(e,player(owner,{x:0,y:-40,z:11}),owner).allowed,false);
 assert.equal(interactionDecision(e,player(owner,{x:0,y:-4,z:11},false),owner).allowed,false);
});
test('the command itself rejects a stale discovery result, a remote caller, and a raw bypass',()=>{
 const s=fixture(),request={id:'gate',action:'airlock_open_outer',expectedRevision:0},subject=player();
 assert.throws(()=>s.command(request),/physical subject/);
 assert.throws(()=>s.command(request,{subject:player(guest),accountId:guest}),/authorized/);
 subject.position.y=-40;assert.throws(()=>s.command(request,{subject,accountId:owner}),/range/);
 subject.position.y=-4;s.command(request,{subject,accountId:owner});assert.equal(s.entities.get('gate').componentState.target.outer,1);
 assert.equal(s.entities.get('gate').componentState.outer,0);s.step(2);assert.equal(s.entities.get('gate').componentState.outer,1);
});
test('revoking access blocks a previously authorized subject without trapping it inside',()=>{
 const s=fixture();s.configureAccess({id:'gate',policy:{...policy,allowedAccountIds:[guest]}});
 assert.equal(interactionDecision(s.entities.get('gate'),player(guest),guest).allowed,true);
 s.configureAccess({id:'gate',policy});assert.equal(interactionDecision(s.entities.get('gate'),player(guest),guest).allowed,false);
 assert.equal(movementDenial(s.entities.values(),guest,{x:0,y:0,z:11},{x:0,y:-5,z:11}),null);
 assert.match(movementDenial(s.entities.values(),guest,{x:0,y:-5,z:11},{x:0,y:5,z:11}),/access-denied/);
});
test('rotated protection volumes block whole crossing segments, including large simulation steps',()=>{
 const s=fixture();s.move({id:'gate',position:{x:20,y:30,z:10},headingDeg:90});
 assert.match(movementDenial(s.entities.values(),guest,{x:-100,y:30,z:11},{x:100,y:30,z:11}),/access-denied/);
 assert.equal(movementDenial(s.entities.values(),owner,{x:-100,y:30,z:11},{x:100,y:30,z:11}),null);
 assert.equal(movementDenial(s.entities.values(),guest,{x:-100,y:30,z:30},{x:100,y:30,z:30}),null);
});
test('room vehicle integration enforces access and policy survives restart without publishing allowlists',()=>{
 const room=new SimulationRoom('access-test'),id=randomUUID();
 room.infrastructure.place({id:'gate',modelId:'future-airlock-pedestrian',position:{x:0,y:4,z:10},accessPolicy:policy});
 const origin={lat:room.config.origin.latitude,lon:room.config.origin.longitude};
 room.groundControls.attach({id,ownerAccountId:guest,terrainAssetId:'test-car',definitionId:'patria-amv',pose:{x:0,y:0,headingRad:0},surface:{origin,minX:-20,minY:-20,stepM:2,rows:21,cols:21,heights:Array(441).fill(10)}});
 const lease=room.groundControls.claim({id,ownerAccountId:guest,actor:'test'});room.groundControls.drive({id,actor:'test',leaseId:lease.id,sequence:1,throttle:1,steering:0,brake:0,durationMs:2000});
 for(let i=0;i<60;i++)room.step(1/30);
 const v=room.groundControls.observe(id);assert.ok(v.position.y<2);assert.match(v.controlStatus,/access-denied/);
 assert.equal(physicalSubject(room,'vehicle',id).ownerAccountId,guest);
 const publicEntity=room.snapshot().infrastructure[0];assert.ok(publicEntity.accessControl.protected);assert.equal(publicEntity.accessPolicy,undefined);
 const restored=new SimulationRoom(room.id,{},room.exportState());assert.deepEqual(restored.infrastructure.entities.get('gate').accessPolicy.allowedAccountIds,[owner]);
});

test('player walking is stopped at unauthorized infrastructure by the actual room tick',()=>{
 const room=new SimulationRoom('walking-access'),id=randomUUID();
 room.infrastructure.place({id:'gate',modelId:'future-airlock-pedestrian',position:{x:0,y:4,z:10},accessPolicy:policy});
 room.playerPresence.binding=()=>({zone:{minX:-20,maxX:20,minY:-20,maxY:20,floorZ:11}});
 room.playerPresence.players.set(id,{id,ownerAccountId:guest,position:{x:0,y:1.5,z:11},revision:0,lastSeenAt:Date.now(),lease:{expiresAt:Date.now()+10000},input:{east:0,north:1,remaining:1,expiresAt:Date.now()+1000}});
 room.step(.5);
 const p=room.playerPresence.observe(id);assert.equal(p.position.y,1.5);assert.equal(p.controlError.code,'access-denied');
});

test('an aircraft reaching its task destination cannot mark arrival through a denied crossing',()=>{
 const room=new SimulationRoom('arrival-access'),id=randomUUID();
 const origin={lat:room.config.origin.latitude,lon:room.config.origin.longitude};
 room.groundControls.attach({id,ownerAccountId:guest,terrainAssetId:'arrival-test',definitionId:'black-hornet',sourcePose:{z:10},pose:{x:0,y:0,headingRad:0},surface:{origin,minX:-20,minY:-20,stepM:2,rows:21,cols:21,heights:Array(441).fill(0)}});
 const v=room.groundControls.get(id);v.airborne=true;v.position.z=10;v.missionTerrainReady=true;v.rotorRpm=900;
 v.mission={id:randomUUID(),status:'running',type:'fly_to',target:{x:0,y:1,z:10},destination:{lat:origin.lat+1/6378137*180/Math.PI,lon:origin.lon},altitudeM:10,waitForTask:true,landing:false,startDistanceM:0,travelledM:0,remainingM:1,arrivalToleranceM:1.5};
 room.groundControls.step(.1,()=> 'access-denied:gate');
 assert.equal(v.mission.status,'blocked');assert.equal(v.mission.reason,'access-denied:gate');assert.equal(v.position.y,0);
});
