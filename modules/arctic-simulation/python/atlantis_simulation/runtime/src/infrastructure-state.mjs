import fs from 'node:fs';
import {randomUUID} from 'node:crypto';
import {AIRLOCK_MODELS,AIRLOCK_CONTRACT,airlockState,commandAirlock,stepAirlock} from './airlock-component.mjs';
import {FACILITY_MODELS,FACILITY_CONTRACT,facilityState,commandFacility,stepFacility} from './facility-component.mjs';
const catalog=JSON.parse(fs.readFileSync(new URL('./infrastructure-catalog.json',import.meta.url),'utf8'));
const models=new Map(catalog.models.map(m=>[m.id,m]));
function position(value){
 if(!value||!['x','y','z'].every(k=>typeof value[k]==='number'&&Number.isFinite(value[k])))throw Error('position requires finite numeric x, y, z in local ENU metres');
 return {x:value.x,y:value.y,z:value.z};
}
function heading(value=0){if(typeof value!=='number'||!Number.isFinite(value))throw Error('headingDeg must be finite');return ((value%360)+360)%360;}
export class InfrastructureState {
 constructor(records=[]){this.entities=new Map();for(const record of records){this.place(record);if(AIRLOCK_MODELS.has(record.modelId))this.entities.get(record.id).componentState=airlockState(record.componentState);else if(FACILITY_MODELS.has(record.modelId))this.entities.get(record.id).componentState=facilityState(record.componentState);}}
 static componentContract(){return {...structuredClone(AIRLOCK_CONTRACT),facilities:structuredClone(FACILITY_CONTRACT)};}
 static catalog(){return structuredClone(catalog);}
 place(input){
  if(!input||!models.has(input.modelId))throw Error('unknown infrastructure modelId');
  const id=input.id??randomUUID();if(typeof id!=='string'||!id.trim()||id.length>128)throw Error('invalid infrastructure id');
  if(this.entities.has(id))throw Error('duplicate infrastructure id');
  const sourceVehicleId=input.sourceVehicleId??null;
  if(sourceVehicleId!==null&&(typeof sourceVehicleId!=='string'||!sourceVehicleId.trim()))throw Error('invalid sourceVehicleId');
  const entity={id,modelId:input.modelId,position:position(input.position),headingDeg:heading(input.headingDeg),sourceVehicleId,visualOnly:true};
  if(AIRLOCK_MODELS.has(input.modelId)){entity.componentState=airlockState();entity.visualOnly=false;entity.simulationScope='door-motion-only';}
  else if(FACILITY_MODELS.has(input.modelId)){entity.componentState=facilityState();entity.visualOnly=false;entity.simulationScope='facility-door-motion-only';}
  this.entities.set(id,entity);return structuredClone(entity);
 }
 move(input){
  const entity=this.entities.get(input?.id);if(!entity)throw Error('unknown infrastructure id');
  if(entity.sourceVehicleId)throw Error('bound infrastructure follows its authoritative vehicle; move that vehicle instead');
  const next={...entity,position:position(input.position),headingDeg:heading(input.headingDeg??entity.headingDeg)};
  this.entities.set(entity.id,next);return structuredClone(next);
 }
 remove(input){const entity=this.entities.get(input?.id);if(!entity)throw Error('unknown infrastructure id');this.entities.delete(entity.id);return structuredClone(entity);}
 snapshot(){return structuredClone([...this.entities.values()]);}
 command(input){const entity=this.entities.get(input?.id);if(!entity?.componentState)throw Error('Asset has no controllable component');if(FACILITY_MODELS.has(entity.modelId))commandFacility(entity.componentState,input);else commandAirlock(entity.componentState,input);return structuredClone(entity);}
 step(seconds){for(const entity of this.entities.values())if(entity.componentState){if(FACILITY_MODELS.has(entity.modelId))stepFacility(entity.componentState,seconds);else stepAirlock(entity.componentState,seconds);}}
}
