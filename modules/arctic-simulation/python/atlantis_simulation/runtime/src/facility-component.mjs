import {airlockState,commandAirlock,stepAirlock} from './airlock-component.mjs';
export const FACILITY_MODELS=new Set(['future-city-water-plant','future-city-wastewater','future-city-backup-power','future-city-logistics','future-city-clinic','future-city-fire-rescue','future-city-solid-waste']);
export const FACILITY_CONTRACT={schema:'facility-access-v1',scope:'standalone service facility entry and freight doors',
 actions:{facility_entry_outer:{requires:['inner door closed','freight doors closed']},facility_entry_inner:{requires:['outer door closed','freight doors closed']},facility_entry_close:{requires:[]},facility_freight_open:{requires:['entry doors and targets closed']},facility_freight_close:{requires:[]}},
 completion:'Read componentState entry positions or freight position; command acceptance is not completion.',
 limitations:['Game access interlocks only; no pressure or obstacle sensors','No water, energy, treatment, medical or fire simulation','Nested showcase cities are not automatically expanded into authoritative entities','Freight opening bypasses the environmental envelope; no safe environmental release is implied']};
export function facilityState(saved){
 const s=saved===undefined?{schema:'facility-access-v1',revision:0,entry:airlockState(),freight:0,freightTarget:0}:structuredClone(saved);
 if(s.schema!=='facility-access-v1'||!Number.isSafeInteger(s.revision)||s.revision<0)throw Error('Invalid facility revision/schema');
 s.entry=airlockState(s.entry);
 if(!Number.isFinite(s.freight)||s.freight<0||s.freight>1||![0,1].includes(s.freightTarget))throw Error('Invalid freight state');
 if((s.freight>0||s.freightTarget>0)&&[s.entry.outer,s.entry.inner,s.entry.target.outer,s.entry.target.inner].some(n=>n>0))throw Error('Conflicting facility entry and freight state');
 return s;
}
export function commandFacility(s,{action,expectedRevision}){
 if(!Number.isSafeInteger(expectedRevision)||expectedRevision!==s.revision)throw Error('Stale facility revision; read state before commanding');
 if(!Object.hasOwn(FACILITY_CONTRACT.actions,action))throw Error('Unknown facility action');
 if(action==='facility_freight_open'){
  if([s.entry.outer,s.entry.inner,s.entry.target.outer,s.entry.target.inner].some(n=>n>0))throw Error('Close entry doors and wait for completion first');
  s.freightTarget=1;
 }else if(action==='facility_freight_close')s.freightTarget=0;
 else{
  if(action!=='facility_entry_close'&&(s.freight>0||s.freightTarget>0))throw Error('Close freight doors and wait for completion first');
  const a={facility_entry_outer:'airlock_open_outer',facility_entry_inner:'airlock_open_inner',facility_entry_close:'airlock_close'}[action];
  commandAirlock(s.entry,{action:a,expectedRevision:s.entry.revision});
 }
 s.revision++;
}
export function stepFacility(s,seconds){
 if(!Number.isFinite(seconds)||seconds<0)throw Error('Invalid facility elapsed time');
 stepAirlock(s.entry,seconds);const d=s.freightTarget-s.freight;s.freight+=Math.sign(d)*Math.min(Math.abs(d),seconds/2);
}
