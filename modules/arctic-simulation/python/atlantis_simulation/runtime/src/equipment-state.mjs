import fs from 'node:fs';
export const ASSET_CONTROLS=JSON.parse(fs.readFileSync(new URL('./asset-controls.json',import.meta.url),'utf8'));
export function equipmentContract(modelId){const value=ASSET_CONTROLS.models[modelId];if(!value)throw Error('Unknown equipment model');return structuredClone(value);}
function validate(value,spec){if(spec.type==='boolean'){if(typeof value!=='boolean')throw Error('Boolean control required');}else if(typeof value!=='number'||!Number.isFinite(value)||value<spec.min||value>spec.max)throw Error(`Control outside ${spec.min}..${spec.max} ${spec.unit}`);}
export function equipmentState(modelId,saved){
 const contract=equipmentContract(modelId);
 if(saved!==undefined){
  if(saved.version!==1||typeof saved.mechanisms!=='object')throw Error('Invalid equipment state version');
  const s=structuredClone(saved);
  if(Object.keys(s.mechanisms).length!==contract.mechanisms.length)throw Error('Equipment mechanism inventory changed');
  for(const m of contract.mechanisms){const item=s.mechanisms[m.id];if(!item||!Number.isSafeInteger(item.revision)||item.revision<0||!Number.isFinite(item.phase))throw Error('Invalid persisted mechanism');for(const [key,spec] of Object.entries(m.fields)){validate(item.targets[key],spec);if(!Number.isFinite(item.actual[key]))throw Error('Invalid persisted mechanism pose');}}
  return s;
 }
 return {version:1,mechanisms:Object.fromEntries(contract.mechanisms.map(m=>[m.id,{revision:0,phase:0,targets:Object.fromEntries(Object.entries(m.fields).map(([key,s])=>[key,s.default])),actual:Object.fromEntries(Object.entries(m.fields).map(([key,s])=>[key,Number(s.default)]))}]))};
}
const prefix=(a,b)=>a.every((n,i)=>b[i]===n);
export function commandEquipment(modelId,state,{mechanismId,values,expectedRevision}){
 const mechanisms=equipmentContract(modelId).mechanisms,m=mechanisms.find(m=>m.id===mechanismId),s=state.mechanisms[mechanismId];
 if(!m||!s)throw Error('Mechanism not supported by this model');
 if(expectedRevision!==s.revision||!Number.isSafeInteger(expectedRevision))throw Error('Stale mechanism revision');
 if(!values||typeof values!=='object'||Array.isArray(values)||!Object.keys(values).length||Object.keys(values).some(key=>!Object.hasOwn(m.fields,key)))throw Error('Unknown or empty mechanism parameters');
 for(const [key,value] of Object.entries(values))validate(value,m.fields[key]);
 const next={...s.targets,...values};
 const busy=(item,key)=>item.actual[key]>0||Number(item.targets[key])>0;
 if(m.kind==='airlock'){
  if((next.outer_open&&next.inner_open)||(next.outer_open&&s.actual.inner_open>0)||(next.inner_open&&s.actual.outer_open>0))throw Error('Close the other airlock door and wait first');
  if((s.actual.cycle_seconds>0||next.cycle_seconds>0)&&(next.outer_open||next.inner_open||s.actual.outer_open>0||s.actual.inner_open>0))throw Error('Close doors before processing; wait for active cycle');
  if(next.outer_open||next.inner_open)for(const f of mechanisms.filter(n=>n.kind==='freight'&&prefix(n.path,m.path)))if(busy(state.mechanisms[f.id],'open'))throw Error('Close freight doors and wait first');
 }
 if(m.kind==='freight'&&next.open)for(const a of mechanisms.filter(n=>n.kind==='airlock'&&prefix(m.path,n.path)))if(['outer_open','inner_open','cycle_seconds'].some(k=>busy(state.mechanisms[a.id],k)))throw Error('Close entry doors and wait first');
 s.targets=next;s.revision++;
 if(Object.hasOwn(values,'cycle_seconds'))s.actual.cycle_seconds=values.cycle_seconds;
 return structuredClone(s);
}
export function stepEquipment(modelId,state,seconds){
 if(!Number.isFinite(seconds)||seconds<0)throw Error('Invalid equipment elapsed time');
 for(const m of ASSET_CONTROLS.models[modelId].mechanisms){const s=state.mechanisms[m.id];
  for(const [key,spec] of Object.entries(m.fields)){
   if(key==='cycle_seconds'){s.actual[key]=Math.max(0,s.actual[key]-seconds);if(s.actual[key]===0)s.targets[key]=0;continue;}
   const target=Number(s.targets[key]),delta=target-s.actual[key];
   const rate=spec.rate??(m.kind==='airlock'||m.kind==='freight'?.5:Infinity);
   s.actual[key]+=Math.sign(delta)*Math.min(Math.abs(delta),rate*seconds);
  }
  if(m.kind==='tracks')s.phase+=s.actual.track_speed_mps*seconds;
  else if(['rotation','fans','propulsion'].includes(m.kind))s.phase=(s.phase+(s.actual.rpm??s.actual.shaft_rpm)*seconds*Math.PI/30)%(2*Math.PI);
 }
}
