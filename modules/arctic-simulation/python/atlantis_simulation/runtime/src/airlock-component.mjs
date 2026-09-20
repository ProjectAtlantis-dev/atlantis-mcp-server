/** Game component only. Not an environmental or life-safety controller. */
export const AIRLOCK_MODELS=new Set(['future-airlock-pedestrian','future-airlock-vehicle','future-decon-pedestrian','future-decon-vehicle']);
export const AIRLOCK_CONTRACT={schema:'airlock-component-v1',scope:'standalone simulated airlocks',
 actions:{airlock_open_outer:{requires:['inner position and target are closed']},airlock_open_inner:{requires:['outer position and target are closed']},airlock_close:{requires:[]}},
 commandFields:['id','action','expectedRevision'],completion:'Read position and target; acceptance is not completion.',
 limitations:['No pressure, occupancy or obstruction sensor simulation','No decontamination efficacy','Not an evacuation route controller','No fire override; protected egress must be designed independently'],
 animation:'Viewer projects server outer/inner positions; no viewer-owned timer',motionSeconds:2};
export function airlockState(saved){
 const s=saved===undefined?{schema:1,revision:0,outer:0,inner:0,target:{outer:0,inner:0}}:structuredClone(saved);
 if(s.schema!==1||!Number.isSafeInteger(s.revision)||s.revision<0)throw Error('Invalid airlock schema/revision');
 for(const side of ['outer','inner'])if(!Number.isFinite(s[side])||s[side]<0||s[side]>1||![0,1].includes(s.target?.[side]))throw Error('Invalid airlock position/target');
 if((s.outer>0||s.target.outer>0)&&(s.inner>0||s.target.inner>0))throw Error('Invalid conflicting airlock state');
 return s;
}
export function commandAirlock(s,{action,expectedRevision}){
 if(!Number.isSafeInteger(expectedRevision)||expectedRevision!==s.revision)throw Error('Stale airlock revision; read state before commanding');
 if(!Object.hasOwn(AIRLOCK_CONTRACT.actions,action))throw Error('Unknown airlock action');
 const side=action==='airlock_open_outer'?'outer':action==='airlock_open_inner'?'inner':null;
 if(side){const other=side==='outer'?'inner':'outer';if(s[other]>0||s.target[other]>0)throw Error(`Close ${other} door and wait for completion first`);}
 s.target={outer:0,inner:0};if(side)s.target[side]=1;s.revision++;
}
export function stepAirlock(s,seconds){
 if(!Number.isFinite(seconds)||seconds<0)throw Error('Invalid airlock elapsed time');
 if(s.paused)return;
 for(const side of ['outer','inner']){const delta=s.target[side]-s[side];s[side]+=Math.sign(delta)*Math.min(Math.abs(delta),seconds/2);}
}
