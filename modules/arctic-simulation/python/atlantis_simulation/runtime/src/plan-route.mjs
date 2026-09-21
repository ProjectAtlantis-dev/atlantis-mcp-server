import {planTerrainRoute} from './terrain-route.mjs';
// Worker process: expensive geographic planning never stalls the simulation tick.
let input='';for await(const chunk of process.stdin)input+=chunk;
const {surface,start,target}=JSON.parse(input);
const route=planTerrainRoute(surface,start,target,{dense:true});
if(!route.complete)throw Error('complete destination route required');
process.stdout.write(JSON.stringify({...route,stepM:surface.stepM}));
