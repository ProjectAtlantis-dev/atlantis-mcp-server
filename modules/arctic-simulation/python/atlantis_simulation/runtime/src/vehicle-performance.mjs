import {readFileSync} from 'node:fs';
const catalog=JSON.parse(readFileSync(new URL('./vehicle-performance.json',import.meta.url),'utf8'));
export function groundPerformance(definitionId){
 const value=catalog[definitionId];
 if(!value)throw Error(`No ground performance specification for ${definitionId}`);
 return value;
}
export function groundProfile(definitionId){
 const {published,simulation:s}=groundPerformance(definitionId);
 return {...s,maxForwardMps:s.forwardSpeedLimitKph/3.6,maxReverseMps:s.reverseSpeedLimitKph/3.6,maxClimbingGrade:(published ? published.climbingGradePercent : s.climbingGradePercent)/100,maxSideGrade:(published ? published.sideSlopePercent : s.sideSlopePercent)/100};
}
