import {waterClearance} from './water-clearance.mjs';
import {groundProfile} from './vehicle-performance.mjs';
const legacyProfile=groundProfile('patria-amv');
export function surfaceHeight(surface,x,y){
  const gx=(x-surface.minX)/surface.stepM,gy=(y-surface.minY)/surface.stepM;
  if(gx<0||gy<0||gx>surface.cols-1||gy>surface.rows-1)return null;
  const ix=Math.min(surface.cols-2,Math.floor(gx)),iy=Math.min(surface.rows-2,Math.floor(gy));
  const tx=gx-ix,ty=gy-iy,h=(xx,yy)=>surface.heights[yy*surface.cols+xx];
  if([h(ix,iy),h(ix+1,iy),h(ix,iy+1),h(ix+1,iy+1)].some(v=>!Number.isFinite(v)))return NaN;
  return (h(ix,iy)*(1-tx)+h(ix+1,iy)*tx)*(1-ty)+(h(ix,iy+1)*(1-tx)+h(ix+1,iy+1)*tx)*ty;
}
export function surfaceNormal(s,p){
  const d=Math.min(1,s.stepM/2),x0=Math.max(s.minX,p.x-d),x1=Math.min(s.minX+(s.cols-1)*s.stepM,p.x+d);
  const y0=Math.max(s.minY,p.y-d),y1=Math.min(s.minY+(s.rows-1)*s.stepM,p.y+d);
  const x=-(surfaceHeight(s,x1,p.y)-surfaceHeight(s,x0,p.y))/(x1-x0);
  const y=-(surfaceHeight(s,p.x,y1)-surfaceHeight(s,p.x,y0))/(y1-y0);
  const length=Math.hypot(x,y,1);return {x:x/length,y:y/length,z:1/length};
}


export function groundHazard(surface,point,headingRad,obstacles=surface.obstacles??[]){
 const {x,y}=point,z=surfaceHeight(surface,x,y);
 const profile=surface.groundProfile??legacyProfile;
 if(surface.navigationDomain==='water'){
  if(z===null)return 'water-coverage-boundary';
  const row=Math.round((y-surface.minY)/surface.stepM),col=Math.round((x-surface.minX)/surface.stepM);
  if(surface.water?.[row*surface.cols+col]!==true)return 'land-or-unverified-water';
  if(surface.waterNavigation&&waterClearance(surface,point)<surface.waterNavigation.hullRadiusM)return 'insufficient-hull-clearance';
  if(obstacles.some(o=>x>=o.minX&&x<=o.maxX&&y>=o.minY&&y<=o.maxY))return 'building-clearance';
  return null;
 }
 if(z===null)return 'terrain-coverage-boundary';
 if(!Number.isFinite(z))return 'terrain-elevation-unavailable';
 if(z<=.25)return 'water-or-sea-level-terrain';
 if(surface.water){
  const row=Math.round((y-surface.minY)/surface.stepM),col=Math.round((x-surface.minX)/surface.stepM);
  if(surface.water[row*surface.cols+col]!==false)return 'water-or-unknown-surface';
 }
 const n=surfaceNormal(surface,point);
 const dx=-n.x/n.z,dy=-n.y/n.z;
 if(!Number.isFinite(dx)||!Number.isFinite(dy))return 'terrain-elevation-unavailable';
 if(headingRad===undefined){
  if(Math.hypot(dx,dy)>Math.hypot(profile.maxClimbingGrade,profile.maxSideGrade))return 'terrain-slope-exceeds-profile';
 }else{
  if(Math.abs(-Math.sin(headingRad)*dx+Math.cos(headingRad)*dy)>profile.maxClimbingGrade)return 'terrain-climb-exceeds-profile';
  if(Math.abs(Math.cos(headingRad)*dx+Math.sin(headingRad)*dy)>profile.maxSideGrade)return 'terrain-side-slope-exceeds-profile';
 }
 if(obstacles.some(o=>x>=o.minX&&x<=o.maxX&&y>=o.minY&&y<=o.maxY))return 'building-clearance';
 return null;
}

export function groundSegmentClear(surface,a,b,headingRad=Math.atan2(-(b.x-a.x),b.y-a.y),obstacleIndex=null){
 const count=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)/(surface.stepM/4)));
 for(let i=0;i<=count;i++){
  const t=i/count;
  const point={x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t};
  if(groundHazard(surface,point,headingRad,obstacleIndex?obstacleIndex.at(point.x,point.y):surface.obstacles??[]))return false;
 }
 return true;
}
