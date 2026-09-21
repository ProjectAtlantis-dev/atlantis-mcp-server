import {waterClearance} from './water-clearance.mjs';
import {SpatialIndex} from './spatial-index.mjs';
import {groundHazard,groundSegmentClear} from './ground-surface.mjs';
/** Bounded navigation over verified DEM cells. Never moves a vehicle. */
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
function segmentDistance(p,a,b){const dx=b.x-a.x,dy=b.y-a.y,l=dx*dx+dy*dy;const t=l?Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/l)):0;return Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy);}
class Frontier {
 constructor(){this.items=[];}
 push(id,score){const a=this.items;let i=a.length;a.push({id,score});while(i){const p=(i-1)>>1;if(a[p].score<=score)break;a[i]=a[p];i=p;}a[i]={id,score};}
 pop(){const a=this.items,first=a[0],last=a.pop();if(a.length){let i=0;while(i*2+1<a.length){let c=i*2+1;if(c+1<a.length&&a[c+1].score<a[c].score)c++;if(a[c].score>=last.score)break;a[i]=a[c];i=c;}a[i]=last;}return first.id;}
 get size(){return this.items.length;}
}
export function planTerrainRoute(surface,start,target,{roadOnly=false,dense=false}={}) {
  const {rows,cols,stepM,minX,minY,heights}=surface;
  if(rows*cols>1024000)throw Error('navigation grid exceeds 1024000 cells');
  const gridPoint=i=>({x:minX+(i%cols)*stepM,y:minY+Math.floor(i/cols)*stepM});
  const index=p=>Math.max(0,Math.min(rows-1,Math.round((p.y-minY)/stepM)))*cols+Math.max(0,Math.min(cols-1,Math.round((p.x-minX)/stepM)));
  if(groundHazard(surface,start))throw Error('vehicle starts outside traversable terrain');
  const first=index(start),goal=index(target);
  const inside=target.x>=minX&&target.y>=minY&&target.x<=minX+(cols-1)*stepM&&target.y<=minY+(rows-1)*stepM;
  const point=i=>i===first?start:i===goal&&inside?target:gridPoint(i);
  const segments=(surface.roads??[]).flatMap(road=>road.path.slice(1).map((b,i)=>({a:road.path[i],b})));
  const roads=new SpatialIndex(segments,({a,b})=>({minX:Math.min(a.x,b.x)-4,maxX:Math.max(a.x,b.x)+4,minY:Math.min(a.y,b.y)-4,maxY:Math.max(a.y,b.y)+4}));
  const obstacles=new SpatialIndex(surface.obstacles??[],o=>({minX:o.minX-2,maxX:o.maxX+2,minY:o.minY-2,maxY:o.maxY+2}));
  const nearRoad=p=>roads.at(p.x,p.y).some(({a,b})=>segmentDistance(p,a,b)<=4);
  const valid=new Uint8Array(rows*cols),cost=new Float64Array(rows*cols);
  for(let i=0;i<valid.length;i++){
    const p=point(i);
    const onRoad=nearRoad(p);cost[i]=onRoad?1:1.8;
    // Bounds already include the vehicle clearance. Prefer extra turning room,
    // but do not reject a controller-valid start inside that preference band.
    const nearby=obstacles.at(p.x,p.y);
    const turningMargin=nearby.some(o=>p.x>=o.minX-2&&p.x<=o.maxX+2&&p.y>=o.minY-2&&p.y<=o.maxY+2);
    if(turningMargin)cost[i]+=20;
    if(surface.waterNavigation){
      const preferred=surface.waterNavigation.preferredClearanceM;
      // Strongly prefer open water while permitting hull-safe narrow passages.
      cost[i]+=12*Math.max(0,1-waterClearance(surface,p)/preferred)**2;
    }
    valid[i]=!groundHazard(surface,p,undefined,nearby)&&(!roadOnly||onRoad||distance(p,start)<=6);
  }
  if(!valid[first])throw Error('vehicle starts outside traversable terrain');

  if(inside&&!valid[goal])throw Error('destination is not traversable');
  const dist=new Float64Array(rows*cols).fill(Infinity),previous=new Int32Array(rows*cols).fill(-1),done=new Uint8Array(rows*cols);
  const open=new Frontier();open.push(first,distance(start,target));dist[first]=0;let best=-1,bestScore=Infinity;
  while(open.size){
    const current=open.pop();if(done[current])continue;done[current]=1;
    const p=point(current),x=current%cols,y=Math.floor(current/cols);
    if(inside&&current===goal){best=current;break;}
    if(!inside&&(x===1||y===1||x===cols-2||y===rows-2)&&distance(p,target)<distance(start,target)-4){
      const candidate=dist[current]+distance(p,target)*1.8;
      if(candidate<bestScore){bestScore=candidate;best=current;}
    }
    for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]){
      const nx=x+dx,ny=y+dy;if(nx<0||nx>=cols||ny<0||ny>=rows)continue;
      const next=ny*cols+nx;if(!valid[next]||done[next])continue;
      if(dx&&dy&&(!valid[y*cols+nx]||!valid[ny*cols+x]))continue;
      const length=distance(p,point(next));
      if(!groundSegmentClear(surface,p,point(next),undefined,obstacles))continue;
      const candidate=dist[current]+length*(cost[current]+cost[next])/2;
      if(candidate<dist[next]){dist[next]=candidate;previous[next]=current;open.push(next,candidate+distance(point(next),target));}
    }
  }
  if(best<0)throw Error('no traversable route in verified terrain');
  const ids=[];for(let i=best;i!==-1;i=previous[i])ids.push(i);ids.reverse();
  // Remove collinear grid points only; never shortcut across a hazard or corner.
  const route=[];for(let k=1;k<ids.length;k++){
    const a=point(ids[k-1]),b=point(ids[k]),c=k+1<ids.length?point(ids[k+1]):null;
    if(dense||!c||(b.x-a.x)*(c.y-b.y)!==(b.y-a.y)*(c.x-b.x))route.push(b);
  }
  if(inside){if(route.length)route[route.length-1]={...target};else route.push({...target});}
  return {points:route,complete:inside,mode:surface.navigationDomain==='water'?'water':roadOnly?'roads':'roads-and-offroad',roadCells:ids.filter(i=>cost[i]===1).length,cells:ids.length};
}
