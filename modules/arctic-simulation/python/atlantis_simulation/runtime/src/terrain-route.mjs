/** Bounded navigation over verified DEM cells. Never moves a vehicle. */
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
function segmentDistance(p,a,b){const dx=b.x-a.x,dy=b.y-a.y,l=dx*dx+dy*dy;const t=l?Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/l)):0;return Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy);}
export function planTerrainRoute(surface,start,target,{roadOnly=false}={}) {
  const {rows,cols,stepM,minX,minY,heights}=surface;
  if(rows*cols>20000)throw Error('navigation grid exceeds 20000 cells');
  const point=i=>({x:minX+(i%cols)*stepM,y:minY+Math.floor(i/cols)*stepM});
  const index=p=>Math.max(0,Math.min(rows-1,Math.round((p.y-minY)/stepM)))*cols+Math.max(0,Math.min(cols-1,Math.round((p.x-minX)/stepM)));
  const roads=surface.roads??[];
  const nearRoad=p=>roads.some(r=>r.path.slice(1).some((b,i)=>segmentDistance(p,r.path[i],b)<=4));
  const valid=new Uint8Array(rows*cols),cost=new Float64Array(rows*cols);
  for(let i=0;i<valid.length;i++){
    const p=point(i),x=i%cols,y=Math.floor(i/cols),h=heights[i];
    const onRoad=nearRoad(p);cost[i]=onRoad?1:1.8;
    const dx=(heights[y*cols+Math.min(cols-1,x+1)]-heights[y*cols+Math.max(0,x-1)])/(stepM*(x===0||x===cols-1?1:2));
    const dy=(heights[Math.min(rows-1,y+1)*cols+x]-heights[Math.max(0,y-1)*cols+x])/(stepM*(y===0||y===rows-1?1:2));
    const collision=(surface.obstacles??[]).some(o=>p.x>=o.minX&&p.x<=o.maxX&&p.y>=o.minY&&p.y<=o.maxY);
    // Bounds already include the vehicle clearance. Prefer extra turning room,
    // but do not reject a controller-valid start inside that preference band.
    const turningMargin=(surface.obstacles??[]).some(o=>p.x>=o.minX-2&&p.x<=o.maxX+2&&p.y>=o.minY-2&&p.y<=o.maxY+2);
    if(turningMargin)cost[i]+=20;
    valid[i]=Number.isFinite(h)&&h>.25&&(!surface.water||surface.water[i]===false)&&Math.hypot(dx,dy)<=Math.tan(Math.PI/6)&&!collision&&(!roadOnly||onRoad||distance(p,start)<=6);
  }
  const first=index(start),goal=index(target);
  if(!valid[first])throw Error('vehicle starts outside traversable terrain');
  const inside=target.x>=minX&&target.y>=minY&&target.x<=minX+(cols-1)*stepM&&target.y<=minY+(rows-1)*stepM;
  if(inside&&!valid[goal])throw Error('destination is not traversable');
  const dist=new Float64Array(rows*cols).fill(Infinity),previous=new Int32Array(rows*cols).fill(-1),done=new Uint8Array(rows*cols);
  const open=new Set([first]);dist[first]=0;let best=-1,bestScore=Infinity;
  while(open.size){
    let current=-1,score=Infinity;
    for(const i of open){const f=dist[i]+distance(point(i),target);if(f<score){score=f;current=i;}}
    open.delete(current);done[current]=1;
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
      const length=stepM*Math.hypot(dx,dy);
      if(Math.abs(heights[next]-heights[current])/length>Math.tan(Math.PI/6))continue;
      const candidate=dist[current]+length*(cost[current]+cost[next])/2;
      if(candidate<dist[next]){dist[next]=candidate;previous[next]=current;open.add(next);}
    }
  }
  if(best<0)throw Error('no traversable route in verified terrain');
  const ids=[];for(let i=best;i!==-1;i=previous[i])ids.push(i);ids.reverse();
  // Remove collinear grid points only; never shortcut across a hazard or corner.
  const route=[];for(let k=1;k<ids.length;k++){
    const a=point(ids[k-1]),b=point(ids[k]),c=k+1<ids.length?point(ids[k+1]):null;
    if(!c||(b.x-a.x)*(c.y-b.y)!==(b.y-a.y)*(c.x-b.x))route.push(b);
  }
  if(inside){if(route.length)route[route.length-1]={...target};else route.push({...target});}
  return {points:route,complete:inside,mode:roadOnly?'roads':'roads-and-offroad',roadCells:ids.filter(i=>cost[i]===1).length,cells:ids.length};
}
