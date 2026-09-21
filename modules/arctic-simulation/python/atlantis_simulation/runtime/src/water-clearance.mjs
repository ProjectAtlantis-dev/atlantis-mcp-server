// Conservative distance to land/unknown water, cached per immutable surface.
// Eight-neighbour distance is a lower bound on Euclidean shore clearance.
const fields=new WeakMap();
function field(s){
 let d=fields.get(s);if(d)return d;
 const {rows,cols}=s,n=rows*cols,q=new Int32Array(n);d=new Float64Array(n).fill(Infinity);let head=0,tail=0;
 for(let i=0;i<n;i++)if(s.water?.[i]!==true){d[i]=0;q[tail++]=i;}
 while(head<tail){const i=q[head++],x=i%cols,y=Math.floor(i/cols);
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
   const xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=cols||yy>=rows)continue;
   const j=yy*cols+xx;if(d[j]!==Infinity)continue;d[j]=d[i]+1;q[tail++]=j;
  }
 }
 fields.set(s,d);return d;
}
export function waterClearance(s,p){
 const x=Math.round((p.x-s.minX)/s.stepM),y=Math.round((p.y-s.minY)/s.stepM);
 if(x<0||y<0||x>=s.cols||y>=s.rows)return 0;
 const offset=Math.hypot(p.x-s.minX-x*s.stepM,p.y-s.minY-y*s.stepM);
 return Math.max(0,field(s)[y*s.cols+x]*s.stepM-s.stepM/Math.SQRT2-offset);
}
export function waterShortcutClear(s,a,b){
 if(!s.waterNavigation)return true;
 // Preference is soft at mask resolution; hard hull clearance is enforced
 // separately. Nearest-cell bounds vary by up to two grid steps along an edge.
 const minimum=Math.min(s.waterNavigation.preferredClearanceM,waterClearance(s,a),waterClearance(s,b))-2*s.stepM;
 const count=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)/(s.stepM/4)));
 for(let i=0;i<=count;i++){const t=i/count;if(waterClearance(s,{x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t})<minimum-1e-6)return false;}
 return true;
}
