/** Broad-phase lookup only: callers still apply the exact geometry predicate. */
export class SpatialIndex {
 constructor(items,bounds,cellSize=64){
  this.cellSize=cellSize;this.cells=new Map();
  for(const item of items){const box=bounds(item);
   for(let y=Math.floor(box.minY/cellSize);y<=Math.floor(box.maxY/cellSize);y++)for(let x=Math.floor(box.minX/cellSize);x<=Math.floor(box.maxX/cellSize);x++){
    const key=`${x},${y}`;let cell=this.cells.get(key);if(!cell){cell=[];this.cells.set(key,cell);}cell.push(item);
   }
  }
 }
 at(x,y){return this.cells.get(`${Math.floor(x/this.cellSize)},${Math.floor(y/this.cellSize)}`)??[];}
}
