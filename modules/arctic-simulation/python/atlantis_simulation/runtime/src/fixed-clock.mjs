/** Monotonic elapsed time is never discarded; catch-up work is bounded per pump. */
export class FixedClock {
  constructor({hz=30,maxStepsPerPump=8}={}) {
    if(!Number.isFinite(hz)||hz<=0)throw new RangeError('hz must be positive');
    if(!Number.isInteger(maxStepsPerPump)||maxStepsPerPump<1)throw new RangeError('maxStepsPerPump must be positive');
    this.stepSeconds=1/hz;this.maxSteps=maxStepsPerPump;this.pendingSeconds=0;
  }
  advance(elapsedSeconds,step) {
    if(!Number.isFinite(elapsedSeconds)||elapsedSeconds<0)throw new RangeError('elapsedSeconds must be finite and nonnegative');
    this.pendingSeconds+=elapsedSeconds;
    let count=0;
    while(this.pendingSeconds+1e-12>=this.stepSeconds&&count<this.maxSteps){
      step(this.stepSeconds);this.pendingSeconds=Math.max(0,this.pendingSeconds-this.stepSeconds);count++;
    }
    return {steps:count,pendingSeconds:this.pendingSeconds};
  }
}
