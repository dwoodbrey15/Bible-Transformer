import {CONFIG,Transformer,trainStep,makeBatch,seededRandom,evaluate,generate} from './model.mjs';
import {packCheckpoint,unpackWeights,unpackOptimizer,storeCheckpoint,readCheckpoint} from './checkpoint.mjs';

const tf=globalThis.tf, $=(id)=>document.getElementById(id);
let model,optimizer,manifest,trainData,valData,busy=false,stopRequested=false,wakeLock=null;
let sampler=seededRandom(42);
let state={steps:0,bytesSeen:0,loss:null,val:null,bestVal:null,staleEvaluations:0,lastEvalStep:-1,history:[],samplerState:42,settings:{context:64,batch:1,lr:0.0005}};
const controls=['train','validate','download','restore','restoreBest','import','generate','runSteps','batch','context','lr','prompt','newBytes','temperature'];
const logLines=[];
function log(text){logLines.push(text);$('log').textContent=logLines.slice(-16).join('\n');$('log').scrollTop=$('log').scrollHeight;}
function status(text){$('status').textContent=text;}
function lock(value){busy=value;controls.forEach(id=>$(id).disabled=value);$('stop').disabled=true;}
function settings(){return {context:Number($('context').value),batch:Number($('batch').value),lr:Number($('lr').value)};}
function sync(){
  $('stepsValue').textContent=state.steps.toLocaleString();
  $('lossValue').textContent=state.loss===null?'—':state.loss.toFixed(3);
  $('valValue').textContent=state.val===null?'—':(state.val/Math.LN2).toFixed(3);
  $('passesValue').textContent=(state.bytesSeen/(trainData?.length||1)).toFixed(2)+'×';drawChart();
}
function drawChart(){
  const canvas=$('chart'), ratio=globalThis.devicePixelRatio||1, width=canvas.clientWidth,height=160;
  canvas.width=width*ratio;canvas.height=height*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
  ctx.strokeStyle='#364137';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(36,12);ctx.lineTo(36,136);ctx.lineTo(width-12,136);ctx.stroke();
  const points=state.history;if(points.length<2)return;
  const lo=Math.min(...points.map(p=>p.loss)),hi=Math.max(...points.map(p=>p.loss)),range=Math.max(hi-lo,.1);
  ctx.fillStyle='#adb8a9';ctx.font='11px system-ui';ctx.fillText(hi.toFixed(2),0,18);ctx.fillText(lo.toFixed(2),0,137);
  ctx.strokeStyle='#dfc58c';ctx.lineWidth=2;ctx.beginPath();
  points.forEach((p,i)=>{const x=40+(p.step-points[0].step)/Math.max(1,points.at(-1).step-points[0].step)*(width-58),y=18+(hi-p.loss)/range*112;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
}
function createOptimizer(){return tf.train.adam(state.settings.lr,0.9,0.95,1e-8);}
async function checkpoint(){
  state.samplerState=sampler.state();
  return packCheckpoint(tf,model,optimizer,{...state,manifest,history:state.history.slice(-400)});
}
async function save(){
  const packed=await checkpoint();await storeCheckpoint(packed);
  $('saved').textContent=`Autosaved at step ${state.steps.toLocaleString()} · ${new Date().toLocaleTimeString()}`;
  return packed;
}
async function autosave(){try{await save();}catch(error){log(`Autosave failed: ${error.message}. Download a checkpoint when the run ends.`);}}
async function trainingValidation(){
  if(state.lastEvalStep===state.steps)return;
  const s=state.settings;
  state.val=await evaluate(tf,model,valData,s.batch,s.context);state.lastEvalStep=state.steps;
  if(state.bestVal===null||state.val<state.bestVal){
    state.bestVal=state.val;state.staleEvaluations=0;
    try{await storeCheckpoint(await checkpoint(),'best');}catch(error){log(`Best-checkpoint save failed: ${error.message}`);}
    log(`New best validation: ${(state.val/Math.LN2).toFixed(3)} bits/byte. Best weights saved.`);
  }else{state.staleEvaluations++;log(`Validation: ${(state.val/Math.LN2).toFixed(3)} bits/byte; ${state.staleEvaluations}/10 checks without improvement.`);}
  sync();
}
async function restore(packed){
  if(!packed)throw Error('No autosave found. Import a downloaded checkpoint instead.');
  if(packed.state.manifest?.streams?.train?.sha256!==manifest.streams.train.sha256 || packed.state.manifest?.streams?.val?.sha256!==manifest.streams.val.sha256) throw Error('The checkpoint was trained on a different corpus.');
  // Validate and assign into the existing model; then replace the optimizer.
  unpackWeights(tf,model,packed);
  optimizer.dispose();state={bestVal:null,staleEvaluations:0,lastEvalStep:-1,...packed.state};
  sampler=seededRandom();sampler.restore(state.samplerState);
  optimizer=createOptimizer();await unpackOptimizer(tf,optimizer,packed);
  $('context').value=state.settings.context;$('batch').value=state.settings.batch;$('lr').value=state.settings.lr;
  sync();status(`Restored step ${state.steps.toLocaleString()}. Ready to continue.`);log(`Restored step ${state.steps}.`);
}
async function action(fn){if(busy)return;lock(true);try{await fn();}catch(error){console.error(error);status(error.message);log(`Error: ${error.message}`);}finally{lock(false);}}
async function sha(bytes){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');}
async function fetchChecked(name){
  const response=await fetch(`data/${name}.bin`);if(!response.ok)throw Error(`Could not load ${name}.bin`);
  const bytes=new Uint8Array(await response.arrayBuffer());
  if(await sha(bytes)!==manifest.streams[name].sha256)throw Error(`Corpus checksum mismatch: ${name}`);return bytes;
}

$('train').onclick=()=>action(async()=>{
  const count=Number($('runSteps').value), s=settings();
  if(!Number.isInteger(count)||count<1||count>500000||s.lr<=0||s.lr>0.01)throw Error('Check the step count and learning rate.');
  if(state.steps && (s.context!==state.settings.context||s.batch!==state.settings.batch)){
    state.val=null;state.bestVal=null;state.staleEvaluations=0;state.lastEvalStep=-1;log('Context or batch changed: validation comparison restarted.');
  }
  state.settings=s;stopRequested=false;$('stop').disabled=false;
  try {wakeLock=await navigator.wakeLock?.request('screen');}catch{}
  const start=performance.now(),begin=state.steps;let earlyStopped=false;
  status('Training on this device. The first step may take time to compile shaders…');
  try {
    for(let n=0;n<count&&!stopRequested;n++) {
      const [x,y]=makeBatch(tf,trainData,s.batch,s.context,sampler);
      try {state.loss=await trainStep(tf,model,optimizer,x,y,s.lr*Math.min(1,(state.steps+1)/100));}
      finally {tf.dispose([x,y]);}
      state.steps++;state.bytesSeen+=s.batch*s.context;
      if(state.steps%10===0||n===0){
        state.history.push({step:state.steps,loss:state.loss});state.history=state.history.slice(-400);
        const elapsed=(performance.now()-start)/1000,stepTime=elapsed/(state.steps-begin);
        sync();status(`Step ${state.steps.toLocaleString()} · ${stepTime.toFixed(2)} s/step · ${Math.round(s.batch*s.context/stepTime)} bytes/s`);
        log(`Step ${state.steps}: loss ${state.loss.toFixed(4)}; ${stepTime.toFixed(2)} s/step`);
      }
      if(state.steps%200===0){await trainingValidation();if(state.staleEvaluations>=10){earlyStopped=true;break;}}
      if(state.steps%100===0)await autosave();
      await tf.nextFrame();
    }
    if(!stopRequested&&!earlyStopped){status('Measuring held-out chapters…');await trainingValidation();}
    if(state.staleEvaluations>=10)earlyStopped=true;
    await autosave();sync();status(earlyStopped?`Early stopping at step ${state.steps.toLocaleString()}: 10 validation checks without improvement. Restore best for generation.`:`Run ${stopRequested?'stopped':'complete'} at step ${state.steps.toLocaleString()}. Download a checkpoint to keep your progress.`);
  } finally {await wakeLock?.release();wakeLock=null;}
});
$('stop').onclick=()=>{stopRequested=true;$('stop').disabled=true;status('Stopping after the current step, then saving…');};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&busy){stopRequested=true;}});
$('validate').onclick=()=>action(async()=>{status('Measuring held-out chapters…');const s=settings();state.val=await evaluate(tf,model,valData,s.batch,s.context);sync();status(`Validation: ${(state.val/Math.LN2).toFixed(3)} bits/byte.`);});
$('generate').onclick=()=>action(async()=>{
  const count=Number($('newBytes').value),temperature=Number($('temperature').value);
  if(!Number.isInteger(count)||count<1||count>1000||temperature<0.1||temperature>2)throw Error('Use 1–1000 bytes and a temperature of 0.1–2.');
  status('Generating on this device…');$('output').textContent=await generate(tf,model,$('prompt').value,count,settings().context,temperature);status('Continuation generated. It may contain invented or altered verses.');
});
$('download').onclick=()=>action(async()=>{
  const packed=await checkpoint(),url=URL.createObjectURL(new Blob([JSON.stringify(packed)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download=`kjv-200k-step-${state.steps}.checkpoint.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
  status('Checkpoint download started. Save it to Files on your iPhone.');
});
$('restore').onclick=()=>action(async()=>restore(await readCheckpoint()));
$('restoreBest').onclick=()=>action(async()=>{const packed=await readCheckpoint('best');if(!packed)throw Error('No best checkpoint yet. Complete a training run first.');await restore(packed);});
$('import').onclick=()=>$('importFile').click();
$('importFile').onchange=()=>{const file=$('importFile').files[0];if(file)action(async()=>{if(file.size>40*1024*1024)throw Error('Checkpoint exceeds 40 MiB.');await restore(JSON.parse(await file.text()));await autosave();});$('importFile').value='';};
window.addEventListener('resize',drawChart);

lock(true);
try {
  if(!tf)throw Error('TensorFlow.js did not load. Confirm vendor/tf.min.js was uploaded.');
  try {await tf.setBackend('webgl');await tf.ready();}catch{await tf.setBackend('cpu');await tf.ready();}
  // Avoid silently training Adam with half-precision textures.
  if(tf.getBackend()==='webgl'&&!tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')){
    await tf.setBackend('cpu');await tf.ready();log('Float32 WebGL textures unavailable; using the slower CPU backend.');
  }
  $('backend').textContent=`${tf.getBackend().toUpperCase()} · on this device`;
  const response=await fetch('data/manifest.json');if(!response.ok)throw Error('Could not load data/manifest.json');manifest=await response.json();
  [trainData,valData]=await Promise.all([fetchChecked('train'),fetchChecked('val')]);
  model=new Transformer(tf);optimizer=createOptimizer();
  if(model.parameterCount!==205376)throw Error('Unexpected model size');
  log(`Verified ${trainData.length.toLocaleString()} training bytes and ${valData.length.toLocaleString()} validation bytes.`);
  status('Ready. New random weights loaded. Run 10–100 steps to measure this device, or restore your checkpoint.');
  sync();lock(false);
}catch(error){console.error(error);status(`Could not start: ${error.message}. Serve this folder over HTTPS (GitHub Pages) or localhost.`);}
