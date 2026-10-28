export function toBase64(buffer) {
  const bytes=new Uint8Array(buffer); let result='';
  for(let i=0;i<bytes.length;i+=32768) result+=String.fromCharCode(...bytes.subarray(i,i+32768));
  return btoa(result);
}

export function fromBase64(text) {
  const binary=atob(text), bytes=new Uint8Array(binary.length);
  for(let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
  return bytes.buffer;
}

export async function packCheckpoint(tf, model, optimizer, state) {
  const weights=await tf.io.encodeWeights(model.weights);
  const named=await optimizer.getWeights();
  const opt=await tf.io.encodeWeights(named);
  // Adam moment tensors belong to the optimizer. Only its synthesized iteration scalar is temporary.
  for(const item of named) if(item.name==='iter') item.tensor.dispose();
  return {format:'kjv-1m-v1',config:model.config,state,
    weights:{specs:weights.specs,data:toBase64(weights.data)},
    optimizer:{specs:opt.specs,data:toBase64(opt.data)}};
}

export function unpackWeights(tf, model, checkpoint) {
  if(checkpoint.format!=='kjv-1m-v1') throw Error('Unknown checkpoint format');
  for(const [key,value] of Object.entries(model.config)) if(checkpoint.config[key]!==value) throw Error(`Configuration mismatch: ${key}`);
  const decoded=tf.io.decodeWeights(fromBase64(checkpoint.weights.data),checkpoint.weights.specs);
  try {
    if(Object.keys(decoded).length!==Object.keys(model.weights).length) throw Error('Wrong number of tensors');
    for(const [key,v] of Object.entries(model.weights)) {
      if(!decoded[key] || decoded[key].dtype!=='float32' || String(decoded[key].shape)!==String(v.shape)) throw Error(`Invalid tensor: ${key}`);
    }
    for(const [key,v] of Object.entries(model.weights)) v.assign(decoded[key]);
  } finally {tf.dispose(decoded);}
}

export async function unpackOptimizer(tf, optimizer, checkpoint) {
  if(!checkpoint.optimizer) return;
  const tensors=tf.io.decodeWeights(fromBase64(checkpoint.optimizer.data),checkpoint.optimizer.specs);
  try {await optimizer.setWeights(checkpoint.optimizer.specs.map(s=>({name:s.name,tensor:tensors[s.name]})));}
  finally {tf.dispose(tensors);}
}

function dbOpen() {
  return new Promise((resolve,reject)=>{
    const request=indexedDB.open('kjv-200k',1);
    request.onupgradeneeded=()=>request.result.createObjectStore('checkpoints');
    request.onsuccess=()=>resolve(request.result); request.onerror=()=>reject(request.error);
  });
}
export async function storeCheckpoint(checkpoint, key='latest') {
  const db=await dbOpen();
  try {await new Promise((resolve,reject)=>{
    const tx=db.transaction('checkpoints','readwrite'); tx.objectStore('checkpoints').put(checkpoint,key);
    tx.oncomplete=resolve; tx.onerror=()=>reject(tx.error); tx.onabort=()=>reject(tx.error);
  });} finally {db.close();}
}
export async function readCheckpoint(key='latest') {
  const db=await dbOpen();
  try {return await new Promise((resolve,reject)=>{
    const request=db.transaction('checkpoints').objectStore('checkpoints').get(key);
    request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);
  });} finally {db.close();}
}
