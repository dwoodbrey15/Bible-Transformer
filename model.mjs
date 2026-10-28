// TensorFlow.js implementation matching model.py, including the RoPE convention.
// tf is injected so the same module can be tested in Node and run in Safari.
export const CONFIG = Object.freeze({vocab_size:256, dim:64, layers:4, heads:4,
  hidden:160, context:256, rope_base:10000, eps:1e-5});

export class Transformer {
  constructor(tf, config=CONFIG, seed=42) {
    this.tf=tf; this.config={...config}; this.weights={};
    const c=this.config;
    if (c.vocab_size!==256 || c.dim%c.heads || (c.dim/c.heads)%2) throw Error('Invalid byte-transformer configuration');
    let n=seed;
    const add=(key,shape,scale=0.02,ones=false)=>{
      const initial=ones ? tf.ones(shape) : tf.randomNormal(shape,0,scale,'float32',n++);
      this.weights[key]=tf.variable(initial,true,key); initial.dispose();
    };
    add('embed',[256,c.dim]);
    for(let i=0;i<c.layers;i++) {
      const p=`blocks.${i}.`;
      add(p+'attn_norm',[c.dim],1,true); add(p+'ffn_norm',[c.dim],1,true);
      for(const key of ['wq','wk','wv']) add(p+key,[c.dim,c.dim]);
      add(p+'wo',[c.dim,c.dim],0.02/Math.sqrt(2*c.layers));
      add(p+'gate',[c.dim,c.hidden]); add(p+'up',[c.dim,c.hidden]);
      add(p+'down',[c.hidden,c.dim],0.02/Math.sqrt(2*c.layers));
    }
    add('norm',[c.dim],1,true);
    this.variables=Object.values(this.weights);
    this.parameterCount=this.variables.reduce((s,v)=>s+v.size,0);
    const half=c.dim/c.heads/2, cos=[],sin=[],mask=[];
    for(let t=0;t<c.context;t++) {
      for(let j=0;j<half;j++) {
        const angle=t/Math.pow(c.rope_base,2*j/(2*half));
        cos.push(Math.cos(angle)); sin.push(Math.sin(angle));
      }
      for(let j=0;j<c.context;j++) mask.push(j>t ? -1e4 : 0);
    }
    this.cos=tf.tensor2d(cos,[c.context,half]); this.sin=tf.tensor2d(sin,[c.context,half]);
    this.mask=tf.tensor2d(mask,[c.context,c.context]);
  }

  forward(ids) {
    const tf=this.tf, c=this.config, [b,t]=ids.shape, d=c.dim, h=c.heads, hd=d/h;
    if(t<1 || t>c.context) throw Error('Context length exceeded');
    return tf.tidy(()=>{
      const cos=this.cos.slice([0,0],[t,hd/2]).reshape([1,1,t,hd/2]);
      const sin=this.sin.slice([0,0],[t,hd/2]).reshape([1,1,t,hd/2]);
      const mask=this.mask.slice([0,0],[t,t]).reshape([1,1,t,t]);
      const linear=(x,w)=>tf.matMul(x.reshape([-1,x.shape.at(-1)]),w).reshape([...x.shape.slice(0,-1),w.shape[1]]);
      const rms=(x,w)=>x.mul(x.square().mean(-1,true).add(c.eps).rsqrt()).mul(w);
      const rope=(x)=>{
        const [left,right]=tf.split(x,2,-1);
        return tf.concat([left.mul(cos).sub(right.mul(sin)),right.mul(cos).add(left.mul(sin))],-1);
      };
      let x=tf.gather(this.weights.embed,ids);
      for(let i=0;i<c.layers;i++) {
        const p=`blocks.${i}.`, w=(key)=>this.weights[p+key];
        let a=rms(x,w('attn_norm'));
        const project=(key)=>linear(a,w(key)).reshape([b,t,h,hd]).transpose([0,2,1,3]);
        const q=rope(project('wq')), k=rope(project('wk')), v=project('wv');
        // Explicitly tile the mask for reliable gradient broadcasting on TF.js backends.
        const scores=tf.matMul(q,k,false,true).mul(1/Math.sqrt(hd)).add(mask.tile([b,h,1,1]));
        a=tf.matMul(tf.softmax(scores,-1),v).transpose([0,2,1,3]).reshape([b,t,d]);
        x=x.add(linear(a,w('wo')));
        a=rms(x,w('ffn_norm'));
        const gate=linear(a,w('gate'));
        x=x.add(linear(gate.mul(tf.sigmoid(gate)).mul(linear(a,w('up'))),w('down')));
      }
      return tf.matMul(rms(x,this.weights.norm).reshape([b*t,d]),this.weights.embed,false,true).reshape([b,t,256]);
    });
  }

  loss(x,y) {
    const tf=this.tf;
    return tf.tidy(()=>{
      const logits=this.forward(x).reshape([-1,256]);
      return tf.oneHot(y.reshape([-1]),256).mul(tf.logSoftmax(logits,-1)).sum(-1).mean().neg();
    });
  }

  dispose() {
    this.variables.forEach(v=>v.dispose()); this.cos.dispose(); this.sin.dispose(); this.mask.dispose();
  }
}

export async function trainStep(tf, model, optimizer, x, y, learningRate, weightDecay=0.1) {
  const result=tf.tidy(()=>{
    const {value,grads}=tf.variableGrads(()=>model.loss(x,y),model.variables);
    const norm=tf.addN(Object.values(grads).map(g=>g.square().sum())).sqrt();
    return {value,grads,norm};
  });
  try {
    const loss=(await result.value.data())[0], norm=(await result.norm.data())[0];
    if(!Number.isFinite(loss) || !Number.isFinite(norm)) throw Error('Non-finite loss/gradients. Lower the learning rate.');
    tf.tidy(()=>{
      const scale=Math.min(1,1/(norm+1e-12));
      const clipped=Object.fromEntries(Object.entries(result.grads).map(([key,g])=>[key,g.mul(scale)]));
      optimizer.learningRate=learningRate;
      for(const v of model.variables) if(v.rank>=2) v.assign(v.mul(1-learningRate*weightDecay));
      optimizer.applyGradients(clipped);
    });
    return loss;
  } finally { tf.dispose(result); }
}

export function makeBatch(tf, corpus, batchSize, context, random=Math.random) {
  if(corpus.length<=context) throw Error('Corpus is shorter than the context');
  const x=new Int32Array(batchSize*context), y=new Int32Array(x.length);
  for(let b=0;b<batchSize;b++) {
    const start=Math.floor(random()*(corpus.length-context));
    for(let t=0;t<context;t++) {
      x[b*context+t]=corpus[start+t]; y[b*context+t]=corpus[start+t+1];
    }
  }
  return [tf.tensor2d(x,[batchSize,context],'int32'),tf.tensor2d(y,[batchSize,context],'int32')];
}

export function seededRandom(seed=42) {
  // Serializable mulberry32 state for checkpointed data sampling.
  let state=seed>>>0;
  const random=()=>{let t=state=(state+0x6D2B79F5)>>>0;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  random.state=()=>state; random.restore=(s)=>{state=s>>>0;}; return random;
}

export async function evaluate(tf, model, corpus, batchSize, context, batches=8) {
  const random=seededRandom(12345);
  let total=0;
  for(let i=0;i<batches;i++) {
    const [x,y]=makeBatch(tf,corpus,batchSize,context,random);
    let loss;
    try {loss=model.loss(x,y); total+=(await loss.data())[0];}
    finally {tf.dispose([x,y,loss]);}
    await tf.nextFrame();
  }
  return total/batches;
}

export async function generate(tf, model, prompt, count, context, temperature=0.8, topK=40) {
  const ids=[...new TextEncoder().encode(prompt || '\n')];
  for(let i=0;i<count;i++) {
    const scores=tf.tidy(()=>{
      const window=ids.slice(-context);
      const x=tf.tensor2d(window,[1,window.length],'int32');
      return model.forward(x).slice([0,window.length-1,0],[1,1,256]).reshape([256]);
    });
    const values=await scores.data(); scores.dispose();
    const ranked=Array.from(values,(v,id)=>({id,v:v/temperature})).sort((a,b)=>b.v-a.v).slice(0,topK);
    const probs=ranked.map(x=>Math.exp(x.v-ranked[0].v));
    let draw=Math.random()*probs.reduce((a,b)=>a+b,0), chosen=ranked.at(-1).id;
    for(let j=0;j<ranked.length;j++){draw-=probs[j];if(draw<=0){chosen=ranked[j].id;break;}}
    ids.push(chosen); await tf.nextFrame();
  }
  return new TextDecoder().decode(new Uint8Array(ids));
}
