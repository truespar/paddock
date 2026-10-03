// Joint-schema routing and scoring remain on GPU. Only final option logits
// cross back to the API thread; softmax/response shaping use the shared API.
kernel void clef_mean(device const float* x [[buffer(0)]],device const uint2* spans [[buffer(1)]],
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*p[1])return;uint row=i/p[0],col=i%p[0];uint2 s=spans[row];float sum=0;
 for(uint r=s.x;r<s.y;++r)sum+=x[ulong(r)*p[0]+col];y[i]=sum/float(s.y-s.x);
}
kernel void clef_lex(device const bfloat* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
 device const uint2* spans [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*p[1])return;uint row=i/p[0],col=i%p[0];uint2 s=spans[row];float sum=0;
 for(uint r=s.x;r<s.y;++r)sum+=float(w[ulong(ids[r])*p[0]+col]);y[i]=sum/float(s.y-s.x);
}
kernel void clef_gather_add(device float* y [[buffer(0)]],device const float* x [[buffer(1)]],
 device const uint* indices [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0]*p[1])y[i]+=x[ulong(indices[i/p[0]])*p[0]+i%p[0]];
}
kernel void clef_route(device const float* options [[buffer(0)]],device const float* fields [[buffer(1)]],
 device const uint2* ranges [[buffer(2)]],device float* out [[buffer(3)]],device float* scratch [[buffer(4)]],
 constant uint* p [[buffer(5)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 // Question ranges are disjoint. Reuse the final-logit plane while it is
 // dead, rather than reserving 16 KiB of shared memory for every question
 // (over 32 KiB with Metal's validation instrumentation).
 threadgroup float red[8];uint D=p[0];uint2 span=ranges[row];
 device float* weights=scratch+span.x;
 for(uint i=0;i<span.y;++i) {
  float sum=0;for(uint j=tid;j<D;j+=256)sum=fma(options[ulong(span.x+i)*D+j],fields[ulong(row)*D+j],sum);
  float dot=laya_sum(sum,red,lane,sg);if(tid==0)weights[i]=dot*rsqrt(float(D));
 }
 threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
 float hi=-INFINITY;for(uint i=0;i<span.y;++i)hi=max(hi,weights[i]);
 float den=0;for(uint i=0;i<span.y;++i)den+=exp(weights[i]-hi);
 for(uint j=tid;j<D;j+=256){float sum=0;for(uint i=0;i<span.y;++i)sum=fma(exp(weights[i]-hi)/den,options[ulong(span.x+i)*D+j],sum);out[ulong(row)*D+j]=sum;}
}
kernel void clef_features(device const float* fields [[buffer(0)]],device const float* options [[buffer(1)]],
 device const uint* qof [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*p[1])return;uint row=i/p[0],col=i%p[0],D=p[0];
 float a=fields[ulong(qof[row])*D+col],b=options[i];ulong at=ulong(row)*4*D+col;
 out[at]=a;out[at+D]=b;out[at+2*D]=a*b;out[at+3*D]=abs(a-b);
}
kernel void clef_score(device const float* lex [[buffer(0)]],device const float* qvec [[buffer(1)]],
 device const float* glob [[buffer(2)]],device const uint* qof [[buffer(3)]],device const uint* rof [[buffer(4)]],
 device const float* fields [[buffer(5)]],device const float* options [[buffer(6)]],device const float* hid [[buffer(7)]],
 device const float* weight [[buffer(8)]],device float* out [[buffer(9)]],constant uint* p [[buffer(10)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 #pragma clang fp reassociate(off)
 threadgroup float red[8];uint D=p[0],W=p[1],q=qof[row],r=rof[q];
 float ab=0,aa=0,bb=0;
 for(uint j=tid;j<D;j+=256){float a=lex[ulong(row)*D+j],b=qvec[ulong(q)*D+j]+glob[ulong(r)*D+j];ab=fma(a,b,ab);aa=fma(a,a,aa);bb=fma(b,b,bb);}
 ab=laya_sum(ab,red,lane,sg);aa=laya_sum(aa,red,lane,sg);bb=laya_sum(bb,red,lane,sg);
 float prior=(ab/max(sqrt(aa),1e-12f))/max(sqrt(bb),1e-12f)*as_type<float>(p[2]);
 float dot=0,ff=0,oo=0,res=0;
 for(uint j=tid;j<W;j+=256){float a=fields[ulong(q)*W+j],b=options[ulong(row)*W+j];dot=fma(a,b,dot);ff=fma(a,a,ff);oo=fma(b,b,oo);res=fma(hid[ulong(row)*W+j],weight[j],res);}
 dot=laya_sum(dot,red,lane,sg);ff=laya_sum(ff,red,lane,sg);oo=laya_sum(oo,red,lane,sg);res=laya_sum(res,red,lane,sg);
 if(tid==0){float cosine=(dot/max(sqrt(ff),1e-8f))/max(sqrt(oo),1e-8f);
  out[row]=prior+as_type<float>(p[4])*fma(as_type<float>(p[3]),cosine,res+as_type<float>(p[5]));}
}
