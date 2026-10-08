// Paddock's Conformer arithmetic, shared contract with embedding_gemma2_audio.cuh.
// F16 is an operand cut, not a residual cut. Each audio clip has its own
// causal domain. No CPU frontend, global reductions or quadratic attention.
kernel void eg2a_mm128(device half* w [[buffer(0)]],device half* x [[buffer(1)]],
    device uint* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]]) {vis_project<128,half,half>(w,x,out,bias,p,g);}
inline float eg2a_sum(float v,threadgroup float* red,uint tid) {
    v=simd_sum(v);threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid%32==0)red[tid/32]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=red[0];for(uint i=1;i<8;++i)sum+=red[i];return sum;
}
inline float4 eg2a_rms(float4 v,device const float* w,threadgroup float* red,uint tid,bool weighted) {
    float r=rsqrt(eg2a_sum(((v.x*v.x+v.y*v.y)+v.z*v.z)+v.w*v.w,red,tid)/1024.0f+1e-6f);
    return (v*r)*(weighted?reinterpret_cast<device const float4*>(w)[tid]:float4(1));
}
kernel void eg2a_positions(device float* out [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i>=13*1024)return;uint j=i%1024;float inc=float(log(10000.0)/511.0);
    float angle=float(12-i/1024)*exp(-float(j%512)*inc),v=j<512?sin(angle):cos(angle);out[i]=p[0]?mlx_bf(v):v;
}
// Two-float butterflies preserve quiet bins without device FP64 (unavailable
// on Apple GPUs). Only the 512-point FFT needs this: the processor rounds its
// windowed samples and final spectrum to F32. A plain F32 FFT loses enough
// cancellation accuracy to perturb log-mel on pure tones and quiet tails.
inline float2 eg2a_add(float2 a,float2 b) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float s=a.x+b.x,v=s-a.x,e=(a.x-(s-v))+(b.x-v);
    e+=(a.y+b.y);float h=s+e;return float2(h,e-(h-s));
}
inline float2 eg2a_mul(float2 a,float2 b) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float p=a.x*b.x,e=fma(a.x,b.x,-p)+(a.x*b.y+a.y*b.x);
    float h=p+e;return float2(h,e-(h-p));
}
kernel void eg2a_mel(device const float* pcm [[buffer(0)]],device const float* window [[buffer(1)]],
    device const float* fb [[buffer(2)]],device const float4* tw [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint frame [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    // Quiet FFT bins are differences of much larger values. Keep the
    // window/FFT rounding boundaries; relaxed contraction amplifies their
    // error through the logarithm even when speech sounds unchanged.
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    threadgroup float4 z[512];threadgroup float mag[257];
    for(uint i=tid;i<512;i+=256){int at=int(frame*160+i)-160;float v=0;
        if(i<320 && at>=0 && at<int(p[0]))v=pcm[at]*window[i];z[reverse_bits(i)>>23]=float4(v,0,0,0);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint size=2;size<=512;size*=2){uint halfsize=size/2,k=tid%halfsize,a=tid/halfsize*size+k,b=a+halfsize;
        float4 t=tw[k*(512/size)],e=z[a],w=z[b];
        float2 re=eg2a_add(eg2a_mul(t.xz,w.xz),-eg2a_mul(t.yw,w.yw));
        float2 im=eg2a_add(eg2a_mul(t.yw,w.xz),eg2a_mul(t.xz,w.yw));
        float2 ar=eg2a_add(e.xz,re),ai=eg2a_add(e.yw,im),br=eg2a_add(e.xz,-re),bi=eg2a_add(e.yw,-im);
        z[a]=float4(ar.x,ai.x,ar.y,ai.y);z[b]=float4(br.x,bi.x,br.y,bi.y);
        threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(uint k=tid;k<257;k+=256)mag[k]=precise::sqrt(fma(z[k].x,z[k].x,z[k].y*z[k].y));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<128){float sum=0;for(uint k=0;k<257;++k)sum=fma(fb[tid*257+k],mag[k],sum);
        out[frame*128+tid]=precise::log(p[1]?max(sum,1e-3f):sum+1e-3f);}
}
kernel void eg2a_subsample(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
    device const float* nw [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float patch[9*128],red[4];uint ci=p[2],co=p[3];
    for(uint i=tid;i<9*ci;i+=co){int t=int(g.y*2+i/ci/3)-1,f=int(g.x*2+i/ci%3)-1;
        patch[i]=(t>=0 && t<int(p[0]) && f>=0 && f<int(p[1]))?x[(t*p[1]+f)*ci+i%ci]:0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);float v=0;
    for(uint c=0;c<ci;++c)for(uint k=0;k<9;++k)v=fma(w[(tid*ci+c)*9+k],patch[k*ci+c],v);
    float s=simd_sum(v);if(tid%32==0)red[tid/32]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=0;for(uint i=0;i<co/32;++i)mean+=red[i];mean/=float(co);float dv=v-mean;
    s=simd_sum(dv*dv);threadgroup_barrier(mem_flags::mem_threadgroup);if(tid%32==0)red[tid/32]=s;
    threadgroup_barrier(mem_flags::mem_threadgroup);float var=0;for(uint i=0;i<co/32;++i)var+=red[i];
    out[(g.y*((p[1]+1)/2)+g.x)*co+tid]=max(dv*rsqrt(var/float(co)+1e-6f)*nw[tid],0.0f);
}
kernel void eg2a_seam(device float4* x [[buffer(0)]],device const float4* y [[buffer(1)]],
    device const float* post [[buffer(2)]],device const float* close [[buffer(3)]],device const float* next [[buffer(4)]],
    device half4* stage [[buffer(5)]],constant uint* p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];uint i=row*256+tid;float4 v=x[i];
    if(p[0]&1){float4 a=clamp(y[i],as_type<float>(p[1]),as_type<float>(p[2]));
        if(p[0]&2)a=eg2a_rms(a,post,red,tid,true);v+=a*as_type<float>(p[3]);}
    if(p[0]&4)v=eg2a_rms(v,close,red,tid,true);
    if(p[0]&5)x[i]=v;
    if(p[0]&16){if(p[0]&8)v=eg2a_rms(v,next,red,tid,true);v=clamp(v,as_type<float>(p[4]),as_type<float>(p[5]));
        if(p[0]&32)reinterpret_cast<device float4*>(stage)[i]=v;else stage[i]=half4(v);}
}
kernel void eg2a_act(device const float* x [[buffer(0)]],device half* y [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0])return;float v=clamp(x[i],as_type<float>(p[1]),as_type<float>(p[2]));
    v=clamp(v/(1.0f+exp(-v)),as_type<float>(p[3]),as_type<float>(p[4]));
    if(p[5])reinterpret_cast<device float*>(y)[i]=v;else y[i]=half(v);
}
kernel void eg2a_attention(device const float4* qkv [[buffer(0)]],device const float4* rel [[buffer(1)]],
    device const float4* pds [[buffer(2)]],device half4* out [[buffer(3)]],device const uint* starts [[buffer(4)]],constant uint* p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint local=row-starts[row];uint lane=tid%32;float4 q=clamp(qkv[row*768+tid],as_type<float>(p[0]),as_type<float>(p[1]))*0.12751743082459868f*pds[lane];
    float s[12],m=-INFINITY;
    for(uint d=0;d<12;++d){s[d]=-INFINITY;if(d>local)continue;
        float4 k=clamp(qkv[(row-d)*768+256+tid],as_type<float>(p[2]),as_type<float>(p[3]))*1.8946361239720118f;
        float4 r=rel[(12-d)*256+tid];float4 ac=q*k,bd=q*r;
        float a=simd_sum(((ac.x+ac.y)+ac.z)+ac.w),b=simd_sum(((bd.x+bd.y)+bd.z)+bd.w);
        s[d]=tanh((a+b)/50.0f)*50.0f;m=max(m,s[d]);}
    float den=0;for(uint d=0;d<12;++d){s[d]=d>local?0:exp(s[d]-m);den+=s[d];}
    float4 v=0;for(int d=11;d>=0;--d)if(uint(d)<=local){float4 a=clamp(qkv[(row-d)*768+512+tid],as_type<float>(p[4]),as_type<float>(p[5]));v=fma(float4(s[d]/den),a,v);}
    v=clamp(v,as_type<float>(p[6]),as_type<float>(p[7]));
    if(p[8])reinterpret_cast<device float4*>(out)[row*256+tid]=v;else out[row*256+tid]=half4(v);
}
kernel void eg2a_conv(device const float4* x [[buffer(0)]],device const float* dw [[buffer(1)]],
    device const float* nw [[buffer(2)]],device half4* out [[buffer(3)]],device const uint* starts [[buffer(4)]],constant uint* p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];float4 v=0;
    for(uint k=0;k<5;++k)if(row-starts[row]+k>=4){uint r=row+k-4;
        float4 a=clamp(x[r*512+tid],as_type<float>(p[0]),as_type<float>(p[1]));
        float4 b=clamp(x[r*512+256+tid],as_type<float>(p[0]),as_type<float>(p[1]));
        float4 w=float4(dw[(tid*4)*5+k],dw[(tid*4+1)*5+k],dw[(tid*4+2)*5+k],dw[(tid*4+3)*5+k]);v=fma(w,a/(1.0f+exp(-b)),v);}
    v=eg2a_rms(v,nw,red,tid,true);v=v/(1.0f+exp(-v));v=clamp(v,as_type<float>(p[2]),as_type<float>(p[3]));
    if(p[4])reinterpret_cast<device float4*>(out)[row*256+tid]=v;else out[row*256+tid]=half4(v);
}
kernel void eg2a_out(device const float* x [[buffer(0)]],device const float* bias [[buffer(1)]],
    device half* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];float v[6],ss=0;for(uint j=0;j<6;++j){uint col=tid+j*256;v[j]=x[row*1536+col]+bias[col];ss+=v[j]*v[j];}
    float inv=rsqrt(eg2a_sum(ss,red,tid)/1536.0f+1e-6f);for(uint j=0;j<6;++j){uint i=row*1536+tid+j*256;
        if(p[0])reinterpret_cast<device float*>(out)[i]=v[j]*inv;else out[i]=half(v[j]*inv);}
}
