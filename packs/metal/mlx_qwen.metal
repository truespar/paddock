// Native affine-checkpoint BF16 operation boundaries. Activations occupy F32
// scheduler buffers but contain BF16 values; recurrent state remains F32.
template<bool Precise=false>
inline float mlx_sigmoid_bf_impl(float x) {
    // Preserve the reference's intermediate BF16 exp/reciprocal/subtraction,
    // even when sigmoid is fused into another operation.
    float e=mlx_bf(Precise?precise::exp(abs(x)):exp(abs(x)));
    float tail=mlx_bf(Precise?precise::divide(1.0f,mlx_bf(1.0f+e)):1.0f/mlx_bf(1.0f+e));
    return x<0 ? tail : mlx_bf(1.0f-tail);
}
inline float mlx_sigmoid_bf(float x) {return mlx_sigmoid_bf_impl<false>(x);}
inline float4 mlx_sigmoid_f32(float4 x) {
    float4 tail=1.0f/(1.0f+exp(abs(x)));
    return select(1.0f-tail,tail,x<0.0f);
}
template<bool Residual, bool Selected>
inline void mlx_rms_impl(device float* x,device const float* delta,device const float* w,
    device const uint* rows,device float* out,constant uint* p,uint row,uint tid,uint lane,uint sg,
    threadgroup float* sums) {
    // Precision boundaries include the non-power-of-two mean division and
    // operation ordering, not only the final BF16 store.
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint n=p[0],source=Selected?rows[row]:row;ulong base=ulong(source)*n;
    uint threads=min(1024u,((n+127)/128)*32);
    if constexpr(Residual) {
        for(uint i=tid;i<n;i+=threads)x[base+i]=mlx_bf(x[base+i]+delta[base+i]);
        threadgroup_barrier(mem_flags::mem_device);
    }
    float sum=0;
    for(uint first=tid*4;first<n;first+=threads*4) {
        for(uint j=0;j<4 && first+j<n;++j) {
            uint i=first+j;float v=x[base+i];
            sum+=v*v;
        }
    }
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    float mean=precise::divide(simd_sum(lane<threads/32?sums[lane]:0.0f),float(n));
    float inv=precise::rsqrt(mean+as_type<float>(p[2]));
    // MLX rounds the normalized activation before applying the BF16 weight.
    // One final cast alone changes roughly a quarter of the oracle outputs.
    for(uint i=tid;i<n;i+=threads)out[ulong(row)*n+i]=mlx_bf(mlx_bf(x[base+i]*inv)*w[i]);
}
kernel void mlx_rms(device float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[32];mlx_rms_impl<false,false>(x,x,w,reinterpret_cast<device const uint*>(x),out,p,r,tid,lane,sg,sums);
}
kernel void mlx_residual_rms(device float* x [[buffer(0)]],device const float* delta [[buffer(1)]],device const float* w [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[32];mlx_rms_impl<true,false>(x,delta,w,reinterpret_cast<device const uint*>(x),out,p,r,tid,lane,sg,sums);
}
kernel void mlx_rms_selected(device float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const uint* rows [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[32];mlx_rms_impl<false,true>(x,x,w,rows,out,p,r,tid,lane,sg,sums);
}
kernel void mlx_residual(device float* x [[buffer(0)]],device const float* delta [[buffer(1)]],constant uint* p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {if(i<p[0])x[i]=mlx_bf(x[i]+delta[i]);}
kernel void mlx_swiglu(device float* gate [[buffer(0)]],device const float* up [[buffer(1)]],constant uint* p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i<p[0])gate[i]=mlx_bf(mlx_bf(gate[i]*mlx_sigmoid_bf(gate[i]))*up[i]);
}
kernel void mlx_swiglu_precise(device float* gate [[buffer(0)]],device const float* up [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])gate[i]=mlx_bf(mlx_bf(gate[i]*mlx_sigmoid_bf_impl<true>(gate[i]))*up[i]);
}
kernel void mlx_attn_gate_precise(device float* x [[buffer(0)]],device const float* qgate [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=mlx_bf(mlx_bf(x[i])*mlx_sigmoid_bf_impl<true>(qgate[ulong(i/256)*512+256+i%256]));
}
kernel void mlx_swiglu_compact(device const bfloat* gate [[buffer(0)]],device const bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]) {
        float g=float(gate[i]);
        out[i]=bfloat(mlx_bf(g*mlx_sigmoid_bf(g))*float(up[i]));
    } else if(i<p[1])out[i]=bfloat(0);
}
// Joined packed projections retain exactly the standalone affine and MLX
// activation rounding, without materializing two F32 activation planes.
kernel void splash_gateup_reduce(device bfloat* scratch [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint K=p[0],N=p[1],M=p[2],parts=p[3];
    if(i>=M*N)return;
    device float* partial=reinterpret_cast<device float*>(scratch+p[4]*K)+p[4]*(K/64);
    float gate=0,up=0;
    for(uint s=0;s<parts;++s){gate+=partial[ulong(s)*M*N+i];up+=partial[ulong(s+parts)*M*N+i];}
    gate=mlx_bf(gate);up=mlx_bf(up);
    out[ulong(p[5])*N+i]=mlx_bf(mlx_bf(gate*mlx_sigmoid_bf(gate))*up);
}
kernel void splash_gateup_bf16(device bfloat* scratch [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1]*p[2])return;
    device bfloat* values=scratch+p[4]*p[0]+2*p[4]*(p[0]/64);
    float gate=float(values[i]),up=float(values[ulong(p[1])*p[2]+i]);
    out[ulong(p[5])*p[1]+i]=mlx_bf(mlx_bf(gate*mlx_sigmoid_bf(gate))*up);
}

template<bool Precise=false>
inline void mlx_dn_conv_impl(device const float* x,device const float* w,device const float* history,
    device const uint* meta,device const uint* bounds,device float* out,constant uint* p,uint i) {
    uint row=i/p[2],d=i%p[2];if(row>=p[3])return;
    uint first=bounds[row*2],slot=meta[row*2],pos=meta[first*2+1];float sum=0;
    for(uint j=0;j<4;++j) {
        int r=int(row)+int(j)-3;
        float v=r>=int(first)?x[ulong(r)*p[2]+d]:
            (pos>0?history[((ulong(p[5])*p[4]+slot)*3+uint(r-int(first)+3))*p[2]+d]:0.0f);
        sum+=v*w[d*4+j];
    }
    sum=mlx_bf(sum);out[i]=mlx_bf(sum*mlx_sigmoid_bf_impl<Precise>(sum));
}
#define MLX_CONV(NAME,PRECISE) \
kernel void NAME(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const float* history [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* bounds [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {mlx_dn_conv_impl<PRECISE>(x,w,history,meta,bounds,out,p,i);}
MLX_CONV(mlx_dn_conv,false)
MLX_CONV(mlx_dn_conv_precise,true)
#undef MLX_CONV
kernel void mlx_dn_qk_norm(device float* qkv [[buffer(0)]],constant uint* p [[buffer(1)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong base=ulong(g.y)*p[2]+g.x*128;float4 v=*reinterpret_cast<device float4*>(qkv+base+lane*4);
    float inv=precise::rsqrt(simd_sum(dot(v,v))/128.0f+1e-6f);
    // Weak scalar operands are converted to the BF16 tensor dtype by MLX.
    float scale=g.x<p[0]?1.0f/128.0f:mlx_bf(0.08838834764831844f);
    *reinterpret_cast<device float4*>(qkv+base+lane*4)=mlx_bf(mlx_bf(v*inv)*scale);
}
template<bool Precise=false>
inline void mlx_dn_gates_impl(device const float* alpha,device const float* beta,
    device const float* a,device const float* dt,device float* gates,constant uint* p,uint i) {
    if(i>=p[1]*p[3])return;
    float v=mlx_bf(alpha[i]+dt[i%p[1]]);
    float e=mlx_bf(Precise?precise::exp(-abs(v)):exp(-abs(v))),u=1.0f+e;
    // Compensated log(1+x), with x in [0,1]. Preserve tiny softplus values
    // when adding one would round them away (Goldberg's log1p identity).
    float logarithm=u==1.0f ? e : e*(log(u)/(u-1.0f));
    float softplus=mlx_bf(max(v,0.0f)+mlx_bf(logarithm));
    gates[i*2]=a[i%p[1]]*softplus;
    gates[i*2+1]=mlx_sigmoid_bf_impl<Precise>(beta[i]);
}
#define MLX_GATES(NAME,PRECISE) \
kernel void NAME(device const float* alpha [[buffer(0)]],device const float* beta [[buffer(1)]],device const float* a [[buffer(2)]],device const float* dt [[buffer(3)]],device float* gates [[buffer(4)]],constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {mlx_dn_gates_impl<PRECISE>(alpha,beta,a,dt,gates,p,i);}
MLX_GATES(mlx_dn_gates,false)
MLX_GATES(mlx_dn_gates_precise,true)
#undef MLX_GATES
kernel void mlx_dn_gated_norm(device float* x [[buffer(0)]],device const float* z [[buffer(1)]],device const float* w [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    ulong base=(ulong(g.y)*p[1]+g.x)*128;
    float4 v=mlx_bf(*reinterpret_cast<device float4*>(x+base+lane*4));
    float sum=v.x*v.x;sum+=v.y*v.y;sum+=v.z*v.z;sum+=v.w*v.w;
    float inv=precise::rsqrt(simd_sum(sum)/128.0f+as_type<float>(p[6]));
    float4 norm=mlx_bf(mlx_bf(v*inv)*(*reinterpret_cast<device const float4*>(w+lane*4)));
    float4 gate=*reinterpret_cast<device const float4*>(z+base+lane*4);
    float4 activated=gate*mlx_sigmoid_f32(gate);
    *reinterpret_cast<device float4*>(x+base+lane*4)=mlx_bf(activated*norm);
}

// One SIMD group owns a value row: vectorized four-cell state per lane keeps
// the token loop short and supplies four times as many independent groups as
// the GGUF tiny-span layout. State updates/checkpoints remain FP32 throughout.
kernel void mlx_dn_recurrent(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
    device float* state [[buffer(2)]],device const uint* spans [[buffer(3)]],device const uint* meta [[buffer(4)]],
    device float* out [[buffer(5)]],device const uint* checkpoints [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint first=spans[g.z*4],count=spans[g.z*4+1],slot=spans[g.z*4+2],h=g.y;
    uint v=g.x*4+tid/32,d=(tid%32)*4,kh=h/(p[1]/p[0]);
    ulong offset=((ulong(p[5])*p[4]+slot)*p[1]+h)*128*128+v*128+d;
    float4 cells=meta[first*2+1]==0?float4(0):*reinterpret_cast<device float4*>(state+offset);
    for(uint r=first;r<first+count;++r) {
        float decay=precise::exp(gates[(r*p[1]+h)*2]),beta=gates[(r*p[1]+h)*2+1];
        ulong base=ulong(r)*p[2];
        float4 key=*reinterpret_cast<device const float4*>(qkv+base+p[0]*128+kh*128+d);
        float4 query=*reinterpret_cast<device const float4*>(qkv+base+kh*128+d);
        cells*=decay;
        float dot_key=cells.x*key.x;dot_key=fma(cells.y,key.y,dot_key);dot_key=fma(cells.z,key.z,dot_key);dot_key=fma(cells.w,key.w,dot_key);
        float delta=(qkv[base+p[0]*256+h*128+v]-simd_sum(dot_key))*beta;
        cells=fma(key,float4(delta),cells);
        float dot_query=cells.x*query.x;dot_query=fma(cells.y,query.y,dot_query);dot_query=fma(cells.z,query.z,dot_query);dot_query=fma(cells.w,query.w,dot_query);
        float value=simd_sum(dot_query);
        if(tid%32==0)out[(ulong(r)*p[1]+h)*128+v]=mlx_bf(value);
        if(checkpoints[r]!=0) {
            ulong cache=((ulong(p[5])*p[4]+checkpoints[r]-1)*p[1]+h)*128*128+v*128+d;
            *reinterpret_cast<device float4*>(state+cache)=cells;
        }
    }
    *reinterpret_cast<device float4*>(state+offset)=cells;
}

// Share each input vector across independent value rows in registers. Unlike
// changing the reduction width or using a WY transform, this keeps every
// row's four-cell FMA chain and 32-lane reduction identical to the reference
// above. Checkpoint destinations and FP32 state ownership are unchanged.
template<uint V,bool Full=false>
inline void mlx_dn_recurrent_values(device const float* qkv,device const float* gates,
    device float* state,device const uint* spans,device const uint* meta,
    device float* out,device const uint* checkpoints,constant uint* p,uint3 g,uint tid) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint first=spans[g.z*4],count=spans[g.z*4+1],slot=spans[g.z*4+2],h=g.y;
    uint v=(g.x*4+tid/32)*V,d=(tid%32)*4,kh=h/(p[1]/p[0]);
    ulong offset=((ulong(p[5])*p[4]+slot)*p[1]+h)*128*128+v*128+d;
    float4 cells[V];
    #pragma unroll
    for(uint j=0;j<V;++j)cells[j]=meta[first*2+1]==0?float4(0):*reinterpret_cast<device float4*>(state+offset+j*128);
    for(uint r=first;r<first+count;++r) {
        float decay=precise::exp(gates[(r*p[1]+h)*2]),beta=gates[(r*p[1]+h)*2+1];
        ulong base=ulong(r)*p[2];
        float4 key=*reinterpret_cast<device const float4*>(qkv+base+p[0]*128+kh*128+d);
        float4 query=*reinterpret_cast<device const float4*>(qkv+base+kh*128+d);
        #pragma unroll
        for(uint j=0;j<V;++j) {
            cells[j]*=decay;
            float dot_key=cells[j].x*key.x;dot_key=fma(cells[j].y,key.y,dot_key);dot_key=fma(cells[j].z,key.z,dot_key);dot_key=fma(cells[j].w,key.w,dot_key);
            float delta=(qkv[base+p[0]*256+h*128+v+j]-simd_sum(dot_key))*beta;
            cells[j]=fma(key,float4(delta),cells[j]);
            float dot_query=cells[j].x*query.x;dot_query=fma(cells[j].y,query.y,dot_query);dot_query=fma(cells[j].z,query.z,dot_query);dot_query=fma(cells[j].w,query.w,dot_query);
            float value=simd_sum(dot_query);
            if(tid%32==0)out[(ulong(r)*p[1]+h)*128+v+j]=Full?value:mlx_bf(value);
            if(checkpoints[r]!=0) {
                ulong cache=((ulong(p[5])*p[4]+checkpoints[r]-1)*p[1]+h)*128*128+(v+j)*128+d;
                *reinterpret_cast<device float4*>(state+cache)=cells[j];
            }
        }
    }
    #pragma unroll
    for(uint j=0;j<V;++j)*reinterpret_cast<device float4*>(state+offset+j*128)=cells[j];
}
#define MLX_DN_VALUES(NAME,V) \
kernel void NAME(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]], \
device float* state [[buffer(2)]],device const uint* spans [[buffer(3)]],device const uint* meta [[buffer(4)]], \
device float* out [[buffer(5)]],device const uint* checkpoints [[buffer(6)]],constant uint* p [[buffer(7)]], \
uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
mlx_dn_recurrent_values<V>(qkv,gates,state,spans,meta,out,checkpoints,p,g,tid);}
MLX_DN_VALUES(mlx_dn_recurrent_quad,4)
#undef MLX_DN_VALUES

kernel void bonsai_dn_recurrent(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
    device float* state [[buffer(2)]],device const uint* spans [[buffer(3)]],device const uint* meta [[buffer(4)]],
    device float* out [[buffer(5)]],device const uint* checkpoints [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    mlx_dn_recurrent_values<4,true>(qkv,gates,state,spans,meta,out,checkpoints,p,g,tid);
}

// Eight value rows per SIMD group, four lanes per row. Each lane retains
// eight of the reference's float4 chains. The local pairwise tree followed
// by two row-local shuffles reproduces the reference's ascending reduction
// on the qualified Apple10 compiler; do not elect on other GPU families.
#ifndef PADDOCK_APPLE9
template<bool Full=false>
inline void mlx_dn_packed(device const float* qkv,device const float* gates,
    device float* state,device const uint* spans,device const uint* meta,
    device float* out,device const uint* checkpoints,constant uint* p,uint3 g,uint tid) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint first=spans[g.z*4],count=spans[g.z*4+1],slot=spans[g.z*4+2],h=g.y;
    uint v=g.x*32+tid/4,d=(tid%4)*32,kh=h/(p[1]/p[0]);
    ulong offset=((ulong(p[5])*p[4]+slot)*p[1]+h)*128*128+v*128+d;
    float4 cells[8];
    #pragma unroll
    for(uint j=0;j<8;++j)cells[j]=meta[first*2+1]==0?float4(0):*reinterpret_cast<device float4*>(state+offset+j*4);
    for(uint r=first;r<first+count;++r) {
        float decay=precise::exp(gates[(r*p[1]+h)*2]),beta=gates[(r*p[1]+h)*2+1];
        ulong base=ulong(r)*p[2];
        float4 keys[8];float part[8];
        #pragma unroll
        for(uint j=0;j<8;++j) {
            keys[j]=*reinterpret_cast<device const float4*>(qkv+base+p[0]*128+kh*128+d+j*4);
            cells[j]*=decay;
            float dot_key=cells[j].x*keys[j].x;
            dot_key=fma(cells[j].y,keys[j].y,dot_key);dot_key=fma(cells[j].z,keys[j].z,dot_key);dot_key=fma(cells[j].w,keys[j].w,dot_key);
            part[j]=dot_key;
        }
        float sum=((part[0]+part[1])+(part[2]+part[3]))+((part[4]+part[5])+(part[6]+part[7]));
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
        float delta=(qkv[base+p[0]*256+h*128+v]-sum)*beta;
        #pragma unroll
        for(uint j=0;j<8;++j) {
            cells[j]=fma(keys[j],float4(delta),cells[j]);
            float4 query=*reinterpret_cast<device const float4*>(qkv+base+kh*128+d+j*4);
            float dot_query=cells[j].x*query.x;
            dot_query=fma(cells[j].y,query.y,dot_query);dot_query=fma(cells[j].z,query.z,dot_query);dot_query=fma(cells[j].w,query.w,dot_query);
            part[j]=dot_query;
        }
        sum=((part[0]+part[1])+(part[2]+part[3]))+((part[4]+part[5])+(part[6]+part[7]));
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
        if(tid%4==0)out[(ulong(r)*p[1]+h)*128+v]=Full?sum:mlx_bf(sum);
        if(checkpoints[r]!=0) {
            ulong cache=((ulong(p[5])*p[4]+checkpoints[r]-1)*p[1]+h)*128*128+v*128+d;
            #pragma unroll
            for(uint j=0;j<8;++j)*reinterpret_cast<device float4*>(state+cache+j*4)=cells[j];
        }
    }
    #pragma unroll
    for(uint j=0;j<8;++j)*reinterpret_cast<device float4*>(state+offset+j*4)=cells[j];
}
#define MLX_DN_PACKED(NAME,FULL) \
kernel void NAME(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]], \
device float* state [[buffer(2)]],device const uint* spans [[buffer(3)]],device const uint* meta [[buffer(4)]], \
device float* out [[buffer(5)]],device const uint* checkpoints [[buffer(6)]],constant uint* p [[buffer(7)]], \
uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
mlx_dn_packed<FULL>(qkv,gates,state,spans,meta,out,checkpoints,p,g,tid);}
MLX_DN_PACKED(mlx_dn_recurrent_packed,false)
MLX_DN_PACKED(bonsai_dn_recurrent_packed,true)
#undef MLX_DN_PACKED
#endif

// Transactional counterpart of mlx_dn_recurrent. Preserve its head mapping,
// SIMD reduction, explicit FMA and BF16 output boundary; the GGUF verifier
// has different arithmetic. Store a rank-one update, never a full state per
// candidate, and leave the committed state untouched until acceptance.
kernel void mlx_dn_verify(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
    device const float* state [[buffer(2)]],device const uint* spans [[buffer(3)]],device const uint* meta [[buffer(4)]],
    device float* out [[buffer(5)]],device float* updates [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint first=spans[g.z*4],count=spans[g.z*4+1],slot=spans[g.z*4+2],h=g.y;
    uint v=g.x*4+tid/32,d=(tid%32)*4,kh=h/(p[1]/p[0]);
    ulong offset=((ulong(p[5])*p[4]+slot)*p[1]+h)*128*128+v*128+d;
    float4 cells=meta[first*2+1]==0?float4(0):*reinterpret_cast<device const float4*>(state+offset);
    for(uint r=first;r<first+count;++r) {
        float decay=precise::exp(gates[(r*p[1]+h)*2]),beta=gates[(r*p[1]+h)*2+1];
        ulong base=ulong(r)*p[2];
        float4 key=*reinterpret_cast<device const float4*>(qkv+base+p[0]*128+kh*128+d);
        float4 query=*reinterpret_cast<device const float4*>(qkv+base+kh*128+d);
        cells*=decay;
        float dot_key=cells.x*key.x;dot_key=fma(cells.y,key.y,dot_key);dot_key=fma(cells.z,key.z,dot_key);dot_key=fma(cells.w,key.w,dot_key);
        float delta=(qkv[base+p[0]*256+h*128+v]-simd_sum(dot_key))*beta;
        cells=fma(key,float4(delta),cells);
        float dot_query=cells.x*query.x;dot_query=fma(cells.y,query.y,dot_query);dot_query=fma(cells.z,query.z,dot_query);dot_query=fma(cells.w,query.w,dot_query);
        float value=simd_sum(dot_query);
        ulong u=((ulong(p[5])*p[6]+r)*p[1]+h)*257;
        if(tid%32==0) {out[(ulong(r)*p[1]+h)*128+v]=mlx_bf(value);updates[u+128+v]=delta;}
        if(g.x==0 && tid<32) {
            // u is not float4-aligned (257-float records).
            updates[u+d]=key.x;updates[u+d+1]=key.y;updates[u+d+2]=key.z;updates[u+d+3]=key.w;
        }
        if(g.x==0 && tid==0)updates[u+256]=decay;
    }
}

kernel void mlx_dn_verify_commit(device const float* updates [[buffer(0)]],device float* state [[buffer(1)]],
    device const uint* spans [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint first=spans[g.z*4],count=spans[g.z*4+1],slot=spans[g.z*4+2],layer=g.y/p[0],h=g.y%p[0];
    uint v=g.x*4+tid/32,d=(tid%32)*4;
    ulong offset=((ulong(layer)*p[1]+slot)*p[0]+h)*128*128+v*128+d;
    float4 cells=*reinterpret_cast<device const float4*>(state+offset);
    for(uint r=first;r<first+count;++r) {
        ulong u=((ulong(layer)*p[2]+r)*p[0]+h)*257;
        float4 key=float4(updates[u+d],updates[u+d+1],updates[u+d+2],updates[u+d+3]);
        cells*=updates[u+256];
        cells=fma(key,float4(updates[u+128+v]),cells);
    }
    *reinterpret_cast<device float4*>(state+offset)=cells;
}

// Head-256 normalization rounds before and after partial split-half RoPE.
// The precise frequency and explicit first-product FMA preserve the pinned
// GPU reference's rotation at cancellation/tie boundaries. Global relaxed
// math or the opposite contraction changes BF16 results at longer positions.
template<bool ReciprocalPower = false>
inline float mlx_rotary(device const float* x,device const float* w,device const uint* pos,
    ulong base,uint d,float inv,constant uint* p,uint row) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float v=mlx_bf(mlx_bf(x[base+d]*inv)*w[d]);
    if(d<p[5]) {
        uint j=d%(p[5]/2),other=d<p[5]/2?d+p[5]/2:d-p[5]/2;
        uint axis=j%3==1 && j<33?1:(j%3==2 && j<30?2:0);
        float frequency=precise::exp2(-float(j)/float(p[5]/2)*precise::log2(as_type<float>(p[3])));
        if constexpr(ReciprocalPower)frequency=precise::divide(1.0f,precise::pow(as_type<float>(p[3]),float(j)/float(p[5]/2)));
        float angle=float(pos[row*4+axis])*frequency;
        float partner=mlx_bf(mlx_bf(x[base+other]*inv)*w[other]);
        float cosine=fast::cos(angle),sine=fast::sin(angle);
        v=d<p[5]/2?fma(v,cosine,-partner*sine):fma(partner,sine,v*cosine);
    }
    return mlx_bf(v);
}
kernel void mlx_qnorm_rope(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const uint* pos [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong src=(ulong(g.y)*p[0]+g.x)*512,dst=(ulong(g.y)*p[0]+g.x)*256;float sum=0;
    for(uint d=lane;d<256;d+=32)sum+=x[src+d]*x[src+d];
    float inv=precise::rsqrt(simd_sum(sum)/256.0f+as_type<float>(p[4]));
    for(uint d=lane;d<256;d+=32)out[dst+d]=mlx_rotary(x,w,pos,src,d,inv,p,g.y);
}
kernel void mlx_knorm_store(device const float* x [[buffer(0)]],device const float* v [[buffer(1)]],device const float* w [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device bfloat* keys [[buffer(5)]],device bfloat* values [[buffer(6)]],
    device const uint* pos [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong src=(ulong(g.y)*p[1]+g.x)*256;uint slot=meta[g.y*2],position=meta[g.y*2+1];
    uint physical=pages[slot*p[2]+position/16]*16+position%16;ulong dst=(ulong(physical)*p[1]+g.x)*256;
    float sum=0;for(uint d=lane;d<256;d+=32)sum+=x[src+d]*x[src+d];
    float inv=precise::rsqrt(simd_sum(sum)/256.0f+as_type<float>(p[4]));
    for(uint d=lane;d<256;d+=32){keys[dst+d]=bfloat(mlx_rotary(x,w,pos,src,d,inv,p,g.y));values[dst+d]=bfloat(v[src+d]);}
}
// MLX's head-256 RMS reduction assigns four consecutive values to each of
// 64 threads, then reduces the two SIMD partials. A strided 32-thread sum
// changes BF16 rounding even outside the rotary dimensions. Keep this
// contract independent of prompt size, other requests and physical KV pages.
// Algorithm reference: MLX v0.32.3 rms_norm.metal (existing MLX notice).
inline float mlx_head256_inverse(device const float* x,ulong base,float eps,
    uint tid,uint lane,uint sg,threadgroup float* partial) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float sum=0;
    for(uint j=0;j<4;++j){float v=x[base+tid*4+j];sum+=v*v;}
    sum=simd_sum(sum);
    if(lane==0)partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=simd_sum(lane<2?partial[lane]:0.0f);
    return precise::rsqrt(total/256.0f+eps);
}
kernel void mlx_qnorm_rope_contiguous(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const uint* pos [[buffer(2)]],
    device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[2];
    ulong src=(ulong(g.y)*p[0]+g.x)*512,dst=(ulong(g.y)*p[0]+g.x)*256;
    float inv=mlx_head256_inverse(x,src,as_type<float>(p[4]),tid,lane,sg,partial);
    for(uint j=0;j<4;++j){uint d=tid*4+j;out[dst+d]=mlx_rotary<true>(x,w,pos,src,d,inv,p,g.y);}
}
kernel void mlx_knorm_store_contiguous(device const float* x [[buffer(0)]],device const float* v [[buffer(1)]],device const float* w [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device bfloat* keys [[buffer(5)]],device bfloat* values [[buffer(6)]],
    device const uint* pos [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[2];
    ulong src=(ulong(g.y)*p[1]+g.x)*256;uint slot=meta[g.y*2],position=meta[g.y*2+1];
    uint physical=pages[slot*p[2]+position/16]*16+position%16;ulong dst=(ulong(physical)*p[1]+g.x)*256;
    float inv=mlx_head256_inverse(x,src,as_type<float>(p[4]),tid,lane,sg,partial);
    for(uint j=0;j<4;++j){uint d=tid*4+j;keys[dst+d]=bfloat(mlx_rotary<true>(x,w,pos,src,d,inv,p,g.y));values[dst+d]=bfloat(v[src+d]);}
}
kernel void mlx_attn_gate(device float* x [[buffer(0)]],device const float* qgate [[buffer(1)]],constant uint* p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=mlx_bf(mlx_bf(x[i])*mlx_sigmoid_bf(qgate[ulong(i/256)*512+256+i%256]));
}
kernel void mlx_attention_query(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],constant uint* p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {if(i<(p[2]+32)*p[0])out[i]=bfloat(i<p[2]*p[0]?x[i]:0.0f);}

// BK16 leaves room for MPP's implicit staging under shader validation on
// Apple10. BK32 reached 45,824 bytes; the hardware limit is 32 KiB.
#define MLX_PREFILL(NAME,SPLITS,INTERLEAVED,INDEXED) \
kernel void NAME(device bfloat* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]], \
device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]], \
device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]], \
uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup bfloat kv[16*256],probability[32*16]; \
threadgroup float scores[32*16],maximum[32],denominator[32],correction[32]; \
qwen_prefill_tile<SPLITS,bfloat,16,false,INTERLEAVED,INDEXED>(q,k,v,meta,pages,out,limits,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid, \
kv,probability,scores,maximum,denominator,correction,g.z);}
MLX_PREFILL(mlx_attention_prefill,1,false,false)
// One partition preserves the ordinary causal block order while consuming
// physical BF16 pages directly. Do not use the interleaved multi-partition
// election here: that would change the existing softmax reduction tree.
MLX_PREFILL(mlx_attention_prefill_direct,1,true,false)
MLX_PREFILL(mlx_attention_prefill_indexed,1,true,true)
MLX_PREFILL(mlx_attention_prefill_split,4,false,false)
MLX_PREFILL(splash_attention_prefill,4,true,false)
#undef MLX_PREFILL

// Keep the 16-token softmax contract, but share a physical page across six
// query heads. Eight positions x six GQA heads form one 48-row MPP tile.
#ifndef PADDOCK_APPLE9
kernel void mlx_attention_prefill_gqa(device bfloat* q [[buffer(0)]],device const bfloat* kc [[buffer(1)]],device const bfloat* vc [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],
    device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]],
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint tile=g.y/4,offset=g.y%4*8,count=tiles[2*tile+1];if(offset>=count)return;
    uint first=tiles[2*tile]+offset;count=min(8u,count-offset);
    uint slot=meta[2*first],last=limits[first+count-1],kh=g.x,kvwidth=p[1]*256;
    threadgroup float scores[48*16],maximum[48],denominator[48],correction[48];
    threadgroup bfloat probability[48*16];
    auto tq=tensor(q+(ulong(g.y)*p[1]+kh)*48*256,extents<int,256,48>());
    auto tp=tensor(probability,extents<int,16,48>());
    auto ts=tensor(scores,extents<int,16,48>());
    constexpr auto qkd=matmul2d_descriptor(48,16,256,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pvd=matmul2d_descriptor(48,256,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroups<8>> qk;
    matmul2d<pvd,execution_simdgroups<8>> pv;
    auto page=tensor(const_cast<device bfloat*>(vc),dextents<int,2>{256,16},array<int,2>{1,int(kvwidth)});
    auto accum=pv.get_destination_cooperative_tensor<decltype(tp),decltype(page),float>();
    for(ushort i=0;i<accum.get_capacity();++i)accum[i]=0;
    if(tid<48){maximum[tid]=-INFINITY;denominator[tid]=0;}
    for(uint base=0;base<=last;base+=16) {
        uint physical=pages[slot*p[2]+base/16]*16;
        auto key=tensor(const_cast<device bfloat*>(kc)+ulong(physical)*kvwidth+kh*256,
            dextents<int,2>{256,int(min(16u,last-base+1))},array<int,2>{1,int(kvwidth)});
        auto score=qk.get_destination_cooperative_tensor<decltype(tq),decltype(key),float>();
        for(ushort i=0;i<score.get_capacity();++i)score[i]=0;
        qk.run(tq,key,score);score.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<48*4) {
            uint row=tid/4,lane=tid%4,local=row/6,pos=local<count?limits[first+local]:0;
            float high=maximum[row];
            for(uint j=lane;j<16;j+=4)if(local<count && base+j<=pos)high=max(high,scores[row*16+j]*as_type<float>(p[3]));
            high=max(high,simd_shuffle_xor(high,1));high=max(high,simd_shuffle_xor(high,2));
            float old=isfinite(maximum[row])?exp(maximum[row]-high):0.0f,sum=0;
            for(uint j=lane;j<16;j+=4) {
                float prob=local<count && base+j<=pos?exp(scores[row*16+j]*as_type<float>(p[3])-high):0.0f;
                probability[row*16+j]=bfloat(prob);sum+=prob;
            }
            sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
            if(lane==0){maximum[row]=high;correction[row]=old;denominator[row]=local<count?denominator[row]*old+sum:1.0f;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto value=tensor(const_cast<device bfloat*>(vc)+ulong(physical)*kvwidth+kh*256,
            dextents<int,2>{256,int(min(16u,last-base+1))},array<int,2>{1,int(kvwidth)});
        auto product=pv.get_destination_cooperative_tensor<decltype(tp),decltype(value),float>();
        for(ushort i=0;i<product.get_capacity();++i)product[i]=0;
        pv.run(tp,value,product);
        #pragma unroll
        for(ushort i=0;i<product.get_capacity();++i)if(product.is_valid_element(i))
            accum[i]=accum[i]*correction[product.get_multidimensional_index(i)[1]]+product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(ushort i=0;i<accum.get_capacity();++i)if(accum.is_valid_element(i)) {
        auto ij=accum.get_multidimensional_index(i);uint local=ij[1]/6,head=kh*6+ij[1]%6;
        if(local<count)out[(ulong(first+local)*p[0]+head)*256+ij[0]]=accum[i]/denominator[ij[1]];
    }
}
#endif
