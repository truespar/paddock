// Native Bonsai: packed 2-bit ternary + FP16 group-128 scale; F32 decoder.
// Original kernels. Hadamard mathematics follows the checkpoint's explicit
// normalized Sylvester contract; no Python or expanded weight matrices.
inline float bonsai_weight(device const uchar* w,uint K,uint N,ulong i) {
    uint word=reinterpret_cast<device const uint*>(w)[i/16];
    int trit=int((word>>((i%16)*2))&3)-1;
    return float(trit)*float(reinterpret_cast<device const half*>(w+ulong(K)*N/4)[i/128]);
}

// Four stripes per thread. Five butterfly stages are SIMD shuffles, three
// exchange through 4 KiB shared memory, and the final two stay in registers.
inline float4 bonsai_fwht(float4 value,uint tid,threadgroup float* exchange) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    for(uint stride=1;stride<32;stride*=2) {
        float4 peer=simd_shuffle_xor(value,stride);
        value=(tid&stride)?peer-value:value+peer;
    }
    for(uint stride=32;stride<256;stride*=2) {
        for(uint j=0;j<4;++j)exchange[j*256+tid]=value[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 peer;for(uint j=0;j<4;++j)peer[j]=exchange[j*256+(tid^stride)];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        value=(tid&stride)?peer-value:value+peer;
    }
    float4 next(value.x+value.y,value.x-value.y,value.z+value.w,value.z-value.w);
    return float4(next.x+next.z,next.y+next.w,next.x-next.z,next.y-next.w)*(1.0f/32.0f);
}
kernel void bonsai_rotate(device const float* x [[buffer(0)]],device const float* signs [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float exchange[1024];
    uint col=g.x*1024+tid;ulong row=ulong(g.y)*p[0];float4 v;
    for(uint j=0;j<4;++j)v[j]=x[row+col+j*256]*signs[col+j*256];
    v=bonsai_fwht(v,tid,exchange);
    for(uint j=0;j<4;++j)out[row+col+j*256]=v[j];
}
kernel void bonsai_embed(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device const float* signs [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float exchange[1024];
    uint col=g.x*1024+tid;ulong row=ulong(ids[g.y])*p[0];float4 v;
    for(uint j=0;j<4;++j)v[j]=float(half(bonsai_weight(w,p[0],p[1],row+col+j*256)));
    v=bonsai_fwht(v,tid,exchange);
    for(uint j=0;j<4;++j)out[ulong(g.y)*p[0]+col+j*256]=float(half(v[j]*signs[col+j*256]));
}
kernel void bonsai_a(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=-precise::exp(x[i]);
}
kernel void bonsai_mv(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint K=p[0],N=p[1],n=g.x*4+tid/32,lane=tid%32;
    if(n>=N || g.y>=p[2])return;
    float4 sums=0;
    for(uint k=lane*4;k<K;k+=128) {
        float4 a=*reinterpret_cast<device const float4*>(x+ulong(g.y)*K+k),b;
        if(p[3]==0)b=*reinterpret_cast<device const float4*>(w+(ulong(n)*K+k)*4);
        else {ulong i=ulong(n)*K+k;uint word=reinterpret_cast<device const uint*>(w)[i/16];
            float scale=float(reinterpret_cast<device const half*>(w+ulong(K)*N/4)[i/128]);
            b=(float4((uint4(word)>>uint4((i%16)*2,(i%16)*2+2,(i%16)*2+4,(i%16)*2+6))&3)-1.0f)*scale;}
        sums=fma(a,b,sums);
    }
    float sum=simd_sum((sums.x+sums.y)+(sums.z+sums.w));
    if(lane==0)out[ulong(g.y)*N+n]=sum;
}

// A lane owns a complete packed word, so its 16 codes load once. Eight lanes
// own one scale group. Concurrent requests reuse the decoded word in registers
// instead of launching independent full-weight walks for each request.
template<uint R,bool Full=false,bool Bits=false>
inline void bonsai_vectors(device const uchar* w,device const float* x,device float* out,
    uint K,uint N,uint M,uint type,uint2 g,uint tid) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint n=g.x*16+tid/8,lane=tid%8,first=g.y*R;
    if(n>=N)return;
    float4 accum[R];for(uint r=0;r<R;++r)accum[r]=0;
    for(uint k=lane*16;k<K;k+=128) {
        float4 weights[4];
        if(!Full && type==0) {
            for(uint j=0;j<4;++j)weights[j]=*reinterpret_cast<device const float4*>(w+(ulong(n)*K+k+j*4)*4);
        } else {
            ulong i=ulong(n)*K+k;uint word=reinterpret_cast<device const uint*>(w)[i/16];
            float scale=float(reinterpret_cast<device const half*>(w+ulong(K)*N/4)[i/128]);
            for(uint j=0;j<4;++j) {
                uint4 codes=(uint4(word)>>uint4(j*8,j*8+2,j*8+4,j*8+6))&3;
                if constexpr(Bits) {
                    // Validated ternary codes select {-scale, signed zero,
                    // scale} by bits. No integer-to-float conversion or
                    // per-weight scale multiplication; preserve signed zero.
                    uint4 value=select(uint4(as_type<uint>(scale)),uint4(as_type<uint>(scale)&0x80000000u),codes==1);
                    weights[j]=as_type<float4>(value^select(uint4(0),uint4(0x80000000u),codes==0));
                } else { weights[j]=(float4(codes)-1.0f)*scale; }
            }
        }
        for(uint r=0;r<R;++r)if(Full || first+r<M) {
            for(uint j=0;j<4;++j) {
                float4 value=*reinterpret_cast<device const float4*>(x+ulong(first+r)*K+k+j*4);
                accum[r]=fma(value,weights[j],accum[r]);
            }
        }
    }
    for(uint r=0;r<R;++r) {
        float4 v=accum[r];float sum=kquant_sum<8>((v.x+v.y)+(v.z+v.w));
        if(lane==0 && (Full || first+r<M))out[ulong(first+r)*N+n]=sum;
    }
}
#define BONSAI_VECTORS(R) \
kernel void bonsai_vectors##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {bonsai_vectors<R>(w,x,out,p[0],p[1],p[2],p[3],g,tid);}
BONSAI_VECTORS(1)
BONSAI_VECTORS(4)
#undef BONSAI_VECTORS
// Only packed matrices with exactly R live rows select these entry points.
// Hoist the dtype and row predicates out of the inner FMA loop without
// changing operand precision, accumulation order or reduction topology.
#define BONSAI_FULL(R) \
kernel void bonsai_full##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {bonsai_vectors<R,true>(w,x,out,p[0],p[1],p[2],p[3],g,tid);}
BONSAI_FULL(1)
BONSAI_FULL(2)
BONSAI_FULL(3)
BONSAI_FULL(4)
#undef BONSAI_FULL
#define BONSAI_BITS(R) \
kernel void bonsai_bits##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {bonsai_vectors<R,true,true>(w,x,out,p[0],p[1],p[2],p[3],g,tid);}
BONSAI_BITS(1)
BONSAI_BITS(2)
BONSAI_BITS(3)
BONSAI_BITS(4)
#undef BONSAI_BITS

// Concatenate independent Q/K/V or gate/up dispatch domains, not their
// buffers. Narrow projections occupy the GPU alongside their wider peers.
#define BONSAI_MULTI(NAME,R,BITS) \
kernel void NAME(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
uint N=p[1],nx=(N+15)/16;device const uchar* w=w0;device float* out=o0; \
if(g.x>=nx){g.x-=nx;N=p[2];nx=(N+15)/16;w=w1;out=o1; \
if(g.x>=nx){g.x-=nx;N=p[3];w=w2;out=o2;}} \
bonsai_vectors<R,true,BITS>(w,x,out,p[0],N,p[4],0x102,g,tid);}
BONSAI_MULTI(bonsai_multi1,1,false)
BONSAI_MULTI(bonsai_multi2,2,false)
BONSAI_MULTI(bonsai_multi3,3,false)
BONSAI_MULTI(bonsai_multi4,4,false)
BONSAI_MULTI(bonsai_bits_multi1,1,true)
BONSAI_MULTI(bonsai_bits_multi2,2,true)
BONSAI_MULTI(bonsai_bits_multi3,3,true)
BONSAI_MULTI(bonsai_bits_multi4,4,true)
#undef BONSAI_MULTI
// Recurrent input: two independent packed planes and two small F32 gates.
// Schedule their disjoint output domains together. F32 gates read the
// original activation; only packed weights consume the rotated activation.
#define BONSAI_MIXED(R) \
kernel void bonsai_mixed##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]],device const uchar* w3 [[buffer(3)]], \
device const float* rotated [[buffer(4)]],device const float* x [[buffer(5)]],device float* o0 [[buffer(6)]],device float* o1 [[buffer(7)]],device float* o2 [[buffer(8)]],device float* o3 [[buffer(9)]], \
constant uint* p [[buffer(10)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
uint a=(p[1]+15)/16,b=(p[2]+15)/16,c=(p[3]+15)/16; \
if(g.x<a){bonsai_vectors<R,true,true>(w0,rotated,o0,p[0],p[1],p[5],0x102,g,tid);return;}g.x-=a; \
if(g.x<b){bonsai_vectors<R,true,true>(w1,rotated,o1,p[0],p[2],p[5],0x102,g,tid);return;}g.x-=b; \
if(g.x<c){bonsai_vectors<R>(w2,x,o2,p[0],p[3],p[5],0,g,tid);return;}g.x-=c; \
bonsai_vectors<R>(w3,x,o3,p[0],p[4],p[5],0,g,tid);}
BONSAI_MIXED(1)
BONSAI_MIXED(2)
BONSAI_MIXED(3)
BONSAI_MIXED(4)
#undef BONSAI_MIXED
template<uint BM,uint BN=16,uint BK=128,bool Relaxed=false,typename W=float,uint Padding=4>
inline void bonsai_mm(device const uchar* w,device float* x,device float* out,constant uint* p,
    uint2 g,uint tid,threadgroup W* weights) {
    uint K=p[0],N=p[1],M=p[2],n=g.x*BN,m=g.y*BM;
    auto input=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK+Padding});
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,Relaxed,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.template get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    for(uint base=0;base<K;base+=BK) {
        // A thread owns one packed word: load it once, consume all 16 trits.
        for(uint index=tid;index<BN*BK/16;index+=128) {
        uint col=n+index/(BK/16),k=base+(index%(BK/16))*16;
        if(p[3]==0) {
            for(uint j=0;j<16;++j)weights[(index/(BK/16))*(BK+Padding)+(index%(BK/16))*16+j]=col<N?
                reinterpret_cast<device const float*>(w)[ulong(col)*K+k+j]:0.0f;
        } else {
            uint word=col<N?reinterpret_cast<device const uint*>(w)[(ulong(col)*K+k)/16]:0;
            float scale=col<N?float(reinterpret_cast<device const half*>(w+ulong(K)*N/4)[(ulong(col)*K+k)/128]):0.0f;
            for(uint j=0;j<16;++j)weights[(index/(BK/16))*(BK+Padding)+(index%(BK/16))*16+j]=float(int((word>>(j*2))&3)-1)*scale;
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a=input.slice(base,m);op.run(a,b,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    c.store(output.slice(n,m));
}
#define BONSAI_MM(BM) \
kernel void bonsai_mm##BM(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float weights[16*132];bonsai_mm<BM>(w,x,out,p,g,tid,weights);}
BONSAI_MM(32)
BONSAI_MM(64)
#undef BONSAI_MM
// Wide output planes benefit from a 32-column tile. BK=64 also leaves
// room for Metal shader-validation instrumentation within the 32 KiB limit.
// Narrow projections retain the original 16-column occupancy contract.
#define BONSAI_TILE(M,N,K) \
kernel void bonsai_tile##M##x##N##x##K(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float weights[N*(K+4)];bonsai_mm<M,N,K>(w,x,out,p,g,tid,weights);}
BONSAI_TILE(32,32,64)
BONSAI_TILE(64,32,64)
#undef BONSAI_TILE
// M5 prompt-only arithmetic. Dispatch roles, not the number of neighbouring
// requests, choose this contract. Decoder state and operands remain F32.
#define BONSAI_PREFILL(M) \
kernel void bonsai_prefill##M(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float weights[32*68];bonsai_mm<M,32,64,true>(w,x,out,p,g,tid,weights);}
BONSAI_PREFILL(32)
BONSAI_PREFILL(64)
BONSAI_PREFILL(16)
BONSAI_PREFILL(128)
#undef BONSAI_PREFILL
#define BONSAI_PREHALF(M,K) \
kernel void bonsai_prehalf##M##x##K(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
constexpr uint Pad=K==256?0:4;threadgroup half weights[32*(K+Pad)];bonsai_mm<M,32,K,true,half,Pad>(w,x,out,p,g,tid,weights);}
BONSAI_PREHALF(32,128)
BONSAI_PREHALF(64,128)
BONSAI_PREHALF(128,128)
// Longer prompt tiles amortize staging/barriers over two exact F16 scale
// groups. Decoder and short prompt elections keep their prior tile.
// No padding on the deep tile: even the doubled shader-instrumentation
// reservation must fit the 32 KiB threadgroup limit.
BONSAI_PREHALF(128,256)
#undef BONSAI_PREHALF

template<bool Residual,bool Selected>
inline void bonsai_rms_impl(device float* x,device const float* delta,device const float* w,
    device const uint* rows,device float* out,constant uint* p,uint row,uint tid,uint lane,uint sg,threadgroup float* sums) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint n=p[0],source=Selected?rows[row]:row,threads=min(1024u,((n+127)/128)*32);ulong base=ulong(source)*n;
    if constexpr(Residual) {for(uint i=tid;i<n;i+=threads)x[base+i]+=delta[base+i];threadgroup_barrier(mem_flags::mem_device);}
    float sum=0;for(uint first=tid*4;first<n;first+=threads*4)for(uint j=0;j<4 && first+j<n;++j){float v=x[base+first+j];sum+=v*v;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    float inv=precise::rsqrt(precise::divide(simd_sum(lane<threads/32?sums[lane]:0.0f),float(n))+as_type<float>(p[2]));
    for(uint i=tid;i<n;i+=threads)out[ulong(row)*n+i]=(x[base+i]*inv)*w[i];
}
kernel void bonsai_rms(device float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[32];bonsai_rms_impl<false,false>(x,x,w,reinterpret_cast<device const uint*>(x),out,p,r,tid,lane,sg,sums);
}
kernel void bonsai_residual_rms(device float* x [[buffer(0)]],device const float* delta [[buffer(1)]],device const float* w [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[32];bonsai_rms_impl<true,false>(x,delta,w,reinterpret_cast<device const uint*>(x),out,p,r,tid,lane,sg,sums);
}
kernel void bonsai_rms_selected(device float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const uint* rows [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[32];bonsai_rms_impl<false,true>(x,x,w,rows,out,p,r,tid,lane,sg,sums);
}
kernel void bonsai_dn_qk_norm(device float* qkv [[buffer(0)]],constant uint* p [[buffer(1)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong base=ulong(g.y)*p[2]+g.x*128;float4 v=*reinterpret_cast<device float4*>(qkv+base+lane*4);
    float inv=precise::rsqrt(simd_sum(dot(v,v))/128.0f+1e-6f);
    *reinterpret_cast<device float4*>(qkv+base+lane*4)=(v*inv)*(g.x<p[0]?1.0f/128.0f:0.08838834764831844f);
}
kernel void bonsai_dn_gated_norm(device float* x [[buffer(0)]],device const float* z [[buffer(1)]],device const float* w [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    ulong base=(ulong(g.y)*p[1]+g.x)*128;float4 v=*reinterpret_cast<device float4*>(x+base+lane*4);
    float sum=v.x*v.x;sum+=v.y*v.y;sum+=v.z*v.z;sum+=v.w*v.w;
    float inv=precise::rsqrt(simd_sum(sum)/128.0f+as_type<float>(p[6]));
    float4 norm=(v*inv)*(*reinterpret_cast<device const float4*>(w+lane*4));
    float4 gate=*reinterpret_cast<device const float4*>(z+base+lane*4);
    *reinterpret_cast<device float4*>(x+base+lane*4)=(gate*mlx_sigmoid_f32(gate))*norm;
}

kernel void bonsai_knorm_store(device const float* input [[buffer(0)]],device const float* value [[buffer(1)]],
    device const float* norm [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],
    device float* keys [[buffer(5)]],device float* values [[buffer(6)]],device const uint* mrope [[buffer(7)]],constant uint* p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong src=(ulong(group.y)*p[1]+group.x)*256;
    uint slot=meta[group.y*2],pos=meta[group.y*2+1],physical=pages[slot*p[2]+pos/16]*16+pos%16;
    ulong dst=(ulong(physical)*p[1]+group.x)*256;
    float sum=0;for(uint d=lane;d<256;d+=32)sum+=input[src+d]*input[src+d];
    float inv=rsqrt(simd_sum(sum)/256.0f+as_type<float>(p[4]));
    for(uint d=lane;d<256;d+=32) {
        float v=input[src+d]*inv*norm[d];
        if(d<p[5]) {
            uint j=d%(p[5]/2),other=d<p[5]/2?d+p[5]/2:d-p[5]/2;
            uint axis=j%3==1 && j<33?1:(j%3==2 && j<30?2:0);
            float angle=float(mrope[group.y*4+axis])*pow(as_type<float>(p[3]),-2.0f*float(j)/float(p[5]));
            float partner=input[src+other]*inv*norm[other];
            v=v*cos(angle)+(d<p[5]/2?-partner:partner)*sin(angle);
        }
        keys[dst+d]=v;values[dst+d]=value[src+d];
    }
}

// Consume physical F32 pages directly: no context-sized attention matrix,
// transient KV duplication, or half-precision downgrade of the reference.
kernel void bonsai_attention_prefill(device float* q [[buffer(0)]],device const float* kc [[buffer(1)]],device const float* vc [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],
    device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float unused[1],probability[32*16],scores[32*16],maximum[32],denominator[32],correction[32];
    qwen_prefill_tile<1,float,16,false,true,true>(q,kc,vc,meta,pages,out,limits,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid,
        unused,probability,scores,maximum,denominator,correction);
}
