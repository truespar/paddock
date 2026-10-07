// Original Gemma kernels. The two cache geometries and learned Q/K scales
// follow Google's Gemma 4 graph; projection K=V does not mean cached K=V.
// F32 verification tiles reuse the exact dequantized weight across 8/16
// target rows. This extends our register-reuse GEMV, not a CPU replay or
// a change of the GGUF quantization. Smaller tiles retain their decode rungs.
#define GEMMA_VERIFY(R) \
kernel void gemma_verify##R(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device float* out [[buffer(2)]], constant uint* p [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    kquant_dispatch<R,false,float>(w,x,out,p[0],p[1],p[2],p[3],as_type<float>(p[4]),g.x*16+sg*4+lane/8,g.y*R,lane%8); }
GEMMA_VERIFY(8)
GEMMA_VERIFY(16)
#undef GEMMA_VERIFY

// Full narrow verification shapes remove per-row predicates from the inner
// dot product. Every row is real: no padding into another request's storage.
#define GEMMA_FULL(R) \
kernel void gemma_full##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
kquant_dispatch<R,true,float>(w,x,out,p[0],p[1],p[2],p[3],as_type<float>(p[4]),g.x*16+tid/8,g.y*R,tid%8); }
GEMMA_FULL(5)
GEMMA_FULL(6)
GEMMA_FULL(7)
GEMMA_FULL(8)
#undef GEMMA_FULL

// Precision-preserving matrix verification: stage the original quantized
// weights as F32, consume F32 activations directly and disallow MPP relaxed
// precision. Only this bounded 16-column tile is expanded, not the model.
template<uint BM>
inline void gemma_f32_tile(device const uchar* w,device float* x,device float* out,
    constant uint* p,uint2 g,uint tid,threadgroup float* weights) {
    constexpr uint BK=256,BN=16;
    uint K=p[0],N=p[1],M=p[2],n=g.x*BN,m=g.y*BM;
    auto input=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK+4});
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.template get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    for(uint base=0;base<K;base+=BK) {
        for(uint i=tid*4;i<BN*BK;i+=512){uint col=n+i/BK,k=base+i%BK;
            float4 v=col<N?kquant4(w,p[3],ulong(col)*K+k):float4(0);
            *reinterpret_cast<threadgroup float4*>(weights+(i/BK)*(BK+4)+i%BK)=v;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Dynamic row extents retain bounds checks on ragged verification.
        auto a=input.slice(base,m);op.run(a,b,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<c.get_capacity();++i)c[i]*=as_type<float>(p[4]);
    c.store(output.slice(n,m));
}
#define GEMMA_F32(BM) \
kernel void gemma_f32_##BM(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]], \
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float weights[16*260];gemma_f32_tile<BM>(w,x,out,p,g,tid,weights);}
GEMMA_F32(8)
GEMMA_F32(16)
GEMMA_F32(32)
#undef GEMMA_F32

// Hoist each 32-value group's scale metadata once and specialize its format
// outside the contraction. This preserves F32 operands and the exact same
// MPP reduction as gemma_f32_32; it only changes bounded weight staging.
template<uint Type>
inline void gemma_staged_f32(device const uchar* w,device float* x,device float* out,
    constant uint* p,uint2 g,uint tid,threadgroup float* weights) {
    constexpr uint BK=256,BN=16,BM=32;
    uint K=p[0],N=p[1],M=p[2],n=g.x*BN,m=g.y*BM;
    auto input=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK+4});
    auto output=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto c=op.template get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
    for(uint i=0;i<c.get_capacity();++i)c[i]=0;
    for(uint base=0;base<K;base+=BK) {
        uint i=tid*32;
        kquant_stage32<Type,float>(w,weights+(i/BK)*(BK+4)+i%BK,K,N,n+i/BK,base+i%BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a=input.slice(base,m);op.run(a,b,c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<c.get_capacity();++i)c[i]*=as_type<float>(p[4]);
    c.store(output.slice(n,m));
}
kernel void gemma_staged_f32_32(device const uchar* w [[buffer(0)]],device float* x [[buffer(1)]],
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float weights[16*260];
    switch(p[3]) {
    case 12:gemma_staged_f32<12>(w,x,out,p,g,tid,weights);break;
    case 13:gemma_staged_f32<13>(w,x,out,p,g,tid,weights);break;
    case 14:gemma_staged_f32<14>(w,x,out,p,g,tid,weights);break;
    case 23:gemma_staged_f32<23>(w,x,out,p,g,tid,weights);break;}
}

// Column coarsening: the same F32 activation vector feeds two output columns
// in a lane. Trade additional accumulators for fewer load instructions, without
// changing dequantization, per-column K ordering or the eight-lane reduction.
template<uint R,uint Type>
inline void gemma_pair(device const uchar* w,device const float* x,device float* out,
    uint K,uint N,float scale,uint n,uint lane) {
    if(n>=N)return;
    if constexpr(Type==14) {
        float4 sums[2][R];for(uint c=0;c<2;++c)for(uint r=0;r<R;++r)sums[c][r]=0;
        for(uint k=lane*16;k<K;k+=128) {
            device const uchar* blocks[2];float scales[2];uint ix=k%256;
            for(uint c=0;c<2;++c){blocks[c]=w+((ulong(min(n+c,N-1))*K+k)/256)*210;
                scales[c]=float(*reinterpret_cast<device const half*>(blocks[c]+208))*float(reinterpret_cast<device const char*>(blocks[c]+192)[ix/16]);}
            float4 partial[2][R];for(uint c=0;c<2;++c)for(uint r=0;r<R;++r)partial[c][r]=0;
            #pragma unroll
            for(uint j=0;j<4;++j) {
                uint index=ix+j*4,part=index/128,r=index%128;float4 value[2];
                for(uint c=0;c<2;++c){
                    uint lo=as_type<uint>(*reinterpret_cast<device const packed_ushort2*>(blocks[c]+part*64+r%64));
                    uint hi=as_type<uint>(*reinterpret_cast<device const packed_ushort2*>(blocks[c]+128+part*32+r%32));
                    uint code=((lo>>((r/64)*4))&0x0f0f0f0f)|(((hi>>((r/32)*2))&0x03030303)<<4);
                    value[c]=float4(as_type<uchar4>(code))-32.0f;
                }
                for(uint row=0;row<R;++row){float4 v=*reinterpret_cast<device const float4*>(x+ulong(row)*K+k+j*4);
                    for(uint c=0;c<2;++c)partial[c][row]=fma(value[c],v,partial[c][row]);}
            }
            for(uint c=0;c<2;++c)for(uint r=0;r<R;++r)sums[c][r]=fma(partial[c][r],scales[c],sums[c][r]);
        }
        for(uint c=0;c<2;++c)for(uint r=0;r<R;++r){float4 t=sums[c][r];float v=kquant_sum<8>(t.x+t.y+t.z+t.w);
            if(lane==0 && n+c<N)out[ulong(r)*N+n+c]=v*scale;}
        return;
    }
    float4 total[2][R];for(uint c=0;c<2;++c)for(uint r=0;r<R;++r)total[c][r]=0;
    for(uint k=lane*32;k<K;k+=256) {
        device const uchar* blocks[2];float ds[2],ms[2];uint s=(k%256)/32;
        for(uint c=0;c<2;++c) {
            // An odd final column may share the first column's reads, but
            // never writes a nonexistent output or accesses past the weights.
            blocks[c]=w+((ulong(min(n+c,N-1))*K+k)/256)*(Type==12?144:176);
            device const uchar* sc=blocks[c]+4;
            uint d=s<4?sc[s]&63:(sc[s+4]&15)|((sc[s-4]>>6)<<4);
            uint m=s<4?sc[s+4]&63:(sc[s+4]>>4)|((sc[s]>>6)<<4);
            ds[c]=float(*reinterpret_cast<device const half*>(blocks[c]))*float(d);
            ms[c]=float(*reinterpret_cast<device const half*>(blocks[c]+2))*float(m);
        }
        #pragma unroll
        for(uint j=0;j<8;++j) {
            float4 value[2];
            for(uint c=0;c<2;++c) {
                uint packed=(*reinterpret_cast<device const uint*>(blocks[c]+(Type==12?16:48)+(s/2)*32+j*4)>>((s%2)*4))&0x0f0f0f0f;
                if constexpr(Type==13)packed|=((*reinterpret_cast<device const uint*>(blocks[c]+16+j*4)>>s)&0x01010101)<<4;
                value[c]=float4(as_type<uchar4>(packed))*ds[c]-ms[c];
            }
            for(uint r=0;r<R;++r){float4 v=*reinterpret_cast<device const float4*>(x+ulong(r)*K+k+j*4);
                for(uint c=0;c<2;++c)total[c][r]=fma(value[c],v,total[c][r]);}
        }
    }
    for(uint c=0;c<2;++c)for(uint r=0;r<R;++r) {
        float4 t=total[c][r];float v=kquant_sum<8>(t.x+t.y+t.z+t.w);
        if(lane==0 && n+c<N)out[ulong(r)*N+n+c]=v*scale;
    }
}
#define GEMMA_PAIR(R) \
kernel void gemma_pair##R(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]], \
constant uint* p [[buffer(3)]],uint g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
if(p[3]==12)gemma_pair<R,12>(w,x,out,p[0],p[1],as_type<float>(p[4]),g*32+(tid/8)*2,tid%8); \
else if(p[3]==13)gemma_pair<R,13>(w,x,out,p[0],p[1],as_type<float>(p[4]),g*32+(tid/8)*2,tid%8); \
else gemma_pair<R,14>(w,x,out,p[0],p[1],as_type<float>(p[4]),g*32+(tid/8)*2,tid%8);}
GEMMA_PAIR(2)
GEMMA_PAIR(3)
GEMMA_PAIR(4)
#undef GEMMA_PAIR

#define GEMMA_MULTI_PAIR(R) \
kernel void gemma_multi_pair##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
uint n0=(p[1]+31)/32,n1=(p[2]+31)/32,N=p[1],ty=p[5];device const uchar* w=w0;device float* out=o0; \
if(g>=n0){g-=n0;N=p[2];ty=p[6];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];ty=p[7];w=w2;out=o2;}} \
if(ty==12)gemma_pair<R,12>(w,x,out,p[0],N,1.0f,g*32+(tid/8)*2,tid%8); \
else if(ty==13)gemma_pair<R,13>(w,x,out,p[0],N,1.0f,g*32+(tid/8)*2,tid%8); \
else gemma_pair<R,14>(w,x,out,p[0],N,1.0f,g*32+(tid/8)*2,tid%8);}
GEMMA_MULTI_PAIR(3)
GEMMA_MULTI_PAIR(4)
#undef GEMMA_MULTI_PAIR

// Only rotary positions advance; the assistant's committed-KV bounds live
// in another buffer and must stay fixed through every step of the chain.
kernel void gemma_draft_advance(device const uint* ids [[buffer(0)]], device uint* meta [[buffer(1)]],
    device uint* out [[buffer(2)]], constant uint* p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if(row>=p[0])return;out[row*p[2]+p[1]]=ids[row];meta[row*2+1]++;
}
inline uint gemma_physical(device const uint* pages, uint slot, uint pos, constant uint* p) {
    // p: Q heads, KV heads, page stride, window (0=global), ring tokens.
    // A zero ring size elects full paged storage with a sliding read mask
    // (Laguna). Existing Gemma/Muse sliding caches always pass a real ring.
    return p[3] && p[4] ? slot*p[4]+pos%p[4] : pages[slot*p[2]+pos/16]*16+pos%16;
}
// p: heads, head dim, norm type, epsilon, rope base, global-factor flag.
kernel void gemma_qnorm(device float* q [[buffer(0)]], device const uchar* norm [[buffer(1)]],
    device const uint* meta [[buffer(2)]], device const float* factors [[buffer(3)]],
    constant uint* p [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint hd=p[1]; ulong src=(ulong(g.y)*p[0]+g.x)*hd;
    float sum=0; for(uint d=lane;d<hd;d+=32)sum+=q[src+d]*q[src+d];
    float inv=rsqrt(simd_sum(sum)/float(hd)+as_type<float>(p[3]));
    // Each lane owns both halves of a rotary pair; in-place writes cannot
    // alter any other lane's partner or the already-completed norm reduction.
    for(uint j=lane;j<hd/2;j+=32) {
        float a=q[src+j]*inv*weight(norm,p[2],j), b=q[src+j+hd/2]*inv*weight(norm,p[2],j+hd/2);
        float angle=float(meta[g.y*2+1])*pow(as_type<float>(p[4]),-2.0f*float(j)/float(hd));
        if(p[5])angle/=factors[j];
        float c=cos(angle),s=sin(angle);
        q[src+j]=a*c-b*s;q[src+j+hd/2]=a*s+b*c;
    }
}
// p[0..4] cache geometry; then hd, norm type, eps, rope base.
kernel void gemma_kv_store(device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device const uchar* norm [[buffer(2)]], device const uint* meta [[buffer(3)]],
    device const uint* pages [[buffer(4)]], device const float* factors [[buffer(5)]],
    device half* keys [[buffer(6)]], device half* values [[buffer(7)]], constant uint* p [[buffer(8)]],
    uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint hd=p[5],slot=meta[g.y*2],pos=meta[g.y*2+1];
    ulong src=(ulong(g.y)*p[1]+g.x)*hd,dst=(ulong(gemma_physical(pages,slot,pos,p))*p[1]+g.x)*hd;
    float ks=0,vs=0;for(uint d=lane;d<hd;d+=32){ks+=k[src+d]*k[src+d];vs+=v[src+d]*v[src+d];}
    float ki=rsqrt(simd_sum(ks)/float(hd)+as_type<float>(p[7]));
    float vi=rsqrt(simd_sum(vs)/float(hd)+as_type<float>(p[7]));
    for(uint j=lane;j<hd/2;j+=32) {
        float a=k[src+j]*ki*weight(norm,p[6],j),b=k[src+j+hd/2]*ki*weight(norm,p[6],j+hd/2);
        float angle=float(pos)*pow(as_type<float>(p[8]),-2.0f*float(j)/float(hd));
        if(!p[3])angle/=factors[j];
        float c=cos(angle),s=sin(angle);
        keys[dst+j]=half(a*c-b*s);keys[dst+j+hd/2]=half(a*s+b*c);
        values[dst+j]=half(v[src+j]*vi);values[dst+j+hd/2]=half(v[src+j+hd/2]*vi);
    }
}
// Two sandwich norms and the intervening residual in one dispatch. Scaling
// belongs after the residual, never on the FFN branch alone. The second norm
// uses the scaled state and its own epsilon, preserving the graph's order.
// p: width, post weight type, next weight type, eps, output scale.
kernel void gemma_sandwich(device float* x [[buffer(0)]], device const float* delta [[buffer(1)]],
    device const uchar* post [[buffer(2)]], device const uchar* next [[buffer(3)]], device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];uint n=p[0];ulong base=ulong(row)*n;float sum=0;
    for(uint d=tid;d<n;d+=256){float z=delta[base+d];sum+=z*z;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=rsqrt(simd_sum(lane<8?sums[lane]:0.0f)/float(n)+as_type<float>(p[3]));
    // Don't recycle sums until every SIMD group consumed the first reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);sum=0;
    for(uint d=tid;d<n;d+=256){float z=(x[base+d]+delta[base+d]*inv*weight(post,p[1],d))*as_type<float>(p[4]);x[base+d]=z;sum+=z*z;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    inv=rsqrt(simd_sum(lane<8?sums[lane]:0.0f)/float(n)+as_type<float>(p[3]));
    for(uint d=tid;d<n;d+=256)out[base+d]=x[base+d]*inv*weight(next,p[2],d);
}
kernel void gemma_geglu(device float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<p[0])gate[i]=vis_gelu(gate[i])*up[i];
}
kernel void gemma_softcap(device float* logits [[buffer(0)]], constant uint* p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if(i<p[0]){float c=as_type<float>(p[1]);logits[i]=c*precise::tanh(clamp(logits[i]/c,-10.0f,10.0f));}
}

// Split-K GQA decode: one key tile serves every query head in its group.
// HD/GQA are compile-time so the compiler sees bounded register footprints.
// Exact online-softmax state composition, no context-sized score allocation.
template<uint HD,uint GQA,bool Muse=false,typename KV=half>
inline void gemma_decode(device const float* q,device const KV* k,device const KV* v,
    device const uint* meta,device const uint* pages,device const uint* rows,device float* out,
    constant uint* p,uint3 g,uint tid,uint lane,uint sg,threadgroup float* scores,threadgroup float* prob,
    threadgroup float* highs,threadgroup float* sums) {
    uint row=rows[g.y],kh=g.x,slot=meta[row*2],length=meta[row*2+1]+1;
    uint low=p[3] && length>p[3]?length-p[3]:0;
    uint span=(length-low+p[5]-1)/p[5],first=low+g.z*span,last=min(first+span,length);
    float acc[GQA][HD/128],maximum[GQA],denom[GQA];
    for(uint h=0;h<GQA;++h){maximum[h]=-INFINITY;denom[h]=0;for(uint d=0;d<HD/128;++d)acc[h][d]=0;}
    for(uint base=first;base<last;base+=32) {
        uint token=base+tid/4,dl=tid%4;float dots[GQA];for(uint h=0;h<GQA;++h)dots[h]=0;
        if(token<last) {
            ulong ko=(ulong(gemma_physical(pages,slot,token,p))*p[1]+kh)*HD;
            for(uint j=0;j<HD/16;++j){uint d=dl*4+j*16;float4 key=float4(*reinterpret_cast<device const vec<KV,4>*>(k+ko+d));
                for(uint h=0;h<GQA;++h)dots[h]+=dot(key,*reinterpret_cast<device const float4*>(q+(ulong(row)*p[0]+kh*GQA+h)*HD+d));}
        }
        for(uint h=0;h<GQA;++h){dots[h]+=simd_shuffle_xor(dots[h],1);dots[h]+=simd_shuffle_xor(dots[h],2);
            if constexpr(Muse)dots[h]*=0.08838834764831845f;
            if(dl==0)scores[h*32+tid/4]=token<last?dots[h]:-INFINITY;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint h=sg;h<GQA;h+=4){float z=scores[h*32+lane],hi=simd_max(z),pr=isfinite(hi)?exp(z-hi):0.0f;
            float sum=simd_sum(pr);prob[h*32+lane]=pr;if(lane==0){highs[h]=hi;sums[h]=sum;}}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mul[GQA];for(uint h=0;h<GQA;++h){float hi=max(maximum[h],highs[h]),old=isfinite(maximum[h])?exp(maximum[h]-hi):0.0f;
            mul[h]=exp(highs[h]-hi);denom[h]=denom[h]*old+sums[h]*mul[h];maximum[h]=hi;
            for(uint d=0;d<HD/128;++d)acc[h][d]*=old;}
        for(uint j=0;j<min(32u,last-base);++j){ulong vi=(ulong(gemma_physical(pages,slot,base+j,p))*p[1]+kh)*HD+tid;
            for(uint d=0;d<HD/128;++d){float value=float(v[vi+d*128]);for(uint h=0;h<GQA;++h)acc[h][d]+=value*(prob[h*32+j]*mul[h]);}}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint h=0;h<GQA;++h){ulong dst=((ulong(g.y)*p[0]+kh*GQA+h)*p[5]+g.z)*(HD+2);
        for(uint d=0;d<HD/128;++d)out[dst+tid+d*128]=acc[h][d];
        if(tid==0){out[dst+HD]=maximum[h];out[dst+HD+1]=denom[h];}}
}
#define GEMMA_DECODE(HD,GQA) \
kernel void gemma_decode##HD(device const float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]], \
device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) { \
threadgroup float scores[GQA*32],prob[GQA*32],highs[GQA],sums[GQA];gemma_decode<HD,GQA>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);}
GEMMA_DECODE(256,2)
GEMMA_DECODE(512,8)
#undef GEMMA_DECODE
// p: Q heads, split count, head dim.
kernel void gemma_merge(device const float* parts [[buffer(0)]],device float* out [[buffer(1)]],device const uint* rows [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint hd=p[2],rh=rows[g/p[0]]*p[0]+g%p[0];ulong base=ulong(g)*p[1]*(hd+2);
    float hi=-INFINITY;for(uint s=0;s<p[1];++s)hi=max(hi,parts[base+s*(hd+2)+hd]);
    for(uint d=lane;d<hd;d+=32){float acc=0,denom=0;for(uint s=0;s<p[1];++s){ulong src=base+s*(hd+2);float l=parts[src+hd+1];
        if(l>0){float c=exp(parts[src+hd]-hi);acc+=parts[src+d]*c;denom+=l*c;}}out[ulong(rh)*hd+d]=acc/denom;}
}

// One lane owns one partition's normalizer (the host bounds splits at 32).
// Broadcast it across the head instead of loading/recomputing exp for every
// dimension. Both sums retain ascending partition order and F32 arithmetic;
// empty tail partitions contribute nothing, including when their max is -inf.
template<uint HD>
inline void gemma_merge_vectors(device const float* parts,device float* out,device const uint* rows,
    constant uint* p,uint g,uint lane) {
    uint rh=rows[g/p[0]]*p[0]+g%p[0];ulong base=ulong(g)*p[1]*(HD+2);
    float hi=lane<p[1]?parts[base+lane*(HD+2)+HD]:-INFINITY;
    float l=lane<p[1]?parts[base+lane*(HD+2)+HD+1]:0;
    float maximum=simd_max(hi),c=l>0?exp(hi-maximum):0;
    float denom=0;float4 acc[HD/128];for(uint d=0;d<HD/128;++d)acc[d]=0;
    for(uint s=0;s<p[1];++s){float ls=simd_shuffle(l,s),cs=simd_shuffle(c,s);
        if(ls>0){denom+=ls*cs;
            // Partition stride is HD+2, so odd partitions are only 8-byte
            // aligned. packed_float4 avoids an invalid float4-aligned load.
            for(uint d=0;d<HD/128;++d)acc[d]+=float4(*reinterpret_cast<device const packed_float4*>(parts+base+s*(HD+2)+lane*4+d*128))*cs;}}
    for(uint d=0;d<HD/128;++d)*reinterpret_cast<device float4*>(out+ulong(rh)*HD+lane*4+d*128)=acc[d]/denom;
}
kernel void gemma_merge_shared(device const float* parts [[buffer(0)]],device float* out [[buffer(1)]],device const uint* rows [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    if(p[2]==256)gemma_merge_vectors<256>(parts,out,rows,p,g,lane);
    else gemma_merge_vectors<512>(parts,out,rows,p,g,lane);
}

// MPP online-softmax prefill, with a smaller key tile for global HD=512.
// The SWA lower bound is per query, not per chunk; partial tiles never read
// sequence-neighbour metadata. Rings include CHUNK slack before all KV writes.
template<uint HD,uint KT,uint BM=32,typename T=half,bool Image=false,bool Muse=false,typename KV=half,bool Relaxed=false,bool Bidir=false,typename P=T,bool Exp2=false,uint TilePanel=0>
inline void gemma_prefill(device T* q,device const KV* k,device const KV* v,device const uint* meta,
    device const uint* pages,device float* out,constant uint* p,uint head,uint first,uint count,uint tid,
    threadgroup T* kv,threadgroup P* probability,threadgroup float* scores,
    threadgroup float* maximum,threadgroup float* denominator,threadgroup float* correction,device const uint* limits=nullptr) {
    constexpr uint G=128/BM;
    // Stream contraction/output panels, not the entire HD512 KV tile. MPP
    // compiler staging counts against the same 32 KiB threadgroup budget.
    constexpr uint Panel=TilePanel?TilePanel:(sizeof(T)==4?128:(HD<256?HD:256));
    uint kh=head/(p[0]/p[1]),slot=meta[2*first],lastpos=meta[2*(first+count-1)+1],width=p[0]*HD;
    if constexpr(Image)lastpos=limits[first+count-1];
    uint low=p[3] && meta[2*first+1]+1>p[3]?meta[2*first+1]+1-p[3]:0;
    // Retrieval has a symmetric window and no persistent KV pages. `limits`
    // is the last local position; meta's slot is the packed sequence start.
    // Compile-time specialization leaves every generative caller unchanged.
    if constexpr(Bidir) {
        lastpos=p[3]?min(limits[first],lastpos+p[3]):limits[first];
        low=p[3] && meta[2*first+1]>p[3]?meta[2*first+1]-p[3]:0;
        if constexpr(Exp2)low=low/KT*KT;
    }
    auto tq=tensor(q+ulong(first)*width+head*HD,dextents<int,2>(HD,count),array<int,2>{1,int(width)});
    auto tk=tensor(kv,extents<int,Panel,KT>(),array<int,2>{1,Panel});
    auto tv=tensor(kv,extents<int,KT,Panel>(),array<int,2>{1,KT});
    auto tp=tensor(probability,extents<int,KT,BM>(),array<int,2>{1,KT});
    auto ts=tensor(scores,extents<int,KT,BM>(),array<int,2>{1,KT});
    constexpr auto qkd=matmul2d_descriptor(BM,KT,Panel,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    // The MLX Llama graph opts into its reference's reduced-precision F32
    // probability contraction. All other model arithmetic stays unchanged.
    constexpr auto pvd=matmul2d_descriptor(BM,Panel,KT,false,true,Relaxed,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroups<4>> qk;matmul2d<pvd,execution_simdgroups<4>> pv;
    static_assert(HD/Panel<=4);
    auto acc0=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    auto acc1=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    auto acc2=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    auto acc3=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<acc0.get_capacity();++i){acc0[i]=0;acc1[i]=0;acc2[i]=0;acc3[i]=0;}
    if(tid<BM){maximum[tid]=-INFINITY;denominator[tid]=0;}
    for(uint base=low;base<=lastpos;base+=KT){
        auto score=qk.template get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        for(uint i=0;i<score.get_capacity();++i)score[i]=0;
        for(uint panel=0;panel<HD/Panel;++panel){
            for(uint i=tid;i<KT*Panel;i+=128){uint t=base+i/Panel,d=panel*Panel+i%Panel;
                kv[i]=t<=lastpos?T(k[(ulong(Bidir?slot+t:gemma_physical(pages,slot,t,p))*p[1]+kh)*HD+d]):T(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto query=tq.slice(panel*Panel,0);qk.run(query,tk,score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if constexpr(Muse)for(uint i=0;i<score.get_capacity();++i)score[i]*=0.08838834764831845f;
        if constexpr(Exp2)for(uint i=0;i<score.get_capacity();++i)score[i]*=1.44269504089f;
        score.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint row=tid/G,lane=tid%G,pos=row<count?meta[2*(first+row)+1]:0;
        uint upper=pos;if constexpr(Image)upper=row<count?limits[first+row]:0;
        uint lo=p[3] && pos+1>p[3]?pos+1-p[3]:0;
        if constexpr(Bidir) { upper=p[3]?min(limits[first],pos+p[3]):limits[first];lo=p[3] && pos>p[3]?pos-p[3]:0; }
        float high=maximum[row];for(uint j=lane;j<KT;j+=G)if(row<count && base+j<=upper && base+j>=lo)high=max(high,scores[row*KT+j]);
        #pragma unroll
        for(uint shift=1;shift<G;shift*=2)high=max(high,simd_shuffle_xor(high,shift));
        float old=isfinite(maximum[row])?(Exp2?fast::exp2(maximum[row]-high):exp(maximum[row]-high)):0.0f,sum=0;
        for(uint j=lane;j<KT;j+=G){float pr=row<count && base+j<=upper && base+j>=lo?(Exp2?fast::exp2(scores[row*KT+j]-high):exp(scores[row*KT+j]-high)):0.0f;
            probability[row*KT+j]=P(pr);sum+=pr;}
        #pragma unroll
        for(uint shift=1;shift<G;shift*=2)sum+=simd_shuffle_xor(sum,shift);
        if(lane==0){maximum[row]=high;correction[row]=old;denominator[row]=row<count?denominator[row]*old+sum:1.0f;}
        for(uint panel=0;panel<HD/Panel;++panel){
            for(uint i=tid;i<KT*Panel;i+=128){uint t=base+i%KT,d=panel*Panel+i/KT;
                kv[i]=t<=lastpos?T(v[(ulong(Bidir?slot+t:gemma_physical(pages,slot,t,p))*p[1]+kh)*HD+d]):T(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto product=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
            for(uint i=0;i<product.get_capacity();++i)product[i]=0;pv.run(tp,tv,product);
            uint i=0;for(auto it=product.begin();it!=product.end();++it,++i){auto ij=it.get_multidimensional_index();if(it.is_valid_element()){
                if(panel==0)acc0[i]=acc0[i]*correction[ij[1]]+*it;
                else if(panel==1)acc1[i]=acc1[i]*correction[ij[1]]+*it;
                else if(panel==2)acc2[i]=acc2[i]*correction[ij[1]]+*it;
                else acc3[i]=acc3[i]*correction[ij[1]]+*it;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for(uint panel=0;panel<HD/Panel;++panel){
        thread auto& acc=panel==0?acc0:(panel==1?acc1:(panel==2?acc2:acc3));
        for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();if(it.is_valid_element())*it/=denominator[ij[1]];}
        auto dst=tensor(out+ulong(first)*width+head*HD+panel*Panel,dextents<int,2>(Panel,count),array<int,2>{1,int(width)});acc.store(dst);
    }
}
#define GEMMA_PREFILL(HD,KT) \
kernel void gemma_prefill##HD(device half* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]], \
device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup half kv[KT*(HD<256?HD:256)],prob[KT*32];threadgroup float scores[KT*32],maximum[32],denom[32],correction[32]; \
gemma_prefill<HD,KT>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction);}
// MPP's compiled staging (including shader validation) doubles these source
// arrays on Apple10. KT32/16 exceeded the 32 KiB limit at 45,824/39,680 bytes.
// Keep all 32 queries; panel HD512 and halve only HD256's streamed key tile.
GEMMA_PREFILL(256,16)
GEMMA_PREFILL(512,16)
#undef GEMMA_PREFILL

// The scheduler writes each complete image span before this dispatch. RoPE
// and SWA lower bounds retain the real query position; only the upper bound
// extends to the last soft token in that same image, never the next image.
#define GEMMA_IMAGE_PREFILL(HD,KT) \
kernel void gemma_image_prefill##HD(device half* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]], \
device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]], \
device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup half kv[KT*(HD<256?HD:256)],prob[KT*32];threadgroup float scores[KT*32],maximum[32],denom[32],correction[32]; \
gemma_prefill<HD,KT,32,half,true>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction,limits);}
GEMMA_IMAGE_PREFILL(256,16)
GEMMA_IMAGE_PREFILL(512,16)
#undef GEMMA_IMAGE_PREFILL

// MoE keeps queries/probabilities F32 across prefill/decode. Stream 128-wide
// KV panels from the same F16 persistent cache; no expanded cache copy.
#define GMOE_PREFILL(HD) \
kernel void gmoe_prefill##HD(device float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]], \
device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float kv[16*128],prob[16*32],scores[16*32],maximum[32],denom[32],correction[32]; \
gemma_prefill<HD,16,32,float>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction); } \
kernel void gmoe_image_prefill##HD(device float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]], \
device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]], \
device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float kv[16*128],prob[16*32],scores[16*32],maximum[32],denom[32],correction[32]; \
gemma_prefill<HD,16,32,float,true>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction,limits); }
GMOE_PREFILL(256)
GMOE_PREFILL(512)
#undef GMOE_PREFILL

// An eight-query F32 attention tile shares the KV reads across the entire
// speculative block in the sliding layers. The 16-key staging tile fits in
// threadgroup memory without rounding Q or P. Global HD512 retains decode.
#define GEMMA_VERIFY_ATTN(HD,KT) \
kernel void gemma_verify_attn##HD(device float* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]], \
device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float kv[KT*128],prob[KT*8],scores[KT*8],maximum[8],denom[8],correction[8]; \
gemma_prefill<HD,KT,8,float>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction);}
GEMMA_VERIFY_ATTN(256,16)
#undef GEMMA_VERIFY_ATTN
