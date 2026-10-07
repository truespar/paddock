// Original Kolibri implementation, composed with Paddock's own packed
// affine primitives and paged online-softmax attention. No expanded weights.
// The router chooses raw FP32 logits+bias; mixture weights use UNBIASED
// sigmoid logits, with no normalization. This is not Qwen/Laguna routing.
kernel void kolibri_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
    device uint* ids [[buffer(2)]],device float* weights [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float score[12];for(uint j=0;j<12;++j)score[j]=logits[row*384+lane+j*32]+bias[lane+j*32];
    for(uint pick=0;pick<6;++pick){
        float best=-INFINITY;for(uint j=0;j<12;++j)best=max(best,score[j]);best=simd_max(best);
        uint id=UINT_MAX;for(uint j=0;j<12;++j)if(score[j]==best)id=min(id,lane+j*32);id=simd_min(id);
        // Invalid nonfinite checkpoints must not turn into out-of-range GPU reads.
        if(id==UINT_MAX)id=0;
        if(lane==0){float x=logits[row*384+id];float tail=1.0f/(1.0f+exp(abs(x)));
            ids[row*6+pick]=id;weights[row*6+pick]=mlx_bf(x<0?tail:1.0f-tail);}
        for(uint j=0;j<12;++j)if(lane+j*32==id)score[j]=-INFINITY;
    }
}
// 384 counters exceed the older 256-expert tile prefix. One workgroup with
// a deterministic integer scan; no float atomics or host expert-ID readback.
kernel void kolibri_tiles(device const uint* counts [[buffer(0)]],device uint* tiles [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint prefix[512];prefix[tid]=tid<384?(counts[tid]+15)/16:0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint offset=1;offset<512;offset*=2){uint v=tid>=offset?prefix[tid-offset]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup);prefix[tid]+=v;threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(tid==0)tiles[0]=prefix[511];
    if(tid<384){uint first=tid?prefix[tid-1]:0;
        for(uint t=first;t<prefix[tid];++t){tiles[1+2*t]=tid;tiles[2+2*t]=(t-first)*16;}}
}
kernel void kolibri_expert_mv(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint entry=g.y;
    q4a_vector<4,64>(w,x+ulong(p[3]?entry:entry/6)*p[0],y+ulong(entry)*p[1],
        p[0],384*p[1],p[1],g.x*16+tid/32*4,tid%32,ids[entry]);
}
template<bool Packed>
inline void kolibri_grouped_impl(device const uchar* w,device const float* x,
    device const uint* lists,device const uint* counts,device const uint* tiles,
    device float* out,constant uint* p,uint2 g,uint tid,threadgroup bfloat* acts,threadgroup bfloat* weights) {
    if(g.y>=tiles[0])return;
    constexpr uint BM=16,BN=32,BK=64;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(BM,counts[expert]-first);
    uint K=p[0],N=p[1],col=g.x*BN;
    auto a=tensor(acts,extents<int,BK,BM>(),array<int,2>{1,BK});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<K;base+=BK){
        for(uint i=tid;i<BM*BK;i+=128){uint r=i/BK,entry=r<count?lists[expert*p[2]*6+first+r]:0;
            acts[i]=bfloat(r<count?x[ulong(p[3]?entry:entry/6)*K+base+i%BK]:0);}
        // Decode eight adjacent nibbles per packed load and share the affine
        // metadata. Keep the same BF16 tile and contraction order: this is
        // storage traffic/addressing work, not a new numerical election.
        device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*384*N/2);
        device const bfloat* biases=scales+ulong(K)*384*N/64;
        if constexpr(Packed)for(uint i=tid*8;i<BN*BK;i+=128*8){uint n=col+i/BK;float4 lo=0,hi=0;
            if(n<N){ulong at=(ulong(expert)*N+n)*K+base+i%BK;
                uint codes=reinterpret_cast<device const uint*>(w)[at/8];
                float scale=float(scales[at/64]),bias=float(biases[at/64]);
                lo=fma(float4(codes&15,(codes>>4)&15,(codes>>8)&15,(codes>>12)&15),float4(scale),float4(bias));
                hi=fma(float4((codes>>16)&15,(codes>>20)&15,(codes>>24)&15,codes>>28),float4(scale),float4(bias));}
            *reinterpret_cast<threadgroup bfloat4*>(weights+i)=bfloat4(lo);
            *reinterpret_cast<threadgroup bfloat4*>(weights+i+4)=bfloat4(hi);}
        else for(uint i=tid;i<BN*BK;i+=128){uint n=col+i/BK;
            weights[i]=bfloat(n<N?q4a_value_t<4,64>(w,K,384*N,(ulong(expert)*N+n)*K+base+i%BK):0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && ij[1]<count && col+ij[0]<N){uint entry=lists[expert*p[2]*6+first+ij[1]];
            out[ulong(entry)*N+col+ij[0]]=mlx_bf(*it);}}
}
#define KOLIBRI_GROUPED(Name,Packed) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]], \
    device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) {threadgroup bfloat acts[16*64],weights[32*64]; \
    kolibri_grouped_impl<Packed>(w,x,lists,counts,tiles,out,p,g,tid,acts,weights);}
KOLIBRI_GROUPED(kolibri_grouped,true)
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
KOLIBRI_GROUPED(kolibri_grouped_scalar,false)
#endif
#undef KOLIBRI_GROUPED
kernel void kolibri_fold(device const float* experts [[buffer(0)]],device const float* prob [[buffer(1)]],
    device float* shared [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0])return;uint row=i/2560,d=i%2560;float sum=0;
    for(uint e=0;e<6;++e)sum+=mlx_bf(experts[(ulong(row)*6+e)*2560+d]*prob[row*6+e]);
    shared[i]=mlx_bf(mlx_bf(sum)+shared[i]);
}
kernel void kolibri_rope(device float* q [[buffer(0)]],device const uint* meta [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    ulong at=(ulong(g.y)*p[0]+g.x)*128;
    for(uint j=lane;j<64;j+=32){float a=q[at+j],b=q[at+j+64];
        float angle=float(meta[2*g.y+1])*exp(-float(j)/64.0f*log(10000.0f));
        float c=cos(angle),s=sin(angle);
        q[at+j]=mlx_bf(a*c-b*s);q[at+j+64]=mlx_bf(a*s+b*c);}
}
kernel void kolibri_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],
    device bfloat* kc [[buffer(2)]],device bfloat* vc [[buffer(3)]],device const uint* meta [[buffer(4)]],
    device const uint* pages [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*512)return;uint row=i/512,pos=meta[2*row+1],slot=meta[2*row];
    ulong at=(ulong(pages[slot*p[1]+pos/16])*16+pos%16)*512+i%512;
    kc[at]=bfloat(k[i]);vc[at]=bfloat(v[i]);
}
kernel void kolibri_decode(device const float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[12*32],prob[12*32],highs[12],sums[12];
    gemma_decode<128,12,true,bfloat>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums);
}
kernel void kolibri_prefill(device float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float kv[16*128],prob[16*32],scores[16*32],maximum[32],denom[32],correction[32];
    gemma_prefill<128,16,32,float,false,true,bfloat,true>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction);
}
kernel void kolibri_round(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=mlx_bf(x[i]);
}
