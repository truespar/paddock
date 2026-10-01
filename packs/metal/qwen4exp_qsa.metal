// Original Flash Next text QSA: exact whole-block selection (512x4 tokens)
// plus the incomplete causal tail. No context-sized attention mask and no
// CPU top-k. Logical blocks stay per-slot even when KV pages are permuted.
// Math studied in Qwen's published graph; no external kernels are copied.
kernel void q4s_bf16_mm(device const ushort* w [[buffer(0)]],device const float* x [[buffer(1)]],
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[1024],b[1024];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint row=g.y*32,col=g.x*32;
    for(uint base=0;base<p[0];base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint r=row+i/32,n=col+i/32,k=base+i%32;
            a[i]=r<p[2] ? x[ulong(r)*p[0]+k] : 0;
            b[i]=n<p[1] ? as_type<float>(uint(w[ulong(n)*p[0]+k])<<16) : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();
        if(row+ij[1]<p[2] && col+ij[0]<p[1])y[ulong(row+ij[1])*p[1]+col+ij[0]]=*it;
    }
}
// Text-only partial split-half RoPE: all three mrope axes are position.
// p: head width, heads, input head stride, input row stride, epsilon.
kernel void q4s_norm_rope(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device const uint4* meta [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong src=ulong(g.y)*p[3]+g.x*p[2],dst=(ulong(g.y)*p[1]+g.x)*p[0];
    // Preserve both rotary partners before an in-place norm writes either.
    threadgroup float original[256];
    float sq=0;for(uint d=lane;d<p[0];d+=32){float v=x[src+d];original[d]=v;sq+=v*v;}
    float inv=rsqrt(simd_sum(sq)/float(p[0])+as_type<float>(p[4]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<p[0];d+=32) {
        float v=original[d]*inv*w[d];
        if(d<64) {
            float angle=float(meta[g.y].y)*pow(1e7f,-float(d%32)/32.0f);
            uint partner=d<32 ? d+32 : d-32;
            float other=original[partner]*inv*w[partner];
            v=v*cos(angle)+(d<32 ? -other : other)*sin(angle);
        }
        y[dst+d]=v;
    }
}
// All incoming KV writes are disjoint. p: page stride, block stride, rows.
kernel void q4s_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],
 device const uint4* meta [[buffer(2)]],device const uint* pages [[buffer(3)]],
 device half* kc [[buffer(4)]],device half* vc [[buffer(5)]],constant uint* p [[buffer(6)]],
 uint i [[thread_position_in_grid]]) {
    uint row=i/512,d=i%512;if(row>=p[2])return;
    uint4 m=meta[row];uint physical=pages[m.x*p[0]+m.y/16]*16+m.y%16;
    kc[ulong(physical)*512+d]=half(k[i]);vc[ulong(physical)*512+d]=half(v[i]);
}
// Commit a completed block once, pooling before norm/rotation. Keep raw
// carry and pooled keys F32: F16 raw-key rounding before normalization can
// amplify batch-dependent projection noise. Main paged KV stays F16. This
// explicit index-cache class needs its own whole-model reference gate.
kernel void q4s_pool(device const float* raw [[buffer(0)]],device const float* ring [[buffer(1)]],
 device const float* norm [[buffer(2)]],device const uint4* meta [[buffer(3)]],
 device float* pooled [[buffer(4)]],constant uint* p [[buffer(5)]],
 uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint4 m=meta[row];if(m.y%4!=3)return;
    threadgroup float values[128];float sq=0;
    for(uint d=lane;d<128;d+=32) {
        float sum=0;
        for(uint j=0;j<4;++j) {
            int r=int(row)-3+int(j);uint pos=m.y-3+j;
            sum+=r>=int(m.z) ? raw[ulong(r)*128+d] : ring[(m.x*4+pos%4)*128+d];
        }
        float v=sum*0.25f;values[d]=v;sq+=v*v;
    }
    float inv=rsqrt(simd_sum(sq)/128.0f+1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong dst=(ulong(m.x)*p[1]+m.y/4)*128;
    for(uint d=lane;d<128;d+=32) {
        float v=values[d]*inv*norm[d];
        if(d<64) {
            float angle=float(m.y-3)*pow(1e7f,-float(d%32)/32.0f);
            uint partner=d<32 ? d+32 : d-32;
            float other=values[partner]*inv*norm[partner];
            v=v*cos(angle)+(d<32 ? -other : other)*sin(angle);
        }
        pooled[dst+d]=v;
    }
}
kernel void q4s_ring_commit(device const float* raw [[buffer(0)]],device float* ring [[buffer(1)]],
 device const uint4* meta [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/128,d=i%128;if(row>=p[2])return;uint4 m=meta[row];
    if(row+4>=m.w)ring[(m.x*4+m.y%4)*128+d]=raw[i];
}
// Four indexer heads share each key tile. Small visible windows bypass
// scoring, but still populate pooled keys for later sparse continuation.
kernel void q4s_score(device const float* q [[buffer(0)]],device const float* keys [[buffer(1)]],
 device const uint4* meta [[buffer(2)]],device float* scores [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint row=g.y,start=g.x*32,blocks=(meta[row].y+1)/4;
    if(blocks<=512 || start>=blocks)return;
    threadgroup float a[512],b[1024],dots[512];
    auto at=tensor(a,extents<int,32,16>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<128;base+=32) {
        for(uint i=tid;i<512;i+=128)a[i]=i/32<4 ? q[ulong(row)*512+(i/32)*128+base+i%32] : 0;
        for(uint i=tid;i<1024;i+=128) {
            uint block=start+i/32;
            b[i]=block<blocks ? keys[(ulong(meta[row].x)*p[1]+block)*128+base+i%32] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto dt=tensor(dots,extents<int,32,16>(),array<int,2>{1,32});acc.store(dt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<32 && start+tid<blocks) {
        float sum=0;bool valid=true;
        for(uint h=0;h<4;++h){float v=dots[h*32+tid];valid=valid && isfinite(v);sum+=max(v,0.0f);}
        scores[ulong(row)*p[1]+start+tid]=valid ? sum*rsqrt(128.0f) : NAN;
    }
}
// Exact radix threshold selection, not 512 serial argmax reductions. Scores
// are nonnegative F32; their uint bit order is monotonic. Four byte passes
// find the kth score. Equal scores elect the lower logical block first.
// Ordered compaction yields chronological KV traversal and no float atomics.
inline uint q4s_score_bits(float x) { return x==0.0f ? 0u : as_type<uint>(x); }
// A 256-row ordered scan needs only eight SIMD totals and two barriers,
// not eight full-threadgroup Hillis-Steele rounds per compaction pass.
inline uint q4s_scan(uint value,threadgroup uint* groups,uint tid) {
    uint prefix=simd_prefix_inclusive_sum(value);
    if(tid%32==31)groups[tid/32]=prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint g=0;g<tid/32;++g)prefix+=groups[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return prefix;
}
kernel void q4s_select(device const float* scores [[buffer(0)]],device const uint4* meta [[buffer(1)]],
 device uint* selected [[buffer(2)]],device uint* counts [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint blocks=(meta[row].y+1)/4;
    if(blocks<=512) {
        for(uint i=tid;i<512;i+=256)selected[ulong(row)*512+i]=i<blocks ? i : ~0u;
        if(tid==0)counts[row]=blocks;
        return;
    }
    threadgroup atomic_uint hist[256],bad;
    threadgroup uint prefix,rank,greater,total,equals,groups[8],equal_total,keep_total;
    if(tid==0){prefix=0;rank=512;atomic_store_explicit(&bad,0,memory_order_relaxed);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int shift=24;shift>=0;shift-=8) {
        atomic_store_explicit(&hist[tid],0,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint mask=shift==24 ? 0u : (~0u<<(shift+8));
        for(uint i=tid;i<blocks;i+=256) {
            float s=scores[ulong(row)*p[1]+i];
            if(!isfinite(s) || s<0)atomic_store_explicit(&bad,1,memory_order_relaxed);
            uint bits=q4s_score_bits(s);
            if((bits&mask)==prefix)atomic_fetch_add_explicit(&hist[(bits>>shift)&255],1,memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0)for(int bin=255;bin>=0;--bin) {
            uint n=atomic_load_explicit(&hist[bin],memory_order_relaxed);
            if(rank>n)rank-=n;else {prefix|=uint(bin)<<shift;break;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(atomic_load_explicit(&bad,memory_order_relaxed)) {
        for(uint i=tid;i<512;i+=256)selected[ulong(row)*512+i]=~0u;
        if(tid==0)counts[row]=~0u;return;
    }
    uint high=0;
    for(uint i=tid;i<blocks;i+=256)high+=q4s_score_bits(scores[ulong(row)*p[1]+i])>prefix;
    uint high_prefix=q4s_scan(high,groups,tid);
    if(tid==255)greater=high_prefix;
    if(tid==0){total=0;equals=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint base=0;base<blocks;base+=256) {
        uint i=base+tid,bits=i<blocks ? q4s_score_bits(scores[ulong(row)*p[1]+i]) : 0;
        uint eq=i<blocks && bits==prefix;
        uint eq_prefix=q4s_scan(eq,groups,tid);
        uint equal_rank=equals+eq_prefix;
        if(tid==255)equal_total=eq_prefix;
        bool keep=i<blocks && (bits>prefix || (eq && equal_rank<=512-greater));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint keep_prefix=q4s_scan(uint(keep),groups,tid);
        if(tid==255)keep_total=keep_prefix;
        if(keep)selected[ulong(row)*512+total+keep_prefix-1]=i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0){total+=keep_total;equals+=equal_total;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0)counts[row]=total;
}
inline uint q4s_token(device const uint* selected,uint row,uint i,uint blocks) {
    return i<min(blocks,512u)*4 ? selected[ulong(row)*512+i/4]*4+i%4 : blocks*4+i-min(blocks,512u)*4;
}
// One query's 12 GQA heads share sparse K/V tiles. F32 TensorOps operands
// and accumulation, F16 paged storage. Device scratch avoids the compiled
// 32KiB TG ceiling; each (row,KV head,split) owns one reusable 32x256 tile.
// MPP tensor's element type must be float, not const float; q stays read-only.
// Storage-only compaction. Every row owns its first partition; short logical
// spans own three extra partitions in a fixed eight-row tail for their slot.
// No tile, operand, reduction or per-sequence split election changes.
inline ulong q4s_compact_unit(uint row,uint split,device const uint4* meta,constant uint* p) {
    return split==0 ? ulong(row) : ulong(p[7])+3*(meta[row].x*8+row-meta[row].z)+split-1;
}
template<bool Compact>
inline ulong q4s_part_base(uint row,uint head,uint split,device const uint4* meta,constant uint* p) {
    if(Compact)return (q4s_compact_unit(row,split,meta,p)*24+head)*258;
    return ((ulong(row)*24+head)*p[4]+split)*258;
}
template<typename CacheT,bool PerSequence,bool Logical=false,bool Compact=false>
inline void q4s_attention_impl(device float* q,device const CacheT* kc,
 device const CacheT* vc,device const uint4* meta,device const uint* pages,
 device const uint* selected,device const uint* counts,device float* scratch,
 device float* parts,constant uint* p,uint3 g,uint tid,
 threadgroup float* scores,threadgroup float* prob,threadgroup float* maxima,threadgroup float* denom,threadgroup float* correction) {
    uint row=g.y,kh=g.x,split=g.z,blocks=(meta[row].y+1)/4;
    uint length=min(blocks,512u)*4+(meta[row].y+1)%4;
    // Sequence-local split election is part of the MLX numerical contract.
    // A decoding row must not switch from four partials to one when an
    // unrelated prefill joins the walk. GGUF retains its existing policy.
    uint splits=PerSequence ? (meta[row].w-meta[row].z<=8 ? 4u : 1u) : p[4];
    if(Logical)splits=((p[5+meta[row].x/32]>>(meta[row].x%32))&1) ? 4u : 1u;
    if(split>=splits) {
        if(!Compact && tid<12) {
            ulong dst=((ulong(row)*24+kh*12+tid)*p[4]+split)*258;
            parts[dst+256]=-INFINITY;parts[dst+257]=0;
        }
        return;
    }
    uint span=((length+splits*32-1)/(splits*32))*32,first=split*span,last=min(first+span,length);
    device float* kv=scratch+(Compact ? q4s_compact_unit(row,split,meta,p)*2+kh : (ulong(row)*2+kh)*p[4]+split)*8192;
    auto tq=tensor(q+ulong(row)*6144+kh*3072,extents<int,256,12>(),array<int,2>{1,256});
    auto tk=tensor(kv,extents<int,256,32>(),array<int,2>{1,256});
    auto tv=tensor(kv,extents<int,32,256>(),array<int,2>{1,32});
    auto tp=tensor(prob,extents<int,32,16>(),array<int,2>{1,32});
    auto ts=tensor(scores,extents<int,32,16>(),array<int,2>{1,32});
    constexpr auto qkd=matmul2d_descriptor(16,32,256,false,true);
    constexpr auto pvd=matmul2d_descriptor(16,256,32,false,true);
    matmul2d<qkd,execution_simdgroups<4>> qk;matmul2d<pvd,execution_simdgroups<4>> pv;
    auto acc=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    if(tid<16){maxima[tid]=-INFINITY;denom[tid]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint base=first;base<last && counts[row]<=512;base+=32) {
        for(uint i=tid;i<8192;i+=128) {
            uint t=base+i/256;float v=0;
            if(t<last) {
                uint token=q4s_token(selected,row,t,blocks);
                uint physical=pages[meta[row].x*p[0]+token/16]*16+token%16;
                v=float(kc[ulong(physical)*512+kh*256+i%256]);
            }
            kv[i]=v;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        auto dot=qk.get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();qk.run(tq,tk,dot);dot.store(ts);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint h=tid/8,lane=tid%8;float hi=maxima[h];
        for(uint j=lane;j<32;j+=8)if(base+j<last && h<12)hi=max(hi,scores[h*32+j]*0.0625f);
        hi=max(hi,simd_shuffle_xor(hi,1));hi=max(hi,simd_shuffle_xor(hi,2));hi=max(hi,simd_shuffle_xor(hi,4));
        float sum=0,old=isfinite(maxima[h]) ? exp(maxima[h]-hi) : 0;
        for(uint j=lane;j<32;j+=8){float v=h<12 && base+j<last ? exp(scores[h*32+j]*0.0625f-hi) : 0;prob[h*32+j]=v;sum+=v;}
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,4);
        if(lane==0){maxima[h]=hi;denom[h]=denom[h]*old+sum;correction[h]=old;}
        for(uint i=tid;i<8192;i+=128) {
            uint t=base+i%32;float v=0;
            if(t<last) {
                uint token=q4s_token(selected,row,t,blocks);
                uint physical=pages[meta[row].x*p[0]+token/16]*16+token%16;
                v=float(vc[ulong(physical)*512+kh*256+i/32]);
            }
            kv[i]=v;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        auto product=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();pv.run(tp,tv,product);
        uint i=0;for(auto it=product.begin();it!=product.end();++it,++i)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();acc[i]=acc[i]*correction[ij[1]]+*it;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();if(ij[1]<12)
            parts[q4s_part_base<Compact>(row,kh*12+ij[1],split,meta,p)+ij[0]]=*it;
    }
    if(tid<12){ulong dst=q4s_part_base<Compact>(row,kh*12+tid,split,meta,p);
        parts[dst+256]=maxima[tid];parts[dst+257]=counts[row]<=512 ? denom[tid] : NAN;}
}
#define Q4S_ATTENTION(NAME,T,PerSequence,Logical,Compact) \
kernel void NAME(device float* q [[buffer(0)]],device const T* kc [[buffer(1)]], \
 device const T* vc [[buffer(2)]],device const uint4* meta [[buffer(3)]],device const uint* pages [[buffer(4)]], \
 device const uint* selected [[buffer(5)]],device const uint* counts [[buffer(6)]],device float* scratch [[buffer(7)]], \
 device float* parts [[buffer(8)]],constant uint* p [[buffer(9)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float scores[512],prob[512],maxima[16],denom[16],correction[16]; \
 q4s_attention_impl<T,PerSequence,Logical,Compact>(q,kc,vc,meta,pages,selected,counts,scratch,parts,p,g,tid,scores,prob,maxima,denom,correction); }
Q4S_ATTENTION(q4s_attention,half,false,false,false)
Q4S_ATTENTION(q4b_attention,bfloat,true,false,false)
Q4S_ATTENTION(q4b_attention_contract,bfloat,true,true,false)
Q4S_ATTENTION(q4b_attention_compact,bfloat,true,true,true)
#undef Q4S_ATTENTION

// On-chip KV staging for decode and prefill passes. BF16 is
// exactly the cache storage type; Q, probabilities, softmax and accumulators
// remain F32. The same 32-token tiles, split boundaries and joins are retained.
// Load: 0 = old 64-bit vectors; 1 = wider-only test comparator;
// 2 = 128-bit vectors with SIMD-shared physical addresses for K and V.
// K keeps an aligned vector pitch; V's independent pitch spreads its scalar
// transposed writes without changing logical tensors, contractions or joins.
template<uint Pad,uint Load=0,uint VPad=Pad,uint Direct=0>
inline void q4b_attention_local_impl(device float* q,device const bfloat* kc,
 device const bfloat* vc,device const uint4* meta,device const uint* pages,
 device const uint* selected,device const uint* counts,
 device float* parts,constant uint* p,uint3 g,uint tid,threadgroup bfloat* kv,
 threadgroup float* scores,threadgroup float* prob,threadgroup float* maxima,
 threadgroup float* denom,threadgroup float* correction) {
    uint row=g.y,kh=g.x,split=g.z,blocks=(meta[row].y+1)/4;
    uint length=min(blocks,512u)*4+(meta[row].y+1)%4;
    uint splits=((p[5+meta[row].x/32]>>(meta[row].x%32))&1) ? 4u : 1u;
    if(split>=splits)return;
    uint span=((length+splits*32-1)/(splits*32))*32,first=split*span,last=min(first+span,length);
    auto tq=tensor(q+ulong(row)*6144+kh*3072,extents<int,256,12>(),array<int,2>{1,256});
    auto tk=tensor(kv,extents<int,256,32>(),array<int,2>{1,256+Pad});
    auto tv=tensor(kv,extents<int,32,256>(),array<int,2>{1,32+VPad});
    auto tp=tensor(prob,extents<int,32,16>(),array<int,2>{1,32});
    auto ts=tensor(scores,extents<int,32,16>(),array<int,2>{1,32});
    constexpr auto qkd=matmul2d_descriptor(16,32,256,false,true);
    constexpr auto pvd=matmul2d_descriptor(16,256,32,false,true);
    matmul2d<qkd,execution_simdgroups<4>> qk;matmul2d<pvd,execution_simdgroups<4>> pv;
    auto acc=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    if(tid<16){maxima[tid]=-INFINITY;denom[tid]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint base=first;base<last && counts[row]<=512;base+=32) {
        // Each SIMD lane resolves one token once, then shares the physical
        // address with the vector loaders for both K and V. No shared-memory
        // address table or extra barrier; only integer addressing changes.
        uint physical_lane=0;
        if(Load==2 && base+tid%32<last) {
            uint token=q4s_token(selected,row,base+tid%32,blocks);
            physical_lane=pages[meta[row].x*p[0]+token/16]*16+token%16;
        }
        // Every SIMD resolves the same 32 addresses, so this election is
        // uniform across all four participating SIMD groups. Partial runs,
        // sparse gaps and permuted pages retain the original gather path.
        uint physical_first=simd_shuffle(physical_lane,0);
        bool contiguous=Direct!=0 && base+32<=last && simd_all(physical_lane==physical_first+tid%32);
        bool direct=contiguous && (Direct&1)!=0;
        if(!direct && Load!=0) {
            for(uint i=tid*8;i<8192;i+=1024) {
                uint physical=Load==2 ? simd_shuffle(physical_lane,i/256) : 0;
                uint4 v=0;
                if(base+i/256<last) {
                    if(Load==1) {
                        uint token=q4s_token(selected,row,base+i/256,blocks);
                        physical=pages[meta[row].x*p[0]+token/16]*16+token%16;
                    }
                    v=*reinterpret_cast<device const uint4*>(kc+ulong(physical)*512+kh*256+i%256);
                }
                *reinterpret_cast<threadgroup uint4*>(kv+i/256*(256+Pad)+i%256)=v;
            }
        } else if(!direct) {
            for(uint i=tid*4;i<8192;i+=512) {
                uint t=base+i/256; bfloat4 v=0;
                if(t<last) {
                    uint token=q4s_token(selected,row,t,blocks);
                    uint physical=pages[meta[row].x*p[0]+token/16]*16+token%16;
                    v=*reinterpret_cast<device const bfloat4*>(kc+ulong(physical)*512+kh*256+i%256);
                }
                *reinterpret_cast<threadgroup bfloat4*>(kv+i/256*(256+Pad)+i%256)=v;
            }
        }
        if(!direct)threadgroup_barrier(mem_flags::mem_threadgroup);
        auto dot=qk.get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        if(direct) {
            // MPP input tensor element types must be unqualified. The cast
            // does not authorize stores: keys/values are read-only operands.
            auto keys=tensor(const_cast<device bfloat*>(kc)+ulong(physical_first)*512+kh*256,extents<int,256,32>(),array<int,2>{1,512});
            qk.run(tq,keys,dot);
        } else qk.run(tq,tk,dot);
        dot.store(ts);
        direct=contiguous && (Direct&2)!=0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint h=tid/8,lane=tid%8;float hi=maxima[h];
        for(uint j=lane;j<32;j+=8)if(base+j<last && h<12)hi=max(hi,scores[h*32+j]*0.0625f);
        hi=max(hi,simd_shuffle_xor(hi,1));hi=max(hi,simd_shuffle_xor(hi,2));hi=max(hi,simd_shuffle_xor(hi,4));
        float sum=0,old=isfinite(maxima[h]) ? exp(maxima[h]-hi) : 0;
        for(uint j=lane;j<32;j+=8){float v=h<12 && base+j<last ? exp(scores[h*32+j]*0.0625f-hi) : 0;prob[h*32+j]=v;sum+=v;}
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,4);
        if(lane==0){maxima[h]=hi;denom[h]=denom[h]*old+sum;correction[h]=old;}
        if(!direct && Load!=0) {
            for(uint i=tid*8;i<8192;i+=1024) {
                uint physical=Load==2 ? simd_shuffle(physical_lane,i/256) : 0;
                uint4 bits=0;
                if(base+i/256<last) {
                    if(Load==1) {
                        uint token=q4s_token(selected,row,base+i/256,blocks);
                        physical=pages[meta[row].x*p[0]+token/16]*16+token%16;
                    }
                    bits=*reinterpret_cast<device const uint4*>(vc+ulong(physical)*512+kh*256+i%256);
                }
                bfloat4 lo=as_type<bfloat4>(bits.xy),hi=as_type<bfloat4>(bits.zw);
                for(uint j=0;j<4;++j) {
                    kv[(i%256+j)*(32+VPad)+i/256]=lo[j];
                    kv[(i%256+4+j)*(32+VPad)+i/256]=hi[j];
                }
            }
        } else if(!direct) {
            for(uint i=tid*4;i<8192;i+=512) {
                uint t=base+i/256; bfloat4 v=0;
                if(t<last) {
                    uint token=q4s_token(selected,row,t,blocks);
                    uint physical=pages[meta[row].x*p[0]+token/16]*16+token%16;
                    v=*reinterpret_cast<device const bfloat4*>(vc+ulong(physical)*512+kh*256+i%256);
                }
                for(uint j=0;j<4;++j)kv[(i%256+j)*(32+VPad)+i/256]=v[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(direct) {
            // Keep V's contiguous dimension as the tensor's inner axis and
            // change the contraction's transpose flag instead of constructing
            // a non-unit-inner-stride view. M/N and the accumulator layout
            // stay fixed; the same 32 terms and online-softmax order remain.
            auto values=tensor(const_cast<device bfloat*>(vc)+ulong(physical_first)*512+kh*256,extents<int,256,32>(),array<int,2>{1,512});
            constexpr auto dd=matmul2d_descriptor(16,256,32,false,false);
            matmul2d<dd,execution_simdgroups<4>> dop;
            auto product=dop.get_destination_cooperative_tensor<decltype(tp),decltype(values),float>();
            dop.run(tp,values,product);
            uint i=0;for(auto it=product.begin();it!=product.end();++it,++i)if(it.is_valid_element()) {
                auto ij=it.get_multidimensional_index();acc[i]=acc[i]*correction[ij[1]]+*it;
            }
        } else {
            auto product=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();pv.run(tp,tv,product);
            uint i=0;for(auto it=product.begin();it!=product.end();++it,++i)if(it.is_valid_element()) {
                auto ij=it.get_multidimensional_index();acc[i]=acc[i]*correction[ij[1]]+*it;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();if(ij[1]<12)
            parts[q4s_part_base<true>(row,kh*12+ij[1],split,meta,p)+ij[0]]=*it;
    }
    if(tid<12){ulong dst=q4s_part_base<true>(row,kh*12+tid,split,meta,p);
        parts[dst+256]=maxima[tid];parts[dst+257]=counts[row]<=512 ? denom[tid] : NAN;}
}
#define Q4B_ATTENTION_LOCAL(Name,Pad,Load,VPad,Direct) \
kernel void Name(device float* q [[buffer(0)]],device const bfloat* kc [[buffer(1)]], \
 device const bfloat* vc [[buffer(2)]],device const uint4* meta [[buffer(3)]],device const uint* pages [[buffer(4)]], \
 device const uint* selected [[buffer(5)]],device const uint* counts [[buffer(6)]],device float* scratch [[buffer(7)]], \
 device float* parts [[buffer(8)]],constant uint* p [[buffer(9)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    alignas(16) threadgroup bfloat kv[32*(256+Pad)>256*(32+VPad) ? 32*(256+Pad) : 256*(32+VPad)]; \
    threadgroup float scores[512],prob[512],maxima[16],denom[16],correction[16]; \
    q4b_attention_local_impl<Pad,Load,VPad,Direct>(q,kc,vc,meta,pages,selected,counts,parts,p,g,tid,kv,scores,prob,maxima,denom,correction); }
Q4B_ATTENTION_LOCAL(q4b_attention_local,0,0,0,false)
Q4B_ATTENTION_LOCAL(q4b_attention_local_pad,8,0,8,false)
Q4B_ATTENTION_LOCAL(q4b_attention_local_wide,8,1,8,false)
Q4B_ATTENTION_LOCAL(q4b_attention_local_gather,8,2,8,false)
Q4B_ATTENTION_LOCAL(q4b_attention_local_vpad1,8,2,1,false)
Q4B_ATTENTION_LOCAL(q4b_attention_direct_runs,8,2,1,3)
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
Q4B_ATTENTION_LOCAL(q4b_attention_direct_keys,8,2,1,1)
Q4B_ATTENTION_LOCAL(q4b_attention_direct_values,8,2,1,2)
#endif
#undef Q4B_ATTENTION_LOCAL

#ifdef PADDOCK_KERNEL_DIAGNOSTICS
// Adjacent dense-prefill queries share KV tiles. Pack only query addresses;
// their F32 values and four padded heads per position remain unchanged.
kernel void q4b_attention_pair_pack(device const float* q [[buffer(0)]],device float* packed [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint groups=(p[0]+1)/2;
    if(i>=groups*16384)return;
    uint group=i/16384,kh=(i/8192)%2,h=(i/256)%32,d=i%256;
    uint row=group*2+h/16;
    packed[i]=row<p[0] && h%16<12 ? q[ulong(row)*6144+kh*3072+(h%16)*256+d] : 0;
}
kernel void q4b_attention_pairs(device float* q [[buffer(0)]],device const bfloat* kc [[buffer(1)]],
 device const bfloat* vc [[buffer(2)]],device const uint4* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],
 device const uint* selected [[buffer(5)]],device const uint* counts [[buffer(6)]],device float* packed [[buffer(7)]],
 device float* parts [[buffer(8)]],constant uint* p [[buffer(9)]],uint3 g [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
    alignas(16) threadgroup bfloat kv[32*264];
    threadgroup float scores[1024],prob[1024],maxima[32],denom[32],correction[32];
    uint row=g.y*2,kh=g.x;
    bool pair=row+1<p[2] && meta[row].x==meta[row+1].x && meta[row+1].y==meta[row].y+1
        && meta[row+1].y<2048 && counts[row]<=512 && counts[row+1]<=512
        && ((p[5+meta[row].x/32]>>(meta[row].x%32))&1)==0;
    if(!pair) {
        for(uint r=row;r<min(row+2,p[2]);++r) {
            q4b_attention_local_impl<8,2,1,3>(q,kc,vc,meta,pages,selected,counts,parts,p,
                uint3(kh,r,g.z),tid,kv,scores,prob,maxima,denom,correction);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }
    if(g.z!=0)return;
    uint first_length=meta[row].y+1,last=meta[row+1].y+1;
    auto tq=tensor(packed+ulong(g.y)*16384+kh*8192,extents<int,256,32>(),array<int,2>{1,256});
    auto tk=tensor(kv,extents<int,256,32>(),array<int,2>{1,264});
    auto tv=tensor(kv,extents<int,32,256>(),array<int,2>{1,33});
    auto tp=tensor(prob,extents<int,32,32>(),array<int,2>{1,32});
    auto ts=tensor(scores,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto qkd=matmul2d_descriptor(32,32,256,false,true);
    constexpr auto pvd=matmul2d_descriptor(32,256,32,false,true);
    matmul2d<qkd,execution_simdgroups<4>> qk;matmul2d<pvd,execution_simdgroups<4>> pv;
    auto acc=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    if(tid<32){maxima[tid]=-INFINITY;denom[tid]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint base=0;base<last;base+=32) {
        uint token=base+tid%32,physical=0;
        if(token<last)physical=pages[meta[row].x*p[0]+token/16]*16+token%16;
        uint first=simd_shuffle(physical,0);
        bool direct=base+32<=last && simd_all(physical==first+tid%32);
        if(!direct) {
            for(uint i=tid*8;i<8192;i+=1024) {
                uint address=simd_shuffle(physical,i/256);uint4 bits=0;
                if(base+i/256<last)bits=*reinterpret_cast<device const uint4*>(kc+ulong(address)*512+kh*256+i%256);
                *reinterpret_cast<threadgroup uint4*>(kv+i/256*264+i%256)=bits;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        auto dot=qk.get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        if(direct) {
            auto keys=tensor(const_cast<device bfloat*>(kc)+ulong(first)*512+kh*256,extents<int,256,32>(),array<int,2>{1,512});
            qk.run(tq,keys,dot);
        } else qk.run(tq,tk,dot);
        dot.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint h=tid/8;h<32;h+=16) {
            uint lane=tid%8,end=h<16 ? first_length : last;float hi=maxima[h];
            for(uint j=lane;j<32;j+=8)if(base+j<end && h%16<12)hi=max(hi,scores[h*32+j]*0.0625f);
            hi=max(hi,simd_shuffle_xor(hi,1));hi=max(hi,simd_shuffle_xor(hi,2));hi=max(hi,simd_shuffle_xor(hi,4));
            float sum=0,old=isfinite(maxima[h]) ? exp(maxima[h]-hi) : 0;
            for(uint j=lane;j<32;j+=8){float v=h%16<12 && base+j<end ? exp(scores[h*32+j]*0.0625f-hi) : 0;prob[h*32+j]=v;sum+=v;}
            sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,4);
            if(lane==0){maxima[h]=hi;denom[h]=denom[h]*old+sum;correction[h]=old;}
        }
        if(!direct) {
            for(uint i=tid*8;i<8192;i+=1024) {
                uint address=simd_shuffle(physical,i/256);uint4 bits=0;
                if(base+i/256<last)bits=*reinterpret_cast<device const uint4*>(vc+ulong(address)*512+kh*256+i%256);
                bfloat4 lo=as_type<bfloat4>(bits.xy),hi=as_type<bfloat4>(bits.zw);
                for(uint j=0;j<4;++j){kv[(i%256+j)*33+i/256]=lo[j];kv[(i%256+4+j)*33+i/256]=hi[j];}
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(direct) {
            auto values=tensor(const_cast<device bfloat*>(vc)+ulong(first)*512+kh*256,extents<int,256,32>(),array<int,2>{1,512});
            constexpr auto desc=matmul2d_descriptor(32,256,32,false,false);
            matmul2d<desc,execution_simdgroups<4>> op;
            auto product=op.get_destination_cooperative_tensor<decltype(tp),decltype(values),float>();op.run(tp,values,product);
            uint i=0;for(auto it=product.begin();it!=product.end();++it,++i)if(it.is_valid_element()) {
                auto ij=it.get_multidimensional_index();
                if(base<first_length || ij[1]>=16)acc[i]=acc[i]*correction[ij[1]]+*it;
            }
        } else {
            auto product=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();pv.run(tp,tv,product);
            uint i=0;for(auto it=product.begin();it!=product.end();++it,++i)if(it.is_valid_element()) {
                auto ij=it.get_multidimensional_index();
                if(base<first_length || ij[1]>=16)acc[i]=acc[i]*correction[ij[1]]+*it;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();uint h=ij[1];
        if(h%16<12)parts[q4s_part_base<true>(row+h/16,kh*12+h%16,0,meta,p)+ij[0]]=*it;
    }
    if(tid<32 && tid%16<12) {
        ulong dst=q4s_part_base<true>(row+tid/16,kh*12+tid%16,0,meta,p);
        parts[dst+256]=maxima[tid];parts[dst+257]=denom[tid];
    }
}
#endif

kernel void q4b_join_gate_compact(device const float* parts [[buffer(0)]],device const float* qg [[buffer(1)]],
    device float* y [[buffer(2)]],device const uint4* meta [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint rh [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint row=rh/24,head=rh%24;
    uint splits=((p[5+meta[row].x/32]>>(meta[row].x%32))&1) ? 4u : 1u;
    float hi=-INFINITY,den=0;
    for(uint s=0;s<splits;++s)hi=max(hi,parts[q4s_part_base<true>(row,head,s,meta,p)+256]);
    float acc[8];for(uint j=0;j<8;++j)acc[j]=0;
    for(uint s=0;s<splits;++s) {
        ulong src=q4s_part_base<true>(row,head,s,meta,p);float d=parts[src+257];if(d==0)continue;
        float factor=exp(parts[src+256]-hi);den+=d*factor;
        for(uint j=0;j<8;++j)acc[j]+=parts[src+lane+j*32]*factor;
    }
    for(uint j=0;j<8;++j){uint d=lane+j*32;y[ulong(rh)*256+d]=mlx_bf(mlx_bf(acc[j]/den)*mlx_sigmoid_bf(qg[ulong(rh)*512+256+d]));}
}
kernel void q4s_join_gate(device const float* parts [[buffer(0)]],device const float* qg [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],
 uint rh [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong base=ulong(rh)*p[4]*258;float hi=-INFINITY,den=0;
    for(uint s=0;s<p[4];++s)hi=max(hi,parts[base+s*258+256]);
    float acc[8];for(uint j=0;j<8;++j)acc[j]=0;
    for(uint s=0;s<p[4];++s) {
        ulong src=base+s*258;float d=parts[src+257];if(d==0)continue;
        float factor=exp(parts[src+256]-hi);den+=d*factor;
        for(uint j=0;j<8;++j)acc[j]+=parts[src+lane+j*32]*factor;
    }
    for(uint j=0;j<8;++j){uint d=lane+j*32;out[ulong(rh)*256+d]=(acc[j]/den)/(1.0f+exp(-qg[ulong(rh)*512+256+d]));}
}
// p extends common fields with source slot, target slot, prefix length,
// reset flag. Stale post-prefix bytes are never read after logical reset.
kernel void q4s_copy(device half* kc [[buffer(0)]],device half* vc [[buffer(1)]],device float* pooled [[buffer(2)]],
 device float* ring [[buffer(3)]],device const uint* pages [[buffer(4)]],constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    uint source=p[5],target=p[6],length=p[7];
    if(i<length*512) {
        uint t=i/512,d=i%512,dp=pages[target*p[0]+t/16]*16+t%16;
        if(p[8]){kc[ulong(dp)*512+d]=0;vc[ulong(dp)*512+d]=0;}
        else {uint sp=pages[source*p[0]+t/16]*16+t%16;kc[ulong(dp)*512+d]=kc[ulong(sp)*512+d];vc[ulong(dp)*512+d]=vc[ulong(sp)*512+d];}
    }
    if(i<(length/4)*128)pooled[ulong(target)*p[1]*128+i]=p[8] ? 0 : pooled[ulong(source)*p[1]*128+i];
    if(i<512)ring[target*512+i]=p[8] ? 0.0f : ring[source*512+i];
}
