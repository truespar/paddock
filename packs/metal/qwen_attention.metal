// Head-256 Qwen attention uses a 32x32 tile to stay inside 32 KiB of
// threadgroup memory. Same online-softmax composition as the Granite path.
// p: Q heads, KV heads, page stride, rope base, rms epsilon, rotary dims.
// GGUF Q projection interleaves [query, gate] within each head. Text M-RoPE
// has identical positions in every section: it reduces to partial split-half
// rotary, not Granite's adjacent-pair convention.
kernel void qwen_qnorm_rope(device const float* input [[buffer(0)]],device const float* norm [[buffer(1)]],
                            device const uint* meta [[buffer(2)]],device float* out [[buffer(3)]],
                            constant uint* p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    ulong src=(ulong(group.y)*p[0]+group.x)*512;
    float sum=0;for(uint d=lane;d<256;d+=32)sum+=input[src+d]*input[src+d];
    float inv=rsqrt(simd_sum(sum)/256.0f+as_type<float>(p[4]));
    ulong dst=(ulong(group.y)*p[0]+group.x)*256;
    for(uint d=lane;d<256;d+=32) {
        float v=input[src+d]*inv*norm[d];
        if(d<p[5]) {
            uint j=d%(p[5]/2),other=d<p[5]/2 ? d+p[5]/2 : d-p[5]/2;
            uint axis=j%3==1 && j<33 ? 1 : (j%3==2 && j<30 ? 2 : 0);
            float angle=float(meta[group.y*4+axis])*pow(as_type<float>(p[3]),-2.0f*float(j)/float(p[5]));
            float partner=input[src+other]*inv*norm[other];
            v=v*cos(angle)+(d<p[5]/2 ? -partner : partner)*sin(angle);
        }
        out[dst+d]=v;
    }
}
kernel void qwen_knorm_store(device const float* input [[buffer(0)]],device const float* value [[buffer(1)]],
                             device const float* norm [[buffer(2)]],device const uint* meta [[buffer(3)]],
                             device const uint* pages [[buffer(4)]],device half* keys [[buffer(5)]],
                             device half* values [[buffer(6)]],device const uint* mrope [[buffer(7)]],constant uint* p [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong src=(ulong(group.y)*p[1]+group.x)*256;
    uint slot=meta[group.y*2],pos=meta[group.y*2+1];
    uint physical=pages[slot*p[2]+pos/16]*16+pos%16;
    ulong dst=(ulong(physical)*p[1]+group.x)*256;
    float sum=0;for(uint d=lane;d<256;d+=32)sum+=input[src+d]*input[src+d];
    float inv=rsqrt(simd_sum(sum)/256.0f+as_type<float>(p[4]));
    for(uint d=lane;d<256;d+=32) {
        float v=input[src+d]*inv*norm[d];
        if(d<p[5]) {
            uint j=d%(p[5]/2),other=d<p[5]/2 ? d+p[5]/2 : d-p[5]/2;
            uint axis=j%3==1 && j<33 ? 1 : (j%3==2 && j<30 ? 2 : 0);
            float angle=float(mrope[group.y*4+axis])*pow(as_type<float>(p[3]),-2.0f*float(j)/float(p[5]));
            float partner=input[src+other]*inv*norm[other];
            v=v*cos(angle)+(d<p[5]/2 ? -partner : partner)*sin(angle);
        }
        keys[dst+d]=half(v);values[dst+d]=half(value[src+d]);
    }
}
kernel void qwen_attn_gate(device float* x [[buffer(0)]],device const float* qgate [[buffer(1)]],
                           constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]/=1.0f+exp(-qgate[ulong(i/256)*512+256+i%256]);
}

// Four (9B), six (27B), or eight (35B MoE) query heads share each KV tile. GQA
// keeps the register footprint and loops specialized to each elected shape.
// Split-K supplies parallelism at c=1;
// the stable (numerator,max,denominator) merge also handles empty splits.
// p: heads, KV heads, page stride, scale, splits. selected row IDs at buffer 5.
template<typename KV,uint GQA,bool Verify=false>
inline void qwen_decode_impl(device const float* q,device const KV* keys,device const KV* values,
    device const uint* meta,device const uint* pages,device const uint* rows,device float* out,
    device const uint* limits,constant uint* p,uint3 group,uint tid,uint lane,uint sg,
    threadgroup float* scores,threadgroup float* prob,threadgroup float* highs,threadgroup float* sums) {
    uint row=rows[group.y],kh=group.x,slot=meta[row*2],length=limits[row]+1;
    // Candidate rows can straddle a split-count boundary. Keep each row's
    // ordinary decode partitioning; p[4] remains the common output stride.
    uint splits=p[4];
    if constexpr(Verify)splits=min(p[4],max((length+127)/128,p[5]));
    uint span=(length+splits-1)/splits,first=group.z*span,last=min(first+span,length);
    float2 acc[GQA];float maximum[GQA],denom[GQA];
    for(uint h=0;h<GQA;++h){acc[h]=0;maximum[h]=-INFINITY;denom[h]=0;}
    for(uint base=first;base<last;base+=32) {
        uint token=base+tid/4,dl=tid%4;
        float dotq[GQA];for(uint h=0;h<GQA;++h)dotq[h]=0;
        if(token<last) {
            uint physical=pages[slot*p[2]+token/16]*16+token%16;
            for(uint j=0;j<16;++j) {
                uint d=dl*4+j*16;
                float4 key=float4(*reinterpret_cast<device const vec<KV,4>*>(keys+(ulong(physical)*p[1]+kh)*256+d));
                for(uint h=0;h<GQA;++h)dotq[h]+=dot(key,*reinterpret_cast<device const float4*>(q+(ulong(row)*p[0]+kh*GQA+h)*256+d));
            }
        }
        for(uint h=0;h<GQA;++h) {
            dotq[h]+=simd_shuffle_xor(dotq[h],1);dotq[h]+=simd_shuffle_xor(dotq[h],2);
            if(dl==0)scores[h*32+tid/4]=token<last ? dotq[h]*as_type<float>(p[3]) : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint h=sg;h<GQA;h+=4) {
            float v=scores[h*32+lane],hi=simd_max(v),pr=isfinite(hi) ? exp(v-hi) : 0.0f;
            float sum=simd_sum(pr);prob[h*32+lane]=pr;
            if(lane==0){highs[h]=hi;sums[h]=sum;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float multiplier[GQA];
        for(uint h=0;h<GQA;++h) {
            float next=max(maximum[h],highs[h]);
            float old=isfinite(maximum[h]) ? exp(maximum[h]-next) : 0.0f;
            multiplier[h]=exp(highs[h]-next);acc[h]*=old;
            denom[h]=denom[h]*old+sums[h]*multiplier[h];maximum[h]=next;
        }
        for(uint j=0;j<min(32u,last-base);++j) {
            uint t=base+j,physical=pages[slot*p[2]+t/16]*16+t%16;
            ulong vi=(ulong(physical)*p[1]+kh)*256+tid;
            float2 v=float2(values[vi],values[vi+128]);
            for(uint h=0;h<GQA;++h)acc[h]+=v*(prob[h*32+j]*multiplier[h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint h=0;h<GQA;++h) {
        ulong dst=((ulong(group.y)*p[0]+kh*GQA+h)*p[4]+group.z)*258;
        out[dst+tid]=acc[h].x;out[dst+tid+128]=acc[h].y;
        if(tid==0){out[dst+256]=maximum[h];out[dst+257]=denom[h];}
    }
}
#define QWEN_DECODE(NAME,KV,GQA,VERIFY) \
kernel void NAME(device const float* q [[buffer(0)]],device const KV* keys [[buffer(1)]],device const KV* values [[buffer(2)]], \
device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]], \
device float* out [[buffer(6)]],device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]], \
uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) { \
threadgroup float scores[GQA*32],prob[GQA*32],highs[GQA],sums[GQA]; \
qwen_decode_impl<KV,GQA,VERIFY>(q,keys,values,meta,pages,rows,out,limits,p,group,tid,lane,sg,scores,prob,highs,sums);}
QWEN_DECODE(qwen_attention_decode,half,6,false)
QWEN_DECODE(qwen_attention_decode_gqa4,half,4,false)
QWEN_DECODE(qwen_attention_decode_gqa8,half,8,false)
QWEN_DECODE(mlx_attention_decode,bfloat,6,false)
QWEN_DECODE(mlx_attention_decode_gqa4,bfloat,4,false)
QWEN_DECODE(mlx_attention_verify,bfloat,6,true)
QWEN_DECODE(mlx_attention_stable,bfloat,6,true)
QWEN_DECODE(mlx_attention_stable_gqa4,bfloat,4,true)
QWEN_DECODE(bonsai_attention_decode,float,6,true)
#undef QWEN_DECODE
kernel void qwen_attention_merge(device const float* parts [[buffer(0)]],device float* out [[buffer(1)]],
                                 device const uint* rows [[buffer(2)]],constant uint* p [[buffer(3)]],
                                 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint rh=rows[group/p[0]]*p[0]+group%p[0];ulong base=ulong(group)*p[1]*258;
    float high=-INFINITY;for(uint s=0;s<p[1];++s)high=max(high,parts[base+s*258+256]);
    float acc[8];for(uint j=0;j<8;++j)acc[j]=0;float denom=0;
    for(uint s=0;s<p[1];++s) {
        ulong src=base+s*258;if(parts[src+257]<=0)continue;
        float c=exp(parts[src+256]-high);denom+=parts[src+257]*c;
        for(uint j=0;j<8;++j)acc[j]+=parts[src+lane+j*32]*c;
    }
    for(uint j=0;j<8;++j)out[ulong(rh)*256+lane+j*32]=acc[j]/denom;
}

template<uint Splits=1,typename KV=half,uint Block=32,bool DeviceTile=false,bool Interleaved=false,bool Indexed=false,typename Tile,typename A=KV>
inline void qwen_prefill_tile(device A* q, device const KV* kc, device const KV* vc,
                         device const uint* meta, device const uint* pages, device float* out, device const uint* limits,
                         constant uint* p, uint head, uint first, uint count, uint tid,
                         Tile kv, threadgroup A* probability, threadgroup float* scores,
                         threadgroup float* maximum, threadgroup float* denominator, threadgroup float* correction,uint part=0) {
    uint kh=head/(p[0]/p[1]),slot=meta[2*first];
    constexpr auto tile_fence=DeviceTile ? mem_flags::mem_threadgroup | mem_flags::mem_device : mem_flags::mem_threadgroup;
    uint lastpos=limits[first+count-1],width=p[0]*256,kvwidth=p[1]*256;
    auto tq=tensor(q+ulong(first)*width+head*256,dextents<int,2>(256,count),array<int,2>{1,int(width)});
    auto tk=tensor(kv,extents<int,256,Block>(),array<int,2>{1,256});
    // The packed path consumes one physical BF16 page directly. Its V operand
    // remains token-major, so the matrix descriptor does not transpose B.
    auto tv=tensor(kv,dextents<int,2>{Interleaved?256:Block,Interleaved?Block:256},array<int,2>{1,Interleaved?256:Block});
    auto tp=tensor(probability,extents<int,Block,32>(),array<int,2>{1,Block});
    auto ts=tensor(scores,extents<int,Block,32>(),array<int,2>{1,Block});
    constexpr auto qk_desc=matmul2d_descriptor(32,Block,256,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc=matmul2d_descriptor(32,256,Block,false,!Interleaved,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc,execution_simdgroups<4>> qk;
    matmul2d<pv_desc,execution_simdgroups<4>> pv;
    auto accum=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<accum.get_capacity();++i)accum[i]=0;
    if(tid<32){maximum[tid]=-INFINITY;denominator[tid]=0;}
    uint span=((lastpos+1+Splits*Block-1)/(Splits*Block))*Block;
    // Fixed block ownership is invariant to the number of query rows. A
    // cached suffix and an uncached prompt therefore share the reduction.
    uint first_block=Interleaved?part*Block:part*span;
    uint end_block=Interleaved?lastpos+1:min((part+1)*span,lastpos+1);
    for(uint base=first_block;base<end_block;base+=Interleaved?Splits*Block:Block) {
        auto score=qk.template get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        for(uint i=0;i<score.get_capacity();++i)score[i]=0;
        if constexpr(Interleaved) {
            static_assert(Block==16,"direct operands must fit one physical page");
            uint physical=pages[slot*p[2]+base/16]*16;
            auto page=tensor(const_cast<device KV*>(kc)+ulong(physical)*kvwidth+kh*256,
                dextents<int,2>{256,int(min(Block,lastpos-base+1))},array<int,2>{1,int(kvwidth)});
            qk.run(tq,page,score);
        } else {
            for(uint i=tid;i<Block*256;i+=128) {
                uint t=base+i/256,d=i%256;
                uint physical=t<=lastpos ? pages[slot*p[2]+t/16]*16+t%16 : 0;
                kv[i]=t<=lastpos ? A(kc[ulong(physical)*kvwidth+kh*256+d]) : A(0);
            }
            threadgroup_barrier(tile_fence);
            qk.run(tq,tk,score);
        }
        score.store(ts);
        threadgroup_barrier(tile_fence);
        uint row=tid/4,lane=tid%4,pos=row<count ? limits[first+row] : 0;
        float high=maximum[row];
        for(uint j=lane;j<Block;j+=4)if(row<count && base+j<=pos)high=max(high,scores[row*Block+j]*as_type<float>(p[3]));
        high=max(high,simd_shuffle_xor(high,1));high=max(high,simd_shuffle_xor(high,2));
        float old=isfinite(maximum[row]) ? exp(maximum[row]-high) : 0.0f;
        float sum=0;
        for(uint j=lane;j<Block;j+=4) {
            float prob=row<count && base+j<=pos ? exp(scores[row*Block+j]*as_type<float>(p[3])-high) : 0.0f;
            probability[row*Block+j]=A(prob);sum+=prob;
        }
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
        if(lane==0) {
            maximum[row]=high;correction[row]=old;
            denominator[row]=row<count ? denominator[row]*old+sum : 1.0f;
        }
        if constexpr(!Interleaved) {
            // All readers of K have completed before this storage becomes V.
            for(uint i=tid;i<Block*256;i+=128) {
                uint t=base+i%Block,d=i/Block;
                uint physical=t<=lastpos ? pages[slot*p[2]+t/16]*16+t%16 : 0;
                kv[i]=t<=lastpos ? A(vc[ulong(physical)*kvwidth+kh*256+d]) : A(0);
            }
        }
        threadgroup_barrier(tile_fence);
        auto product=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
        for(uint i=0;i<product.get_capacity();++i)product[i]=0;
        if constexpr(Interleaved) {
            uint physical=pages[slot*p[2]+base/16]*16;
            auto page=tensor(const_cast<device KV*>(vc)+ulong(physical)*kvwidth+kh*256,
                dextents<int,2>{256,int(min(Block,lastpos-base+1))},array<int,2>{1,int(kvwidth)});
            pv.run(tp,page,product);
        } else pv.run(tp,tv,product);
        if constexpr(Indexed) {
            // A narrow, statically unrolled index lets the compiler simplify
            // cooperative-element validity and row ownership once, instead
            // of walking a dynamic iterator on every 16-token KV page. Keep
            // the same per-element expression and causal reduction order.
            #pragma unroll
            for(ushort i=0;i<product.get_capacity();++i) {
                if(product.is_valid_element(i))
                    accum[i]=accum[i]*correction[product.get_multidimensional_index(i)[1]]+product[i];
            }
        } else {
            uint i=0;
            for(auto it=product.begin();it!=product.end();++it,++i) {
                auto ij=it.get_multidimensional_index();
                if(it.is_valid_element())accum[i]=accum[i]*correction[ij[1]]+*it;
            }
        }
        threadgroup_barrier(tile_fence);
    }
    if constexpr(Splits==1) {
        for(auto it=accum.begin();it!=accum.end();++it) {
            auto ij=it.get_multidimensional_index();
            if(it.is_valid_element())*it/=denominator[ij[1]];
        }
        auto dst=tensor(out+ulong(first)*width+head*256,dextents<int,2>(256,count),array<int,2>{1,int(width)});
        accum.store(dst);
    } else {
        // Partial numerators retain their own online-softmax scale. Empty
        // causal partitions publish denominator zero and are skipped by join.
        for(auto it=accum.begin();it!=accum.end();++it) {
            auto ij=it.get_multidimensional_index();
            if(it.is_valid_element() && uint(ij[1])<count)
                out[((ulong(first+ij[1])*p[0]+head)*Splits+part)*258+ij[0]]=*it;
        }
        if(tid<count) {
            ulong dst=((ulong(first+tid)*p[0]+head)*Splits+part)*258;
            out[dst+256]=maximum[tid];out[dst+257]=denominator[tid];
        }
    }
}


kernel void qwen_attention_prefill(device half* q [[buffer(0)]],device const half* kc [[buffer(1)]],
                                   device const half* vc [[buffer(2)]],device const uint* meta [[buffer(3)]],
                                   device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],
                                   device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]],
                                   device half* staging [[buffer(8)]],constant uint* p [[buffer(9)]],
                                   uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    // Borrow dead DeltaNet scratch for each group's K/V operand. BK32 in
    // threadgroup memory exceeds Apple10's limit with shader validation;
    // shrinking BK also changes online-softmax rounding. Device-backed
    // staging keeps the elected matrix shapes
    // and softmax boundaries, without another allocation or a dtype change.
    device half* kv=staging+(ulong(group.y)*p[0]+group.x)*32*256;
    threadgroup half probability[32*32];
    threadgroup float scores[32*32],maximum[32],denominator[32],correction[32];
    qwen_prefill_tile<1,half,32,true>(q,kc,vc,meta,pages,out,limits,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid,
                      kv,probability,scores,maximum,denominator,correction);
}

// Short append batches have too few query tiles to occupy M5 at long KV
// lengths. Split the exact paged KV domain, then combine with log-sum-exp
// rescaling (Flash-Decoding / FlashInfer append attention). Original kernel;
// no KV approximation, token dropping or persistent workspace expansion.
kernel void qwen_attention_prefill_split(device half* q [[buffer(0)]],device const half* kc [[buffer(1)]],
                                   device const half* vc [[buffer(2)]],device const uint* meta [[buffer(3)]],
                                   device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],
                                   device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]],
                                   device half* staging [[buffer(8)]],constant uint* p [[buffer(9)]],
                                   uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    device half* kv=staging+((ulong(group.y)*p[0]+group.x)*4+group.z)*32*256;
    threadgroup half probability[32*32];
    threadgroup float scores[32*32],maximum[32],denominator[32],correction[32];
    qwen_prefill_tile<4,half,32,true>(q,kc,vc,meta,pages,out,limits,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid,
                      kv,probability,scores,maximum,denominator,correction,group.z);
}
// MoE keeps F32 queries and probabilities across prefill/decode boundaries;
// persistent KV remains the elected F16 representation. Dead recurrent
// scratch supplies bounded F32 K/V tiles, not an expanded persistent cache.
#define QWEN_PREFILL_STRICT(NAME,SPLITS) \
kernel void NAME(device float* q [[buffer(0)]],device const half* kc [[buffer(1)]], \
 device const half* vc [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]], \
 device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]], \
 device float* staging [[buffer(8)]],constant uint* p [[buffer(9)]], \
 uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 device float* kv=staging+((ulong(group.y)*p[0]+group.x)*SPLITS+group.z)*32*256; \
 threadgroup float probability[32*32],scores[32*32],maximum[32],denominator[32],correction[32]; \
 qwen_prefill_tile<SPLITS,half,32,true>(q,kc,vc,meta,pages,out,limits,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid, \
 kv,probability,scores,maximum,denominator,correction,group.z); }
QWEN_PREFILL_STRICT(qwen_attention_prefill_strict,1)
QWEN_PREFILL_STRICT(qwen_attention_prefill_split_strict,4)
#undef QWEN_PREFILL_STRICT

kernel void qwen_attention_prefill_join(device const float* parts [[buffer(0)]],device float* out [[buffer(1)]],
                                 device const uint* tiles [[buffer(2)]],constant uint* p [[buffer(3)]],
                                 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    if(group.z>=tiles[2*group.y+1])return;
    uint rh=(tiles[2*group.y]+group.z)*p[0]+group.x;ulong base=ulong(rh)*4*258;
    float high=-INFINITY;for(uint s=0;s<4;++s)high=max(high,parts[base+s*258+256]);
    float acc[8];for(uint j=0;j<8;++j)acc[j]=0;float denom=0;
    for(uint s=0;s<4;++s) {
        ulong src=base+s*258;if(parts[src+257]<=0)continue;
        float c=exp(parts[src+256]-high);denom+=parts[src+257]*c;
        for(uint j=0;j<8;++j)acc[j]+=parts[src+lane+j*32]*c;
    }
    for(uint j=0;j<8;++j)out[ulong(rh)*256+lane+j*32]=acc[j]/denom;
}
