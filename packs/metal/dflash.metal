template<bool PerQueryWindow=false,typename Scalar=half>
inline void df_prefill_tile(device Scalar* q, device const Scalar* kc, device const Scalar* vc,
                         device const uint* meta, device const uint* pages, device float* out,
                         constant uint* p, uint head, uint first, uint count, uint tid,
                         threadgroup Scalar* kv, threadgroup Scalar* probability, threadgroup float* scores,
                         threadgroup float* maximum, threadgroup float* denominator, threadgroup float* correction) {
    uint kh=head/(p[0]/p[1]),slot=meta[2*first];
    uint lastpos=meta[2*(first+count-1)+1],width=p[0]*128,kvwidth=p[1]*128;
    auto tq=tensor(q+ulong(first)*width+head*128,dextents<int,2>(128,count),array<int,2>{1,int(width)});
    auto tk=tensor(kv,extents<int,128,32>(),array<int,2>{1,128});
    auto tv=tensor(kv,extents<int,32,128>(),array<int,2>{1,32});
    auto tp=tensor(probability,extents<int,32,32>(),array<int,2>{1,32});
    auto ts=tensor(scores,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto qk_desc=matmul2d_descriptor(32,32,128,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc=matmul2d_descriptor(32,128,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc,execution_simdgroups<4>> qk;
    matmul2d<pv_desc,execution_simdgroups<4>> pv;
    auto accum=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<accum.get_capacity();++i)accum[i]=0;
    if(tid<32){maximum[tid]=-INFINITY;denominator[tid]=0;}
    uint window_pos=PerQueryWindow?meta[first*2+1]:lastpos;
    uint firstpos=window_pos+1>2048 ? window_pos+1-2048 : 0;
    for(uint base=firstpos;base<=lastpos;base+=32) {
        for(uint i=tid;i<32*128;i+=128) {
            uint t=base+i/128,d=i%128;
            uint physical=t<=lastpos ? pages[slot*p[2]+t/16]*16+t%16 : 0;
            kv[i]=t<=lastpos ? kc[ulong(physical)*kvwidth+kh*128+d] : Scalar(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto score=qk.get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        for(uint i=0;i<score.get_capacity();++i)score[i]=0;
        qk.run(tq,tk,score);
        score.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint row=tid/4,lane=tid%4,pos=lastpos;
        uint rowpos=row<count?meta[2*(first+row)+1]:lastpos;
        uint lower=PerQueryWindow && rowpos+1>2048 ? rowpos+1-2048 : firstpos;
        float high=maximum[row];
        for(uint j=lane;j<32;j+=4)if(row<count && base+j>=lower && base+j<=pos)high=max(high,scores[row*32+j]*as_type<float>(p[3]));
        high=max(high,simd_shuffle_xor(high,1));high=max(high,simd_shuffle_xor(high,2));
        float old=isfinite(maximum[row]) ? exp(maximum[row]-high) : 0.0f;
        float sum=0;
        for(uint j=lane;j<32;j+=4) {
            float prob=row<count && base+j>=lower && base+j<=pos ? exp(scores[row*32+j]*as_type<float>(p[3])-high) : 0.0f;
            probability[row*32+j]=Scalar(prob);sum+=prob;
        }
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
        if(lane==0) {
            maximum[row]=high;correction[row]=old;
            denominator[row]=row<count ? denominator[row]*old+sum : 1.0f;
        }
        // All readers of K have completed before this storage becomes V.
        for(uint i=tid;i<32*128;i+=128) {
            uint t=base+i%32,d=i/32;
            uint physical=t<=lastpos ? pages[slot*p[2]+t/16]*16+t%16 : 0;
            kv[i]=t<=lastpos ? vc[ulong(physical)*kvwidth+kh*128+d] : Scalar(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto product=pv.get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
        for(uint i=0;i<product.get_capacity();++i)product[i]=0;
        pv.run(tp,tv,product);
        uint i=0;
        for(auto it=product.begin();it!=product.end();++it,++i) {
            auto ij=it.get_multidimensional_index();
            if(it.is_valid_element())accum[i]=accum[i]*correction[ij[1]]+*it;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=accum.begin();it!=accum.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element())*it/=denominator[ij[1]];
    }
    auto dst=tensor(out+ulong(first)*width+head*128,dextents<int,2>(128,count),array<int,2>{1,int(width)});
    accum.store(dst);
}


kernel void df_attention(device half* q [[buffer(0)]],device const half* kc [[buffer(1)]],
                                   device const half* vc [[buffer(2)]],device const uint* meta [[buffer(3)]],
                                   device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],
                                   device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],
                                   uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kv[32*128],probability[32*32];
    threadgroup float scores[32*32],maximum[32],denominator[32],correction[32];
    df_prefill_tile(q,kc,vc,meta,pages,out,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid,
                      kv,probability,scores,maximum,denominator,correction);
}

// Muse uses the query's own sliding-window lower edge, while all rows may
// attend to the block's future noise rows. Advancing the lower edge to the
// block end would silently discard up to 15 legitimate conditioning tokens.
kernel void df_attention_muse(device half* q [[buffer(0)]],device const half* kc [[buffer(1)]],
    device const half* vc [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],
    device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kv[32*128],probability[32*32];
    threadgroup float scores[32*32],maximum[32],denominator[32],correction[32];
    df_prefill_tile<true>(q,kc,vc,meta,pages,out,p,group.x,tiles[2*group.y],tiles[2*group.y+1],tid,
        kv,probability,scores,maximum,denominator,correction);
}

// Original Metal implementation of the upstream DFlash2 checkpoint equations.
// Capture residuals entering the GGUF's shifted target-layer indices.
kernel void df_tap(device const float* h [[buffer(0)]],device float* taps [[buffer(1)]],
                   constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1])taps[ulong(i/p[0])*p[0]*5+p[2]*p[0]+i%p[0]]=h[i];
}

// Two taps, 16 channels/group. Reads previous rows only inside this block.
kernel void df_conv(device const float4* x [[buffer(0)]],device const float4* base [[buffer(1)]],
                    device const float* delta [[buffer(2)]],device float4* out [[buffer(3)]],
                    constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    uint w=p[0]/4,row=i/w,d=i%w;if(row>=p[1])return;
    uint ng=p[0]/16,g=d/4,side=p[3];float4 sum=0;
    for(uint t=0;t<min(2u,row%p[2]+1);++t)
        sum+=(base[(side*2+t)*w+d]+delta[ulong(row)*ng*4+(side*2+t)*ng+g])*x[ulong(row-t)*w+d];
    out[i]=sum;
}

// Full split-half rotary, head128, unlike the target's partial head256.
// p: heads, page stride, epsilon bits, theta bits.
kernel void df_qnorm(device const float* x [[buffer(0)]],device const float* norm [[buffer(1)]],
                     device const uint* meta [[buffer(2)]],device float* out [[buffer(3)]],
                     constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong b=(ulong(g.y)*p[0]+g.x)*128;float sum=0;
    for(uint d=lane;d<128;d+=32)sum+=x[b+d]*x[b+d];
    float inv=rsqrt(simd_sum(sum)/128.0f+as_type<float>(p[2]));
    for(uint d=lane;d<128;d+=32) {
        uint other=d^64;float angle=float(meta[g.y*4])*pow(as_type<float>(p[3]),-float(d%64)/64.0f);
        out[b+d]=(x[b+d]*norm[d]*cos(angle)+(d<64 ? -1.0f : 1.0f)*x[b+other]*norm[other]*sin(angle))*inv;
    }
}
kernel void df_kstore(device const float* x [[buffer(0)]],device const float* v [[buffer(1)]],
                      device const float* norm [[buffer(2)]],device const uint* meta [[buffer(3)]],
                      device const uint* pages [[buffer(4)]],device half* keys [[buffer(5)]],device half* values [[buffer(6)]],
                      device const uint* mrope [[buffer(7)]],device const uint* bounds [[buffer(8)]],
                      constant uint* p [[buffer(9)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    // A large image span may exceed the conditioning ring. Only its last
    // ring-sized suffix writes: no concurrent rows alias a physical entry.
    if(p[5]==0 && meta[g.y*2+1]+p[4]<=meta[(bounds[g.y*2+1]-1)*2+1])return;
    ulong b=(ulong(g.y)*p[0]+g.x)*128;float sum=0;
    for(uint d=lane;d<128;d+=32)sum+=x[b+d]*x[b+d];
    float inv=rsqrt(simd_sum(sum)/128.0f+as_type<float>(p[2]));
    uint slot=meta[g.y*2],pos=meta[g.y*2+1],physical=pages[slot*p[1]+pos/16]*16+pos%16;
    ulong dst=(ulong(physical)*p[0]+g.x)*128;
    for(uint d=lane;d<128;d+=32) {
        uint other=d^64;float angle=float(mrope[g.y*4])*pow(as_type<float>(p[3]),-float(d%64)/64.0f);
        keys[dst+d]=half((x[b+d]*norm[d]*cos(angle)+(d<64 ? -1.0f : 1.0f)*x[b+other]*norm[other]*sin(angle))*inv);
        values[dst+d]=half(v[b+d]);
    }
}

// Hierarchical exact top16. Each lane owns sixteen values in a 4096-token
// partition. Repeated block reduction removes only that partition's winner;
// merge reads just 16 candidates/partition, never rescans the full vocabulary.
template<bool Merge>
inline void df_top_impl(device const uint* input,device uint* out,uint width,uint row,uint tile,uint tid,
                        threadgroup float* maxima,threadgroup uint* ids) {
    float v[16];uint ix[16];
    for(uint j=0;j<16;++j) {
        uint col=tile*4096+tid+j*256;bool live=col<width;
        ix[j]=live ? (Merge ? input[(ulong(row)*width+col)*2] : col) : 0xffffffffu;
        v[j]=live ? as_type<float>(input[(ulong(row)*width+col)*(Merge?2:1)+(Merge?1:0)]) : -INFINITY;
    }
    for(uint k=0;k<16;++k) {
        float best=-INFINITY;uint id=0xffffffffu;
        for(uint j=0;j<16;++j)if(v[j]>best || (v[j]==best && ix[j]<id)){best=v[j];id=ix[j];}
        float mx=simd_max(best);uint im=simd_min(best==mx?id:0xffffffffu);
        if(tid%32==0){maxima[tid/32]=mx;ids[tid/32]=im;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0) {
            best=maxima[0];id=ids[0];for(uint j=1;j<8;++j)if(maxima[j]>best || (maxima[j]==best && ids[j]<id)){best=maxima[j];id=ids[j];}
            ulong dst=((ulong(row)*((width+4095)/4096)+tile)*16+k)*2;
            out[dst]=id;out[dst+1]=as_type<uint>(best);ids[8]=id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;++j)if(ix[j]==ids[8]){v[j]=-INFINITY;ix[j]=0xffffffffu;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void df_top16(device const uint* x [[buffer(0)]],device uint* out [[buffer(1)]],constant uint* p [[buffer(2)]],
                     uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float m[8];threadgroup uint ids[9];df_top_impl<false>(x,out,p[0],g.y,g.x,tid,m,ids);
}
kernel void df_top16_merge(device const uint* x [[buffer(0)]],device uint* out [[buffer(1)]],constant uint* p [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float m[8];threadgroup uint ids[9];df_top_impl<true>(x,out,p[0],row,0,tid,m,ids);
}

// One SIMD per candidate. Codebooks are gathered directly from the exact GGUF
// rows: only 16*256 elements/position, no expanded vocab-sized cache.
template<bool Prefix>
inline void df_select_impl(device const uint* top,device const uchar* pred,
                      device const uchar* succ,device const float* h,
                      device const uint* tokens,device uint* out,constant uint* p,
                      uint block,uint tid,threadgroup float* scores,threadgroup uint& chosen) {
    uint lane=tid%32,sg=tid/32;
    if(tid==0){chosen=tokens[block*p[0]];out[block*p[0]]=chosen;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // p[0] is storage stride; p[3] is the requested prefix including its
    // seed. The autoregressive selector never reads later candidate rows.
    for(uint j=1;j<(Prefix?p[3]:p[0]);++j) {
        uint row=block*p[0]+j;
        for(uint c=sg;c<16;c+=8) {
            uint token=top[(row*16+c)*2];float sum=0;
            for(uint d=lane*4;d<256;d+=128) {
                float4 a=p[1]==30 ? float4(*reinterpret_cast<device const bfloat4*>(pred+(ulong(chosen)*256+d)*2)) : kquant4(pred,p[1],ulong(chosen)*256+d);
                float4 b=p[2]==30 ? float4(*reinterpret_cast<device const bfloat4*>(succ+(ulong(token)*256+d)*2)) : kquant4(succ,p[2],ulong(token)*256+d);
                sum+=dot(a*b,*reinterpret_cast<device const float4*>(h+ulong(row)*256+d));
            }
            sum=simd_sum(sum);if(lane==0)scores[c]=sum+as_type<float>(top[(row*16+c)*2+1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0){uint best=0;for(uint c=1;c<16;++c)if(scores[c]>scores[best])best=c;chosen=top[(row*16+best)*2];out[row]=chosen;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#define DF_SELECT(NAME,PREFIX) \
kernel void NAME(device const uint* top [[buffer(0)]],device const uchar* pred [[buffer(1)]], \
device const uchar* succ [[buffer(2)]],device const float* h [[buffer(3)]], \
device const uint* tokens [[buffer(4)]],device uint* out [[buffer(5)]],constant uint* p [[buffer(6)]], \
uint block [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float scores[16];threadgroup uint chosen; \
df_select_impl<PREFIX>(top,pred,succ,h,tokens,out,p,block,tid,scores,chosen); }
DF_SELECT(df_select,false)
DF_SELECT(df_select_prefix,true)
#undef DF_SELECT
