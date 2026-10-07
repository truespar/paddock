// DiffusionGemma primitives. Packed weights remain resident in their source
// format. Only bounded contraction tiles are unpacked; no full FP16 copy.
// Keep derived scalar rounding explicit. The Metal compiler can fold nested
// BF16(sqrt(float(uint))) incorrectly on Apple10 (53 becomes 3.48e11).
// Integer round-to-nearest-even preserves the scalar's IEEE representation.
inline float dg_scalar_bf(float x) {
    uint b=as_type<uint>(x);
    return as_type<float>((b+0x7fff+((b>>16)&1))&0xffff0000);
}
// MLX's HD256/512 graph rounds QK and normalized probabilities to BF16.
// Two bounded passes retain that contract without an O(context * canvas)
// score allocation. Canvas rows share the same retained encoder window.
template<uint HD,typename KV,bool Bidir=false,typename T=float,uint BM=16,uint KT=16,uint Panel=128>
inline void dg_attention(device T* q,device const KV* k,device const KV* v,
    device const uint* meta,device const uint* pages,device float* out,device const uint* tiles,
    device const uint* limits,constant uint* p,uint2 g,uint tid,
    threadgroup T* kv,threadgroup T* prob,threadgroup float* scores,
    threadgroup float* highs,threadgroup float* sums) {
    constexpr uint G=128/BM;
    uint first=tiles[g.y*2],count=tiles[g.y*2+1],slot=meta[first*2],head=g.x,kh=head/(p[0]/p[1]),width=p[0]*HD;
    uint last=limits[first+count-1],firstpos=meta[first*2+1],low=p[3] && firstpos+1>p[3]?firstpos+1-p[3]:0;
    if(p[5]) low=p[3] && p[7+slot]>p[3]-1?p[7+slot]-(p[3]-1):0;
    if constexpr(Bidir) {last=p[3]?min(last,meta[(first+count-1)*2+1]+p[3]):last;low=p[3] && firstpos>p[3]?firstpos-p[3]:0;}
    auto tq=tensor(q+ulong(first)*width+head*HD,dextents<int,2>(HD,count),array<int,2>{1,int(width)});
    auto tk=tensor(kv,extents<int,Panel,KT>(),array<int,2>{1,Panel});
    auto tv=tensor(kv,extents<int,KT,Panel>(),array<int,2>{1,KT});
    auto tp=tensor(prob,extents<int,KT,BM>(),array<int,2>{1,KT});
    auto ts=tensor(scores,extents<int,KT,BM>(),array<int,2>{1,KT});
    constexpr auto qkd=matmul2d_descriptor(BM,KT,Panel,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pvd=matmul2d_descriptor(BM,Panel,KT,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroups<4>> qk;matmul2d<pvd,execution_simdgroups<4>> pv;
    auto a0=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    auto a1=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    auto a2=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    auto a3=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv),float>();
    for(uint i=0;i<a0.get_capacity();++i){a0[i]=0;a1[i]=0;a2[i]=0;a3[i]=0;}
    if(tid<BM){highs[tid]=-INFINITY;sums[tid]=0;}
    for(uint pass=0;pass<2;++pass)for(uint base=low;base<=last;base+=KT){
        auto acc=qk.template get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
        for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
        for(uint panel=0;panel<HD/Panel;++panel){
            for(uint i=tid;i<KT*Panel;i+=128){uint t=base+i/Panel,d=panel*Panel+i%Panel;kv[i]=t<=last?T(k[(ulong(Bidir?slot+t:gemma_physical(pages,slot,t,p))*p[1]+kh)*HD+d]):T(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);auto query=tq.slice(panel*Panel,0);qk.run(query,tk,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if(p[6])for(uint i=0;i<acc.get_capacity();++i)acc[i]=mlx_bf(acc[i]);acc.store(ts);threadgroup_barrier(mem_flags::mem_threadgroup);
        uint row=tid/G,lane=tid%G,pos=row<count?meta[(first+row)*2+1]:0;
        uint floor=p[5]?low:(p[3] && pos+1>p[3]?pos+1-p[3]:0),upper=row<count?limits[first+row]:0;
        if constexpr(Bidir){floor=p[3] && pos>p[3]?pos-p[3]:0;upper=p[3]?min(upper,pos+p[3]):upper;}
        if(pass==0){float high=highs[row];for(uint j=lane;j<KT;j+=G)if(row<count && base+j>=floor && base+j<=upper)high=max(high,scores[row*KT+j]);
            for(uint s=1;s<G;s*=2)high=max(high,simd_shuffle_xor(high,s));float sum=0;
            for(uint j=lane;j<KT;j+=G)if(row<count && base+j>=floor && base+j<=upper)sum+=exp(scores[row*KT+j]-high);
            for(uint s=1;s<G;s*=2)sum+=simd_shuffle_xor(sum,s);
            if(lane==0){sums[row]=sums[row]*(isfinite(highs[row])?exp(highs[row]-high):0)+sum;highs[row]=high;}
        }else{
            for(uint j=lane;j<KT;j+=G){float value=row<count && base+j>=floor && base+j<=upper?exp(scores[row*KT+j]-highs[row])/sums[row]:0;prob[row*KT+j]=T(p[6]?mlx_bf(value):value);}
            for(uint panel=0;panel<HD/Panel;++panel){
                for(uint i=tid;i<KT*Panel;i+=128){uint t=base+i%KT,d=panel*Panel+i/KT;kv[i]=t<=last?T(v[(ulong(Bidir?slot+t:gemma_physical(pages,slot,t,p))*p[1]+kh)*HD+d]):T(0);}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(panel==0)pv.run(tp,tv,a0);else if(panel==1)pv.run(tp,tv,a1);else if(panel==2)pv.run(tp,tv,a2);else pv.run(tp,tv,a3);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint panel=0;panel<HD/Panel;++panel){auto dst=tensor(out+ulong(first)*width+head*HD+panel*Panel,dextents<int,2>(Panel,count),array<int,2>{1,int(width)});
        if(panel==0)a0.store(dst);else if(panel==1)a1.store(dst);else if(panel==2)a2.store(dst);else a3.store(dst);}
}
#define DG_ATTENTION(NAME,HD,KV) \
kernel void NAME(device float* q [[buffer(0)]],device const KV* k [[buffer(1)]],device const KV* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float kv[16*128],prob[16*16],scores[16*16],highs[16],sums[16];dg_attention<HD,KV>(q,k,v,meta,pages,out,tiles,limits,p,g,tid,kv,prob,scores,highs,sums);}
DG_ATTENTION(dg_attention256,256,bfloat)
DG_ATTENTION(dg_attention512,512,bfloat)
DG_ATTENTION(dg_gguf_attention256,256,half)
DG_ATTENTION(dg_gguf_attention512,512,half)
#undef DG_ATTENTION
inline float dg_weight(device const uchar* w,uint k,uint n,uint ty,ulong i) {
    if(ty==0x100 || ty==0x108) {
        ulong total=ulong(k)*n,bytes=ty==0x100?total/2:total;
        uint code=ty==0x100?(w[i/2]>>((i%2)*4))&15:w[i];
        device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+bytes);
        device const bfloat* biases=scales+total/64;
        return float(code)*float(scales[i/64])+float(biases[i/64]);
    }
    return weight(w,ty,i);
}
// Decode adjacent values together. Calling weight() four times on a K-quant
// superblock decodes the same four values four times. Types are elected on
// the host, so the compiler removes unrelated format branches as well.
template<uint TY>
inline float4 dg_weight4(device const uchar* w,uint k,uint n,ulong i) {
    if constexpr(TY==12 || TY==14) return kquant4(w,TY,i);
    else return float4(dg_weight(w,k,n,TY,i),dg_weight(w,k,n,TY,i+1),
                       dg_weight(w,k,n,TY,i+2),dg_weight(w,k,n,TY,i+3));
}
template<uint TY>
inline void dg_project_packed(device const uchar* w,device const float* x,
    device float* out,constant uint* p,uint2 g,uint tid,
    threadgroup float* a,threadgroup float* b) {
    auto ta=tensor(a,extents<int,64,16>(),array<int,2>{1,64});
    auto tb=tensor(b,extents<int,64,32>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64) {
        for(uint i=tid*4;i<16*64;i+=128*4){uint r=g.y*16+i/64,k=base+i%64;
            float4 z=r<p[2] && k<p[0]?*reinterpret_cast<device const packed_float4*>(x+ulong(r)*p[0]+k):float4(0);
            *reinterpret_cast<threadgroup packed_float4*>(a+i)=z;}
        for(uint i=tid*4;i<32*64;i+=128*4){uint n=g.x*32+i/64,k=base+i%64;
            float4 z=n<p[1] && k<p[0]?dg_weight4<TY>(w,p[0],p[1],ulong(n)*p[0]+k):float4(0);
            if constexpr(TY==0x100 || TY==0x108)z=float4(bfloat4(z));
            *reinterpret_cast<threadgroup packed_float4*>(b+i)=z;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(ta,tb,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();
        uint n=g.x*32+ij[0],r=g.y*16+ij[1];
        if(it.is_valid_element() && r<p[2] && n<p[1])out[ulong(r)*p[1]+n]=(TY==0x100 || TY==0x108)?mlx_bf(*it):*it;}
}
#define DG_PROJECT_PACKED(NAME,TY) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float a[16*64],b[32*64];dg_project_packed<TY>(w,x,out,p,g,tid,a,b);}
DG_PROJECT_PACKED(dg_project_q4,12)
DG_PROJECT_PACKED(dg_project_q5,6)
DG_PROJECT_PACKED(dg_project_q6,14)
DG_PROJECT_PACKED(dg_project_q8,8)
DG_PROJECT_PACKED(dg_project_f32,0)
DG_PROJECT_PACKED(dg_project_a4,0x100)
DG_PROJECT_PACKED(dg_project_a8,0x108)
#undef DG_PROJECT_PACKED
kernel void dg_moe_head(device const float* x [[buffer(0)]],device const float* gamma [[buffer(1)]],
    device const float* pre [[buffer(2)]],device float* router [[buffer(3)]],device float* expert [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    threadgroup float sums[32];uint n=p[0],threads=min(1024u,((n+127)/128)*32);ulong b=ulong(row)*n;
    float inv=gmlx_inv(x+b,n,as_type<float>(p[1]),tid,threads,sums);
    for(uint i=tid;i<n;i+=threads){float z=mlx_bf(x[b+i]*inv);expert[b+i]=mlx_bf(z*pre[i]);
        router[b+i]=mlx_bf(mlx_bf(z*gamma[i])*dg_scalar_bf(1/sqrt(float(n))));}
}
kernel void dg_moe_route(device const float* logits [[buffer(0)]],device const float* scale [[buffer(1)]],
    device uint* ids [[buffer(2)]],device float* weights [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float score[4];for(uint j=0;j<4;++j)score[j]=logits[row*128+lane+j*32];float selected[8];uint chosen[8];
    for(uint pick=0;pick<8;++pick){float best=-INFINITY;for(uint j=0;j<4;++j)best=max(best,score[j]);best=simd_max(best);
        uint id=0;for(uint j=0;j<4;++j)if(score[j]==best)id=max(id,lane+j*32);id=simd_max(id);
        selected[pick]=best;chosen[pick]=id;for(uint j=0;j<4;++j)if(lane+j*32==id)score[j]=-INFINITY;}
    float sum=0,maximum=selected[0];for(uint j=0;j<8;++j){selected[j]=exp(selected[j]-maximum);sum+=selected[j];}
    if(lane==0)for(uint j=0;j<8;++j){ids[row*8+j]=chosen[j];weights[row*8+j]=mlx_bf(mlx_bf(selected[j]/sum)*scale[chosen[j]]);}
}
kernel void dg_moe_geglu(device float* gu [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if(i>=p[0]*p[1])return;ulong o=ulong(i/p[0])*p[0]*2+i%p[0];float x=gu[o];
    float cube=mlx_bf(pow(x,3.0f)),z=mlx_bf(x+mlx_bf(mlx_bf(0.044715f)*cube));
    z=mlx_bf(mlx_bf(0.7978845608028654f)*z);z=mlx_bf(1.0f+mlx_bf(precise::tanh(z)));
    gu[o]=mlx_bf(mlx_bf(mlx_bf(0.5f*x)*z)*gu[o+p[0]]);
}
kernel void dg_moe_fold(device const float* out [[buffer(0)]],device const float* weights [[buffer(1)]],
    device float* routed [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;uint row=i/p[0],n=i%p[0];float sum=0;
    for(uint j=0;j<8;++j)sum+=mlx_bf(out[(ulong(row)*8+j)*p[0]+n]*weights[row*8+j]);routed[i]=mlx_bf(sum);
}
kernel void dg_moe_branches(device float* shared [[buffer(0)]],device const float* routed [[buffer(1)]],
    device const float* spost [[buffer(2)]],device const float* rpost [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    threadgroup float sums[32];uint n=p[0],threads=min(1024u,((n+127)/128)*32);ulong b=ulong(row)*n;
    float a=gmlx_inv(shared+b,n,as_type<float>(p[1]),tid,threads,sums);
    float c=gmlx_inv(routed+b,n,as_type<float>(p[1]),tid,threads,sums);
    for(uint i=tid;i<n;i+=threads)shared[b+i]=mlx_bf(mlx_bf(mlx_bf(shared[b+i]*a)*spost[i])+mlx_bf(mlx_bf(routed[b+i]*c)*rpost[i]));
}
// Packed affine8/GGUF projection with fixed F32 contractions. p: K,N,M,type,
// BF16 operation boundaries. This also handles the flat Q5_0 2112-wide tail.
kernel void dg_affine_vector(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint n=g.x*4+sg;if(n>=p[1])return;
    bool q4=p[3]==0x100;uint pack=q4?8:4,items=p[0]%(pack*64)==0?pack*2:pack;
    ulong total=ulong(p[0])*p[1];device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+(q4?total/2:total));
    device const bfloat* biases=scales+total/64;
    float sum=0;
    for(uint k=lane*items;k<p[0];k+=32*items){float dot=0,xsum=0;
        for(uint j=0;j<items && k+j<p[0];j+=q4?4:1){float part=0,bias_sum=0;
            for(uint vj=0;vj<(q4?4u:1u) && k+j+vj<p[0];++vj){ulong at=ulong(n)*p[0]+k+j+vj;float v=x[ulong(g.y)*p[0]+k+j+vj];
                uint code=q4?(w[at/2]>>((at%2)*4))&15:w[at];part+=v*float(code);bias_sum=q4?mlx_bf(bias_sum+v):bias_sum+v;}
            dot+=part;xsum+=bias_sum;}
        ulong group=(ulong(n)*p[0]+k)/64;sum+=dot*float(scales[group])+xsum*float(biases[group]);}
    sum=simd_sum(sum);if(lane==0)out[ulong(g.y)*p[1]+n]=mlx_bf(sum);
}
kernel void dg_expert_vector(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint n=g.x*4+sg;if(n>=p[1])return;
    bool q4=p[3]==0x100;uint pack=q4?8:4,items=p[0]%(pack*64)==0?pack*2:pack;
    ulong total=ulong(p[0])*p[1]*128;device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+(q4?total/2:total));
    device const bfloat* biases=scales+total/64;
    ulong wr=(ulong(ids[g.y])*p[1]+n)*p[0],xr=p[4]?ulong(g.y)*p[0]*2:ulong(g.y/8)*p[0];float sum=0;
    for(uint k=lane*items;k<p[0];k+=32*items){float dot=0,xsum=0;
        for(uint j=0;j<items && k+j<p[0];j+=q4?4:1){float part=0,bias_sum=0;
            for(uint vj=0;vj<(q4?4u:1u) && k+j+vj<p[0];++vj){ulong at=wr+k+j+vj;float z=x[xr+k+j+vj];uint c=q4?(w[at/2]>>((at%2)*4))&15:w[at];part+=z*float(c);bias_sum=q4?mlx_bf(bias_sum+z):bias_sum+z;}
            dot+=part;xsum+=bias_sum;}
        ulong group=(wr+k)/64;sum+=dot*float(scales[group])+xsum*float(biases[group]);}
    sum=simd_sum(sum);if(lane==0)out[ulong(g.y)*p[1]+n]=mlx_bf(sum);
}
kernel void dg_project(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[16*64],b[32*64];
    auto ta=tensor(a,extents<int,64,16>(),array<int,2>{1,64});
    auto tb=tensor(b,extents<int,64,32>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64) {
        for(uint i=tid;i<16*64;i+=128){uint r=g.y*16+i/64,k=base+i%64;
            a[i]=r<p[2] && k<p[0]?x[ulong(r)*p[0]+k]:0;}
        for(uint i=tid;i<32*64;i+=128){uint n=g.x*32+i/64,k=base+i%64;
            float z=n<p[1] && k<p[0]?dg_weight(w,p[0],p[1],p[3],ulong(n)*p[0]+k):0;
            b[i]=p[4]?mlx_bf(z):z;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(ta,tb,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();
        uint n=g.x*32+ij[0],r=g.y*16+ij[1];
        if(it.is_valid_element() && r<p[2] && n<p[1])out[ulong(r)*p[1]+n]=p[4]?mlx_bf(*it):*it;}
}
kernel void dg_embed(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;
    float x=dg_weight(w,p[0],p[2],p[3],ulong(ids[i/p[0]])*p[0]+i%p[0]);
    float scale=sqrt(float(p[0]));out[i]=p[4]?mlx_bf(mlx_bf(x)*dg_scalar_bf(scale)):x*scale;
}
// probs [rows,vocab] @ packed E [vocab,width]. No resident E-transpose.
// Splits bound command latency and provide enough parallelism for narrow reads.
template<uint TY>
inline void dg_soft_embed_packed(device const uchar* w,device const float* probs,
    device float* parts,constant uint* p,uint3 g,uint tid,
    threadgroup bfloat* a,threadgroup bfloat* b) {
    auto ta=tensor(a,extents<int,64,16>(),array<int,2>{1,64});
    auto tb=tensor(b,extents<int,64,32>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint span=(p[1]+p[4]-1)/p[4],end=min(p[1],(g.z+1)*span);
    for(uint base=g.z*span;base<end;base+=64) {
        for(uint i=tid;i<16*64;i+=128){uint r=g.y*16+i/64,v=base+i%64;
            a[i]=bfloat(r<p[2] && v<end?probs[ulong(r)*p[1]+v]:0);}
        if constexpr(TY==0xffff){
            for(uint i=tid;i<32*64;i+=128){uint n=g.x*32+i/64,v=base+i%64;
                b[i]=bfloat(n<p[0] && v<end?dg_weight(w,p[0],p[1],p[3],ulong(v)*p[0]+n):0);}
        }else{
            // Adjacent embedding columns share one packed block. Scatter its
            // four values into the transposed tile, not a resident transpose.
            for(uint i=tid;i<8*64;i+=128){uint n=g.x*32+(i/64)*4,v=base+i%64;
                float4 z=n<p[0] && v<end?dg_weight4<TY>(w,p[0],p[1],ulong(v)*p[0]+n):float4(0);
                for(uint j=0;j<4;++j)b[(i/64*4+j)*64+i%64]=bfloat(z[j]);}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(ta,tb,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();uint n=g.x*32+ij[0],r=g.y*16+ij[1];
        if(it.is_valid_element() && r<p[2] && n<p[0])parts[(ulong(g.z)*p[2]+r)*p[0]+n]=*it;}
}
#define DG_SOFT_PACKED(NAME,TY) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* probs [[buffer(1)]],device float* parts [[buffer(2)]],constant uint* p [[buffer(3)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup bfloat a[16*64],b[32*64];dg_soft_embed_packed<TY>(w,probs,parts,p,g,tid,a,b);}
DG_SOFT_PACKED(dg_soft_embed,0xffff)
DG_SOFT_PACKED(dg_soft_embed_q6,14)
DG_SOFT_PACKED(dg_soft_embed_q8,8)
DG_SOFT_PACKED(dg_soft_embed_a8,0x108)
#undef DG_SOFT_PACKED
kernel void dg_soft_fold(device const float* parts [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;float x=0;
    for(uint s=0;s<p[2];++s)x+=parts[ulong(s)*p[0]*p[1]+i];
    out[i]=p[3]?mlx_bf(mlx_bf(x)*dg_scalar_bf(sqrt(float(p[0])))):x*sqrt(float(p[0]));
}
kernel void dg_add_norm(device float* x [[buffer(0)]],device const float* delta [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[32];uint n=p[0],threads=p[3]?min(1024u,((n+127)/128)*32):256;ulong b=ulong(row+p[2])*n;
    for(uint d=tid;d<n;d+=threads){float z=x[b+d]+delta[ulong(row)*n+d];x[b+d]=p[3]?mlx_bf(z):z;}
    threadgroup_barrier(mem_flags::mem_device);
    float inv=gmlx_inv(x+b,n,as_type<float>(p[1]),tid,threads,sums);
    for(uint d=tid;d<n;d+=threads){float z=x[b+d]*inv;x[b+d]=p[3]?mlx_bf(z):z;}
}
kernel void dg_zero(device float* out [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])out[i]=0;
}
// One expert-major tile contracts 16 routed tokens, never per-token dispatch.
// p: K,N,rows,type,down,BF16; fused gate/up rows remain in file order.
template<uint TY>
inline void dg_experts_packed(device const uchar* w,device const float* x,
    device const uint* lists,device const uint* counts,
    device const uint* tiles,device float* out,constant uint* p,
    uint2 g,uint tid,threadgroup float* a,threadgroup float* b) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(16u,counts[expert]-first);
    uint stride=p[4]?p[1]:p[1]*2,ntiles=(p[1]+31)/32,plane=p[4]?0:g.x/ntiles,n=(g.x%ntiles)*32;
    auto ta=tensor(a,extents<int,64,16>(),array<int,2>{1,64});
    auto tb=tensor(b,extents<int,64,32>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64){
        for(uint i=tid;i<16*64;i+=128){uint r=i/64,k=base+i%64,entry=r<count?lists[expert*p[2]*8+first+r]:0;
            a[i]=r<count && k<p[0]?x[p[4]?ulong(entry)*p[0]*2+k:ulong(entry/8)*p[0]+k]:0;}
        if constexpr(TY==0xffff){
            for(uint i=tid;i<32*64;i+=128){uint col=n+i/64,k=base+i%64;
                float z=col<p[1] && k<p[0]?dg_weight(w,p[0],128*stride,p[3],(ulong(expert)*stride+plane*p[1]+col)*p[0]+k):0;
                b[i]=p[5]?mlx_bf(z):z;}
        }else{
            for(uint i=tid*4;i<32*64;i+=128*4){uint col=n+i/64,k=base+i%64;
                float4 z=col<p[1] && k<p[0]?dg_weight4<TY>(w,p[0],128*stride,(ulong(expert)*stride+plane*p[1]+col)*p[0]+k):float4(0);
                if constexpr(TY==0x100 || TY==0x108)z=float4(bfloat4(z));
                *reinterpret_cast<threadgroup packed_float4*>(b+i)=z;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(ta,tb,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && ij[1]<count && n+ij[0]<p[1]){
            uint entry=lists[expert*p[2]*8+first+ij[1]];
            out[ulong(entry)*stride+plane*p[1]+n+ij[0]]=p[5]?mlx_bf(*it):*it;}}
}
#define DG_EXPERTS_PACKED(NAME,TY) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float a[16*64],b[32*64];dg_experts_packed<TY>(w,x,lists,counts,tiles,out,p,g,tid,a,b);}
DG_EXPERTS_PACKED(dg_experts,0xffff)
DG_EXPERTS_PACKED(dg_experts_q4,12)
DG_EXPERTS_PACKED(dg_experts_q5,6)
DG_EXPERTS_PACKED(dg_experts_q8,8)
DG_EXPERTS_PACKED(dg_experts_a4,0x100)
DG_EXPERTS_PACKED(dg_experts_a8,0x108)
#undef DG_EXPERTS_PACKED
inline uint dg_random(uint lo,uint hi,uint offset,uint index) {
    uint4 c=uint4(offset,0,index,0);
    for(uint r=0;r<10;++r){ulong a=ulong(c.x)*0xD2511F53u,b=ulong(c.z)*0xCD9E8D57u;
        c=uint4(uint(b>>32)^c.y^lo,uint(b),uint(a>>32)^c.w^hi,uint(a));lo+=0x9E3779B9u;hi+=0xBB67AE85u;}
    return c.x;
}
// p: vocab, inverse temperature bits (negative=greedy), seed lo/hi, offset.
kernel void dg_sample(device float* logits [[buffer(0)]],device uint* picks [[buffer(1)]],
    device uint* draws [[buffer(2)]],device float* entropy [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float high[256],gumbel[256],sum[256],esum[256];threadgroup uint ids[256],sample[256];
    float inv=abs(as_type<float>(p[1])),best=-INFINITY,draw=-INFINITY;uint bi=0,si=0;ulong base=ulong(row)*p[0];
    for(uint i=tid;i<p[0];i+=256){float z=logits[base+i];if(z>best){best=z;bi=i;}
        if(as_type<float>(p[1])>0){float u=clamp((float(dg_random(p[2],p[3],p[4],uint(base+i)))+0.5f)*0x1p-32f,0x1p-33f,0x1.fffffep-1f);
            float v=z*inv-log(-log(u));if(v>draw){draw=v;si=i;}}}
    high[tid]=best;ids[tid]=bi;gumbel[tid]=draw;sample[tid]=si;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=128;s;s>>=1){if(tid<s){uint j=tid+s;
        if(high[j]>high[tid] || (high[j]==high[tid] && ids[j]<ids[tid])){high[tid]=high[j];ids[tid]=ids[j];}
        if(gumbel[j]>gumbel[tid] || (gumbel[j]==gumbel[tid] && sample[j]<sample[tid])){gumbel[tid]=gumbel[j];sample[tid]=sample[j];}}
        threadgroup_barrier(mem_flags::mem_threadgroup);}
    float a=0,b=0;for(uint i=tid;i<p[0];i+=256){float z=(logits[base+i]-high[0])*inv,e=exp(z);a+=e;b+=z*e;}
    sum[tid]=a;esum[tid]=b;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=128;s;s>>=1){if(tid<s){sum[tid]+=sum[tid+s];esum[tid]+=esum[tid+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(uint i=tid;i<p[0];i+=256)logits[base+i]=exp((logits[base+i]-high[0])*inv)/sum[0];
    if(tid==0){picks[row]=ids[0];draws[row]=as_type<float>(p[1])>0?sample[0]:ids[0];entropy[row]=max(0.0f,log(sum[0])-esum[0]/sum[0]);}
}
// Bounded 256-position acceptance. Entropies are sorted with deterministic
// ties; selections are recomputed each step (not accumulated).
kernel void dg_accept(device const float* entropy [[buffer(0)]],device const uint* picks [[buffer(1)]],
    device const uint* draws [[buffer(2)]],device uint* ids [[buffer(3)]],device uint* previous [[buffer(4)]],
    device uint* status [[buffer(5)]],constant uint* p [[buffer(6)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float e[256];threadgroup uint index[256];
    e[tid]=tid<p[0]?entropy[tid]:INFINITY;index[tid]=tid;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint k=2;k<=256;k*=2)for(uint j=k/2;j;j/=2){uint other=tid^j;
        if(other>tid){bool greater=e[tid]>e[other] || (e[tid]==e[other] && index[tid]>index[other]);
            if(greater==((tid&k)==0)){float x=e[tid];e[tid]=e[other];e[other]=x;uint n=index[tid];index[tid]=index[other];index[other]=n;}}
        threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(tid==0){float cumulative=0,mean=0;uint accepted=0;bool stable=p[5]>0;
        for(uint i=0;i<p[0];++i){uint pos=index[i];bool keep=cumulative<=0.1f;cumulative+=e[i];accepted+=uint(keep);
            ids[pos]=keep?draws[pos]:uint((ulong(dg_random(p[2],p[3],p[4],pos))*p[1])>>32);
            stable=stable && previous[pos]==picks[pos];previous[pos]=picks[pos];mean+=entropy[pos];}
        mean/=float(p[0]);status[0]=uint(stable && mean<0.005f);status[1]=accepted;status[2]=as_type<uint>(mean);status[3]=uint(stable);}
}
kernel void dg_labels(device const float* probs [[buffer(0)]],device const uint* labels [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[2])out[i]=probs[ulong(i/p[2])*p[1]+labels[i%p[2]]];
}
