// Original EmbeddingGemma 2 graph operations. The bidirectional, packed
// online-softmax specialization retains O(tokens) storage, not a square mask.
// Tensor-accelerated BF16 contraction over bounded affine tiles. Source codes
// remain packed; no resident dense expansion. One full F32 accumulator per
// output keeps the result independent of other requests in the ragged batch.
template<uint TY,typename T=bfloat,bool BfOut=true,uint BM=32,uint BN=64>
inline void eg2_project(device const uchar* w,device const float* x,device float* y,
    constant uint* p,uint2 g,uint tid,threadgroup T* a,threadgroup T* b) {
    auto ta=tensor(a,extents<int,64,BM>(),array<int,2>{1,64});
    auto tb=tensor(b,extents<int,64,BN>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(BM,BN,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64){
        for(uint i=tid*4;i<BM*64;i+=512){uint r=g.y*BM+i/64,k=base+i%64;
            float4 v=r<p[2]?*reinterpret_cast<device const packed_float4*>(x+ulong(r)*p[0]+k):float4(0);
            *reinterpret_cast<threadgroup vec<T,4>*>(a+i)=vec<T,4>(v);}
        for(uint i=tid*4;i<BN*64;i+=512){uint n=g.x*BN+i/64,k=base+i%64;
            float4 v=n<p[1]?dg_weight4<TY>(w,p[0],p[1],ulong(n)*p[0]+k):float4(0);
            *reinterpret_cast<threadgroup vec<T,4>*>(b+i)=vec<T,4>(v);}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(ta,tb,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ix=it.get_multidimensional_index();uint n=g.x*BN+ix[0],r=g.y*BM+ix[1];
        if(it.is_valid_element() && n<p[1] && r<p[2])y[ulong(r)*p[1]+n]=BfOut?mlx_bf(*it):*it;}
}
#define EG2_PROJECT(NAME,TY,T,BF) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup T a[32*64],b[64*64];eg2_project<TY,T,BF>(w,x,out,p,g,tid,a,b);}
EG2_PROJECT(eg2_project_a8,0x108,bfloat,true)
EG2_PROJECT(eg2_project_a8_f32,0x108,float,false)
EG2_PROJECT(eg2_project_a4,0x100,bfloat,true)
EG2_PROJECT(eg2_project_bf16,30,bfloat,true)
EG2_PROJECT(eg2_project_q8,8,half,false)
EG2_PROJECT(eg2_project_q4,12,half,false)
EG2_PROJECT(eg2_project_q6,14,half,false)
EG2_PROJECT(eg2_project_gguf_bf16,30,half,false)
#undef EG2_PROJECT
// Short requests need more independent output tiles, not batch-dependent
// split-K rounding. Contraction tiles and the full accumulator stay identical.
#define EG2_NARROW(NAME,TY,T,BF) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup T a[16*64],b[16*64];eg2_project<TY,T,BF,16,16>(w,x,out,p,g,tid,a,b);}
EG2_NARROW(eg2_project_a8_narrow,0x108,bfloat,true)
EG2_NARROW(eg2_project_q8_narrow,8,half,false)
#undef EG2_NARROW

// BF16-rounded partition sums with an eight-lane logical fold. A 16-part
// contraction combines (0+8), (1+9), ... before adding the eight totals.
// SIMD matrix contractions retain the reference's eight-product order; a
// wider tensor contraction differs at BF16 ties. Storage stays tile-bounded.
kernel void eg2_project_a8_partition(device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[16*64],b[16*64];
    uint sg=tid/32,lane=tid%32,partitions=p[0]/p[3],fold=max(1u,partitions/8);
    float2 total=0,subtotal=0;
    for(uint ordinal=0;ordinal<partitions;++ordinal){
        uint part=ordinal%fold*8+ordinal/fold;
        auto acc=make_filled_simdgroup_matrix<float,8>(0.0f);
        for(uint base=part*p[3];base<(part+1)*p[3];base+=64){
            for(uint i=tid;i<16*64;i+=128){uint r=g.y*16+i/64,n=g.x*16+i/64,k=base+i%64;
                a[i]=r<p[2]?x[ulong(r)*p[0]+k]:0;
                b[i]=n<p[1]?mlx_bf(dg_weight(w,p[0],p[1],0x108,ulong(n)*p[0]+k)):0;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint k=0;k<64;k+=8){simdgroup_float8x8 left,right;
                simdgroup_load(left,a+(sg/2)*8*64+k,64);
                simdgroup_load(right,b+(sg%2)*8*64+k,64,ulong2(0),true);
                simdgroup_multiply_accumulate(acc,left,right,acc);}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint j=0;j<2;++j)subtotal[j]=mlx_bf(subtotal[j]+mlx_bf(acc.thread_elements()[j]));
        if((ordinal+1)%fold==0){for(uint j=0;j<2;++j)total[j]=mlx_bf(total[j]+subtotal[j]);subtotal=0;}
    }
    uint r=g.y*16+(sg/2)*8+((lane&16)>>2)+((lane&6)>>1);
    uint n=g.x*16+(sg%2)*8+((lane&8)>>1)+((lane&1)<<1);
    if(r<p[2])for(uint j=0;j<2;++j)if(n+j<p[1])out[ulong(r)*p[1]+n+j]=total[j];
}

// Short-request affine contraction keeps decoded weights F32. Eight lanes
// own disjoint 64-weight groups of a channel, reducing eight-element partials
// before a fixed lane tree. Studied MLX 0.32.3's arithmetic, not its code.
kernel void eg2_affine_vector(device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],device float* y [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint lane=tid%8,n=g.x*16+tid/8,kdim=p[0],ndim=p[1];
    if(n>=ndim)return;
    device const bfloat* scale=reinterpret_cast<device const bfloat*>(w+ulong(kdim)*ndim);
    device const bfloat* bias=scale+ulong(kdim)*ndim/64;
    float result=0;
    for(uint base=lane*64;base<kdim;base+=512){
        ulong group=(ulong(n)*kdim+base)/64;
        float s=float(scale[group]),b=float(bias[group]);
        #pragma unroll
        for(uint chunk=0;chunk<64;chunk+=8){
            float dot=0;
            #pragma unroll
            for(uint j=0;j<8;++j){uint k=base+chunk+j;
                float weight=s*float(w[ulong(n)*kdim+k])+b;
                dot+=x[ulong(g.y)*kdim+k]*weight;}
            result+=dot;
        }
    }
    result+=simd_shuffle_down(result,4);
    result+=simd_shuffle_down(result,2);
    result+=simd_shuffle_down(result,1);
    if(lane==0)y[ulong(g.y)*ndim+n]=mlx_bf(result);
}

kernel void eg2_embed(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1])return;ulong at=ulong(ids[i/p[0]])*p[0]+i%p[0];float v;
    if(p[2]==0x100||p[2]==0x108){bool four=p[2]==0x100;ulong total=ulong(p[0])*p[4];device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+(four?total/2:total));
        uint code=four?(w[at/2]>>((at%2)*4))&15:w[at];v=float(code)*float(scales[at/64])+float(scales[total/64+at/64]);}
    else v=weight(w,p[2],at);
    // Apple10 can miscompile a nested bfloat conversion of a parameter-derived
    // scalar. Round its IEEE bits explicitly, as in the diffusion embedder.
    float scale=as_type<float>(p[3]);float bfscale=as_type<float>((p[3]+0x7fff+((p[3]>>16)&1))&0xffff0000);
    out[i]=p[5]?mlx_bf(mlx_bf(v)*bfscale):v*scale;
}
#define EG2_ATTN(D) \
kernel void eg2_attention##D(device float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* ends [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float kv[16*128],pr[32*16],scores[32*16],hi[32],den[32],corr[32]; \
gemma_prefill<D,16,32,float,false,false,float,false,true>(q,k,v,meta,ends,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,pr,scores,hi,den,corr,ends); }
EG2_ATTN(256)
EG2_ATTN(512)
#undef EG2_ATTN
// GGUF attention's operand cut is F16, like its tensor projection lane.
// Keep online maxima, probabilities and accumulation F32. KT32 amortizes
// the score/softmax barriers without growing a quadratic attention buffer.
#define EG2_HALF_ATTN(D) \
kernel void eg2_half_attention##D(device half* q [[buffer(0)]],device const half* k [[buffer(1)]],device const half* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* ends [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup half kv[32*128];threadgroup float pr[32*32],scores[32*32],hi[32],den[32],corr[32]; \
gemma_prefill<D,32,32,half,false,false,half,false,true,float,false,128>(q,k,v,meta,ends,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,pr,scores,hi,den,corr,ends); }
EG2_HALF_ATTN(256)
EG2_HALF_ATTN(512)
#undef EG2_HALF_ATTN
// Shader validation adds compiler staging to explicit threadgroup arrays.
// Stream 128-channel panels to stay below 32 KiB even with that staging.
// In the F32-probability arm, each row subgroup owns the same score/probability
// locations: after reducing its row maximum, it can overwrite its own scores
// in place. The different-width BF16 fallback retains a separate probability
// plane, since aliasing that one would overwrite other threads' F32 inputs.
#define EG2_BATTN(D) \
kernel void eg2_mlx_attention##D(device bfloat* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* ends [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup bfloat kv[32*128],pr[32*32];threadgroup float scores[32*32],hi[32],den[32],corr[32]; \
if(D==512 || ends[tiles[2*g.y]]+1<1024)dg_attention<D,bfloat,true,bfloat,32,32>(q,k,v,meta,ends,out,tiles,ends,p,g,tid,kv,pr,scores,hi,den); \
else gemma_prefill<D,32,32,bfloat,false,false,bfloat,true,true,float,true,128>(q,k,v,meta,ends,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,scores,scores,hi,den,corr,ends); }
EG2_BATTN(256)
EG2_BATTN(512)
#undef EG2_BATTN

// p: width, weight dtype, BF16 boundaries, input scale bits, residual flag,
// scalar dtype. A fused post-norm/add/scale saves whole residual passes while
// preserving each upstream rounding boundary.
kernel void eg2_norm(device const float* x [[buffer(0)]],device const uchar* w [[buffer(1)]],device float* out [[buffer(2)]],device const float* residual [[buffer(3)]],device const uchar* scalar [[buffer(4)]],constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    threadgroup float sums[4];uint n=p[0];ulong b=ulong(row)*n;float sum=0;
    float scale=p[2]?dg_scalar_bf(as_type<float>(p[3])):as_type<float>(p[3]);
    for(uint j=tid*4;j<n;j+=512)for(uint d=0;d<4 && j+d<n;++d){float v=x[b+j+d]*scale;if(p[2])v=mlx_bf(v);sum+=v*v;}
    sum=simd_sum(sum);if(tid%32==0)sums[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=precise::rsqrt(simd_sum(tid%32<4?sums[tid%32]:0.0f)/float(n)+1e-6f);
    for(uint j=tid;j<n;j+=128){float v=x[b+j]*scale;if(p[2])v=mlx_bf(v);
        v*=inv;if(p[2])v=mlx_bf(v);v*=weight(w,p[1],j);if(p[2])v=mlx_bf(v);
        if(p[4]){v+=residual[b+j];if(p[2])v=mlx_bf(v);v*=weight(scalar,p[5],0);if(p[2])v=mlx_bf(v);}
        out[b+j]=v;}
}

// Separate q/k/v head normalization; full split-half RoPE. Unlike generative
// Gemma 4, global heads rotate ALL dimensions. No K=V projection shortcut.
// p: head count, head width, weight dtype, BF16 arithmetic, rope base bits
// (zero for V), output storage (0 F32, 1 BF16, 2 F16).
kernel void eg2_heads(device const float* x [[buffer(0)]],device const uchar* w [[buffer(1)]],device const uint* meta [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    threadgroup float sums[4];uint hd=p[1],threads=hd/4;ulong b=(ulong(g.y)*p[0]+g.x)*hd;float sum=0;
    for(uint z=0;z<4;++z)sum+=x[b+tid*4+z]*x[b+tid*4+z];
    sum=simd_sum(sum);if(tid%32==0)sums[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=precise::rsqrt(simd_sum(tid%32<threads/32?sums[tid%32]:0.0f)/float(hd)+1e-6f);
    for(uint j=tid;j<hd/2;j+=threads){float a=x[b+j]*inv,bv=x[b+j+hd/2]*inv;
        if(p[3]){a=mlx_bf(a);bv=mlx_bf(bv);}
        if(p[4]){a*=weight(w,p[2],j);bv*=weight(w,p[2],j+hd/2);if(p[3]){a=mlx_bf(a);bv=mlx_bf(bv);}
            // The reference first materializes FP32 inverse frequencies and
            // then multiplies positions. Dividing positions by the period
            // instead loses that rounding boundary at long positions.
            float period=p[3]?precise::pow(as_type<float>(p[4]),float(j)*2.0f/float(hd)):pow(as_type<float>(p[4]),float(j)*2.0f/float(hd));
            float frequency=precise::divide(1.0f,period);
            float angle=p[3]?float(meta[g.y*2+1])*frequency:float(meta[g.y*2+1])/period;
            // MLX's standalone sin/cos use precise range reduction. Fast
            // trigonometry differs by BF16 steps at high context positions.
            float c=p[3]?precise::cos(angle):cos(angle),s=p[3]?precise::sin(angle):sin(angle);
            if(p[3]){c=mlx_bf(c);s=mlx_bf(s);float ac=mlx_bf(a*c),bs=mlx_bf(bv*s),as=mlx_bf(a*s),bc=mlx_bf(bv*c);a=mlx_bf(ac-bs);bv=mlx_bf(as+bc);}
            else {float next=a*c-bv*s;bv=a*s+bv*c;a=next;}}
        if(p[5]==2){device half* dst=reinterpret_cast<device half*>(out);dst[b+j]=half(a);dst[b+j+hd/2]=half(bv);}
        else if(p[5]){device bfloat* dst=reinterpret_cast<device bfloat*>(out);dst[b+j]=bfloat(a);dst[b+j+hd/2]=bfloat(bv);}
        else {out[b+j]=p[3]?mlx_bf(a):a;out[b+j+hd/2]=p[3]?mlx_bf(bv):bv;}}
}

// p: rows, layer index, BF16. PLE is token-major, layer-major, width512.
kernel void eg2_ple_gate(device float* gate [[buffer(0)]],device const float* ple [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if(i>=p[0]*512)return;float x=gate[i],v;
    if(p[2]){float cube=mlx_bf(pow(x,3.0f));float z=mlx_bf(x+mlx_bf(mlx_bf(0.044715f)*cube));z=mlx_bf(mlx_bf(0.7978845608028654f)*z);z=mlx_bf(1.0f+mlx_bf(precise::tanh(z)));v=mlx_bf(mlx_bf(0.5f*x)*z);}
    else v=vis_gelu(x);
    v*=ple[(ulong(i/512)*24+p[1])*512+i%512];gate[i]=p[2]?mlx_bf(v):v;
}

// Mask-aware mean pooling and all four MRL normalizations stay on the GPU.
// The normal output uses 768; the same kernel accepts 128/256/512 prefixes.
kernel void eg2_pool(device const float* x [[buffer(0)]],device const uint* ranges [[buffer(1)]],device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint seq [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[768],sums[8];uint first=ranges[seq*2],count=ranges[seq*2+1],dim=p[0];float ss=0;
    for(uint j=tid;j<dim;j+=256){float v=0;for(uint t=0;t<count;++t)v+=x[ulong(first+t)*768+j];v/=float(count);values[j]=v;ss+=v*v;}
    ss=simd_sum(ss);if(tid%32==0)sums[tid/32]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=1.0f/max(sqrt(simd_sum(tid%32<8?sums[tid%32]:0.0f)),1e-12f);
    for(uint j=tid;j<dim;j+=256)out[ulong(seq)*dim+j]=values[j]*inv;
}
