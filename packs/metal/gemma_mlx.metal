// Native Gemma/Muse affine-checkpoint arithmetic. BF16 operation boundaries
// follow MLX-VLM's public graph; all kernels are original Paddock code. The
// common paged/ring attention templates keep their GGUF defaults unchanged.
kernel void gmlx_centered_weights(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]+=1.0f;
}
kernel void gmlx_round(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=mlx_bf(x[i]);
}
kernel void gmlx_embed(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1])out[i]=mlx_bf(mlx_bf(mlx_affine_value(w,p[0],p[2],ulong(ids[i/p[0]])*p[0]+i%p[0]))*mlx_bf(as_type<float>(p[3])));
}
inline float gmlx_inv(device const float* x,uint n,float eps,uint tid,uint threads,threadgroup float* sums) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float sum=0;
    for(uint at=tid*4;at<n;at+=threads*4)for(uint j=0;j<4 && at+j<n;++j){float v=x[at+j];sum+=v*v;}
    sum=simd_sum(sum);if(tid%32==0)sums[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    float inv=precise::rsqrt(precise::divide(simd_sum(tid%32<threads/32?sums[tid%32]:0.0f),float(n))+eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inv;
}
// p: width, arithmetic (0=MLX RMS,1=full-F32 weighted RMS,2=unweighted), eps.
kernel void gmlx_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    threadgroup float sums[32];uint n=p[0],threads=min(1024u,((n+127)/128)*32);ulong b=ulong(r)*n;
    float inv=gmlx_inv(x+b,n,as_type<float>(p[2]),tid,threads,sums);
    for(uint d=tid;d<n;d+=threads){float z=x[b+d]*inv;
        out[b+d]=p[1]==2?mlx_bf(z):mlx_bf((p[1]==0?mlx_bf(z):z)*w[d]);}
}
// p: width, centered-Muse flag, pre eps, post eps, layer scalar. Gemma
// scalar multiplies the complete residual, never only the FFN branch.
kernel void gmlx_sandwich(device float* x [[buffer(0)]],device const float* delta [[buffer(1)]],
    device const float* post [[buffer(2)]],device const float* next [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    threadgroup float sums[32];uint n=p[0],threads=min(1024u,((n+127)/128)*32);ulong b=ulong(r)*n;
    float inv=gmlx_inv(delta+b,n,as_type<float>(p[3]),tid,threads,sums);
    for(uint d=tid;d<n;d+=threads){float z=delta[b+d]*inv;
        z=mlx_bf((p[1]?z:mlx_bf(z))*post[d]);
        x[b+d]=mlx_bf(mlx_bf(x[b+d]+z)*as_type<float>(p[4]));}
    threadgroup_barrier(mem_flags::mem_device);
    inv=gmlx_inv(x+b,n,as_type<float>(p[2]),tid,threads,sums);
    for(uint d=tid;d<n;d+=threads){float z=x[b+d]*inv;out[b+d]=mlx_bf((p[1]?z:mlx_bf(z))*next[d]);}
}
// MLX standard RoPE is NEOX for both models. Muse's GGUF conversion
// permutes rows into interleaved pairs; the native checkpoint is unpermuted.
template<bool Muse>
inline float2 gmlx_rot(float2 x,uint j,uint hd,uint pos,float base,bool global) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if(global && (Muse || j>=hd/8))return x;
    float exponent=2.0f*float(j)/float(hd);
    float angle=precise::divide(float(pos),pow(base,exponent));
    if constexpr(!Muse)angle=global?precise::divide(float(pos),pow(base,exponent)):
        float(pos)*precise::exp2(-exponent*precise::log2(base));
    float c=cos(angle),s=sin(angle);
    if constexpr(Muse){c=mlx_bf(c);s=mlx_bf(s);return float2(mlx_bf(mlx_bf(x.x*c)-mlx_bf(x.y*s)),mlx_bf(mlx_bf(x.x*s)+mlx_bf(x.y*c)));}
    return float2(mlx_bf(fma(x.x,c,-x.y*s)),mlx_bf(fma(x.x,s,x.y*c)));
}
template<bool Muse>
inline void gmlx_qnorm_impl(device float* q,device const float* w,device const uint* meta,constant uint* p,uint2 g,uint lane) {
    #pragma clang fp contract(off)
    uint hd=p[1];ulong b=(ulong(g.y)*p[0]+g.x)*hd;float sum=0;
    for(uint i=lane*4;i<hd;i+=128)for(uint d=0;d<4;++d)sum+=q[b+i+d]*q[b+i+d];
    float inv=precise::rsqrt(simd_sum(sum)/float(hd)+as_type<float>(p[3]));
    for(uint j=lane;j<hd/2;j+=32){float2 z=mlx_bf(float4(q[b+j]*inv,q[b+j+hd/2]*inv,0,0)).xy;
        if constexpr(Muse)z=float2(mlx_bf(z.x*3.87f),mlx_bf(z.y*3.87f));
        else z=float2(mlx_bf(z.x*w[j]),mlx_bf(z.y*w[j+hd/2]));
        z=gmlx_rot<Muse>(z,j,hd,meta[g.y*2+1],as_type<float>(p[4]),p[5]!=0);
        q[b+j]=z.x;q[b+j+hd/2]=z.y;}
}
template<bool Muse>
inline void gmlx_store_impl(device const float* k,device const float* v,device const float* w,
    device const uint* meta,device const uint* pages,device bfloat* ko,device bfloat* vo,constant uint* p,uint2 g,uint lane) {
    #pragma clang fp contract(off)
    uint hd=p[5],slot=meta[2*g.y],pos=meta[2*g.y+1];ulong b=(ulong(g.y)*p[1]+g.x)*hd;
    ulong out=(ulong(gemma_physical(pages,slot,pos,p))*p[1]+g.x)*hd;float ks=0,vs=0;
    for(uint i=lane*4;i<hd;i+=128)for(uint d=0;d<4;++d){ks+=k[b+i+d]*k[b+i+d];vs+=v[b+i+d]*v[b+i+d];}
    float ki=precise::rsqrt(simd_sum(ks)/float(hd)+as_type<float>(p[7]));
    float vi=Muse?1.0f:precise::rsqrt(simd_sum(vs)/float(hd)+as_type<float>(p[7]));
    for(uint j=lane;j<hd/2;j+=32){float2 z(mlx_bf(k[b+j]*ki),mlx_bf(k[b+j+hd/2]*ki));
        if constexpr(!Muse)z=float2(mlx_bf(z.x*w[j]),mlx_bf(z.y*w[j+hd/2]));
        z=gmlx_rot<Muse>(z,j,hd,pos,as_type<float>(p[8]),p[3]==0);
        ko[out+j]=bfloat(z.x);ko[out+j+hd/2]=bfloat(z.y);
        vo[out+j]=bfloat(v[b+j]*vi);vo[out+j+hd/2]=bfloat(v[b+j+hd/2]*vi);}
}
#define GMLX_HEAD(NAME,MUSE) \
kernel void NAME##_qnorm(device float* q [[buffer(0)]],device const float* w [[buffer(1)]],device const uint* meta [[buffer(2)]],device const float* unused [[buffer(3)]],constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {gmlx_qnorm_impl<MUSE>(q,w,meta,p,g,lane);} \
kernel void NAME##_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],device const float* w [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const float* unused [[buffer(5)]],device bfloat* ko [[buffer(6)]],device bfloat* vo [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {gmlx_store_impl<MUSE>(k,v,w,meta,pages,ko,vo,p,g,lane);}
GMLX_HEAD(gmlx,false)
GMLX_HEAD(mmlx,true)
#undef GMLX_HEAD

#define GMLX_ATTN(HD,GQA,MUSE) \
kernel void gmlx_decode##HD(device const float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) { \
threadgroup float scores[GQA*32],prob[GQA*32],highs[GQA],sums[GQA];gemma_decode<HD,GQA,MUSE,bfloat>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums); } \
kernel void gmlx_prefill##HD(device float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float kv[16*128],prob[16*32],scores[16*32],maximum[32],denom[32],correction[32];gemma_prefill<HD,16,32,float,false,MUSE,bfloat>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction); } \
kernel void gmlx_image_prefill##HD(device float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float kv[16*128],prob[16*32],scores[16*32],maximum[32],denom[32],correction[32];gemma_prefill<HD,16,32,float,true,MUSE,bfloat>(q,k,v,meta,pages,out,p,g.x,tiles[2*g.y],tiles[2*g.y+1],tid,kv,prob,scores,maximum,denom,correction,limits); }
GMLX_ATTN(128,16,true)
GMLX_ATTN(256,2,false)
GMLX_ATTN(512,8,false)
#undef GMLX_ATTN

#define GMLX_PIPE(HD,GQA,A) \
kernel void gmlx_value##A##_##HD(device const float* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device const uint* rows [[buffer(5)]],device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) { \
threadgroup float scores[GQA*32],prob[GQA*32],highs[GQA],sums[GQA];gemma_decode<HD,GQA,false,bfloat,A>(q,k,v,meta,pages,rows,out,p,g,tid,lane,sg,scores,prob,highs,sums); }
GMLX_PIPE(256,2,2)
#undef GMLX_PIPE
kernel void gmlx_gate(device float* out [[buffer(0)]],device const float* gate [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])out[i]=mlx_bf(mlx_bf(out[i])*mlx_sigmoid_bf(gate[i]));
}
kernel void gmlx_geglu(device float* gate [[buffer(0)]],device const float* up [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if(i<p[0]){
        float x=gate[i],cube=mlx_bf(pow(x,3.0f));
        float z=mlx_bf(x+mlx_bf(mlx_bf(0.044715f)*cube));
        z=mlx_bf(mlx_bf(0.7978845608028654f)*z);
        z=mlx_bf(1.0f+mlx_bf(precise::tanh(z)));
        gate[i]=mlx_bf(mlx_bf(mlx_bf(0.5f*x)*z)*up[i]);
    }
}
kernel void gmlx_softcap(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]){float c=as_type<float>(p[1]);float v=mlx_bf(x[i]*mlx_bf(as_type<float>(p[2])));
        x[i]=mlx_bf(mlx_bf(precise::tanh(mlx_bf(v/c)))*c);}
}
