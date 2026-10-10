// Original Paddock kernels for MLX's documented affine group-64 storage.
// Format reference: mlx.core.quantize/quantized_matmul. Packed codes are
// consumed directly; neither a framework backend nor persistent expansion.
#ifdef PADDOCK_APPLE9
// macOS 26.5's Apple9 compiler crashes when nested BF16 conversions reach
// q4b_ple_gate's constant program. Preserve round-to-nearest-even boundaries
// using integer bits on the GPU. Keep NaNs distinct from infinity.
inline float mlx_bf(float v) {
    uint bits=as_type<uint>(v);
    if((bits&0x7fffffff)>0x7f800000) return as_type<float>((bits&0xffff0000)|0x00400000);
    return as_type<float>((bits+0x7fff+((bits>>16)&1))&0xffff0000);
}
inline float4 mlx_bf(float4 v) { return float4(mlx_bf(v.x),mlx_bf(v.y),mlx_bf(v.z),mlx_bf(v.w)); }
#else
inline float mlx_bf(float v) { return float(bfloat(v)); }
inline float4 mlx_bf(float4 v) { return float4(vec<bfloat,4>(v)); }
#endif
inline float mlx_affine_value(device const uchar* w,uint K,uint N,ulong i) {
    uint code=(reinterpret_cast<device const uint*>(w)[i/8]>>((i%8)*4))&15;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    return float(code)*float(scales[i/64])+float(scales[ulong(K)*N/64+i/64]);
}
inline void mlx_affine_stage8(device const uchar* w,uint K,uint N,ulong i,threadgroup bfloat* dst) {
    uint bits=reinterpret_cast<device const uint*>(w)[i/8];
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    float s=float(scales[i/64]),b=float(scales[ulong(K)*N/64+i/64]);
    float4 lo=float4((uint4(bits)>>uint4(0,4,8,12))&15)*s+b;
    float4 hi=float4((uint4(bits)>>uint4(16,20,24,28))&15)*s+b;
    *reinterpret_cast<threadgroup vec<bfloat,4>*>(dst)=vec<bfloat,4>(lo);
    *reinterpret_cast<threadgroup vec<bfloat,4>*>(dst+4)=vec<bfloat,4>(hi);
}
kernel void mlx_small(device const bfloat* x [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])out[i]=p[1]==1 ? -exp(float(x[i])) : float(x[i]);
}
kernel void mlx_embed(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1])out[i]=mlx_bf(mlx_affine_value(w,p[0],p[2],ulong(ids[i/p[0]])*p[0]+i%p[0]));
}
kernel void mlx_input(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint pitch=(p[0]+127)/128*128,rows=(p[1]+127)/128*128;
    if(i<pitch*rows)out[i]=bfloat(i/pitch<p[1] && i%pitch<p[0] ? x[ulong(i/pitch)*p[0]+i%pitch] : 0.0f);
}
kernel void mlx_input_compact(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]*p[1])out[i]=bfloat(x[i]);
}

// Contiguous packed-word loads; each pair of output columns shares input
// across requests. A full SIMD group shortens batched K loops without the
// strided group-major access pattern rejected by the serving measurement.
template<uint R,uint Lanes>
inline void mlx_affine_rows(device const uchar* w,device const float* x,device float* out,
    uint K,uint N,uint n,uint lane,uint valid) {
    if(n>=N)return;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float4 sums[2][R];for(uint c=0;c<2;++c)for(uint r=0;r<R;++r)sums[c][r]=0;
    for(uint k=lane*8;k<K;k+=Lanes*8) {
        float4 lo[2],hi[2];float scales2[2],biases2[2];
        for(uint c=0;c<2;++c) {
            ulong i=ulong(min(n+c,N-1))*K+k;
            uint bits=reinterpret_cast<device const uint*>(w)[i/8];
            float s=float(scales[i/64]),b=float(biases[i/64]);
            scales2[c]=s;biases2[c]=b;
            lo[c]=float4((uint4(bits)>>uint4(0,4,8,12))&15);
            hi[c]=float4((uint4(bits)>>uint4(16,20,24,28))&15);
            if constexpr(R!=1){lo[c]=lo[c]*s+b;hi[c]=hi[c]*s+b;}
        }
        for(uint r=0;r<R;++r) {
            if(r>=valid)continue;
            float4 a=*reinterpret_cast<device const float4*>(x+ulong(r)*K+k);
            float4 b=*reinterpret_cast<device const float4*>(x+ulong(r)*K+k+4);
            for(uint c=0;c<2;++c) {
                if constexpr(R==1) {
                    // MLX's one-vector affine contraction reduces the bias
                    // operand in BF16 groups of four before F32 accumulation.
                    float sa=mlx_bf(mlx_bf(mlx_bf(a.x+a.y)+a.z)+a.w);
                    float sb=mlx_bf(mlx_bf(mlx_bf(b.x+b.y)+b.z)+b.w);
                    sums[c][r].x+=scales2[c]*(dot(lo[c],a)+dot(hi[c],b))+biases2[c]*(sa+sb);
                } else {sums[c][r]=fma(lo[c],a,sums[c][r]);sums[c][r]=fma(hi[c],b,sums[c][r]);}
            }
        }
    }
    for(uint c=0;c<2;++c)for(uint r=0;r<R;++r) {
        float4 s=sums[c][r];float sum=kquant_sum<Lanes>(s.x+s.y+s.z+s.w);
        if(lane==0 && n+c<N && r<valid)out[ulong(r)*N+n+c]=mlx_bf(sum);
    }
}
#define MLX_AFFINE(R) \
kernel void mlx_affine##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
constexpr uint Lanes=R==1?8:32,Columns=128/Lanes*2;uint g=group.x; \
uint n0=(p[1]+Columns-1)/Columns,n1=(p[2]+Columns-1)/Columns,N=p[1];device const uchar* w=w0;device float* out=o0; \
if(g>=n0){g-=n0;N=p[2];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];w=w2;out=o2;}} \
mlx_affine_rows<R,Lanes>(w,x+ulong(group.y)*R*p[0],out+ulong(group.y)*R*N,p[0],N,g*Columns+(tid/Lanes)*2,tid%Lanes,min(uint(R),p[4]-group.y*R));}
MLX_AFFINE(1)
MLX_AFFINE(2)
MLX_AFFINE(3)
MLX_AFFINE(4)
MLX_AFFINE(5)
#undef MLX_AFFINE

// Fixed F32 arithmetic contract, independent of the number of live requests.
// Disable implicit regrouping only here; keep each intended FMA explicit.
// BF16 input/output boundaries remain, without the old R1-only rounded bias sum.
inline float mlx_affine_contract8(float4 a,float4 z,float4 lo,float4 hi,float accumulated) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float v=a.x*lo.x;
    v=fma(a.y,lo.y,v);v=fma(a.z,lo.z,v);v=fma(a.w,lo.w,v);
    v=fma(z.x,hi.x,v);v=fma(z.y,hi.y,v);v=fma(z.z,hi.z,v);v=fma(z.w,hi.w,v);
    return accumulated+v;
}
// One column per eight-lane partition, one scalar accumulator per request.
// Coefficients stay live across eight packed words; decoded weights live for
// one word. The false specialization retains the independent legacy baseline.
template<uint R,typename X=float,bool Stable=false>
inline void mlx_affine_stream_rows(device const uchar* w,device const X* x,device float* out,
    uint K,uint N,uint column,uint lane,uint valid) {
    if(column>=N)return;
    device const uint* codes=reinterpret_cast<device const uint*>(w)+ulong(column)*(K/8);
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float sum[R];for(uint r=0;r<R;++r)sum[r]=0;
    for(uint base=lane*64;base<K;base+=512) {
        ulong group=ulong(column)*(K/64)+base/64;
        float s=float(scales[group]),b=float(biases[group]);
        #pragma unroll
        for(uint word=0;word<8;++word) {
            uint bits=codes[base/8+word];
            float4 lo,hi;
            if constexpr(Stable) {
                lo=fma(float4((uint4(bits)>>uint4(0,4,8,12))&15),float4(s),float4(b));
                hi=fma(float4((uint4(bits)>>uint4(16,20,24,28))&15),float4(s),float4(b));
            } else {
                lo=float4((uint4(bits)>>uint4(0,4,8,12))&15)*s+b;
                hi=float4((uint4(bits)>>uint4(16,20,24,28))&15)*s+b;
            }
            #pragma unroll
            for(uint r=0;r<R;++r) {
                uint row=min(r,valid-1);
                device const X* at=x+ulong(row)*K+base+word*8;
                float4 a=float4(*reinterpret_cast<device const vec<X,4>*>(at));
                float4 z=float4(*reinterpret_cast<device const vec<X,4>*>(at+4));
                if constexpr(Stable) { sum[r]=mlx_affine_contract8(a,z,lo,hi,sum[r]); }
                else {
                    float v=a.x*lo.x;
                    v+=a.y*lo.y;v+=a.z*lo.z;v+=a.w*lo.w;
                    v+=z.x*hi.x;v+=z.y*hi.y;v+=z.z*hi.z;v+=z.w*hi.w;
                    sum[r]+=v;
                }
            }
        }
    }
    for(uint r=0;r<R;++r) {
        float v=kquant_sum<8>(sum[r]);
        if(lane==0 && r<valid)out[ulong(r)*N+column]=mlx_bf(v);
    }
}
#define MLX_STREAM(R) \
kernel void mlx_affine_stream##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
uint g=group.x,n0=(p[1]+7)/8,n1=(p[2]+7)/8,N=p[1];device const uchar* w=w0;device float* out=o0; \
if(g>=n0){g-=n0;N=p[2];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];w=w2;out=o2;}} \
mlx_affine_stream_rows<R>(w,x+ulong(group.y)*R*p[0],out+ulong(group.y)*R*N,p[0],N,g*8+tid/8,tid%8,min(uint(R),p[4]-group.y*R));}
MLX_STREAM(2)
MLX_STREAM(3)
MLX_STREAM(4)
MLX_STREAM(5)
#undef MLX_STREAM

#define MLX_COMPACT(R) \
kernel void mlx_affine_compact##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
uint g=group.x,n0=(p[1]+7)/8,n1=(p[2]+7)/8,N=p[1];device const uchar* w=w0;device float* out=o0; \
if(g>=n0){g-=n0;N=p[2];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];w=w2;out=o2;}} \
mlx_affine_stream_rows<R,bfloat>(w,x+ulong(group.y)*R*p[0],out+ulong(group.y)*R*N,p[0],N,g*8+tid/8,tid%8,min(uint(R),p[4]-group.y*R));}
MLX_COMPACT(2)
MLX_COMPACT(3)
MLX_COMPACT(4)
MLX_COMPACT(5)
#undef MLX_COMPACT

#define MLX_STABLE(R) \
kernel void mlx_affine_stable##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
uint g=group.x,n0=(p[1]+7)/8,n1=(p[2]+7)/8,N=p[1];device const uchar* w=w0;device float* out=o0; \
if(g>=n0){g-=n0;N=p[2];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];w=w2;out=o2;}} \
mlx_affine_stream_rows<R,bfloat,true>(w,x+ulong(group.y)*R*p[0],out+ulong(group.y)*R*N,p[0],N,g*8+tid/8,tid%8,min(uint(R),p[4]-group.y*R));}
MLX_STABLE(1)
MLX_STABLE(2)
MLX_STABLE(3)
MLX_STABLE(4)
MLX_STABLE(5)
#undef MLX_STABLE

// Small backbones are dispatch-sensitive. Consume their F32 scheduler planes
// directly, retaining the compact route's BF16 input cut inside the load.
// No conversion dispatch or scratch write; identical eight-lane arithmetic.
template<uint R>
inline void mlx_affine_direct_rows(device const uchar* w,device const float* x,device float* out,
    uint K,uint N,uint column,uint lane,uint valid) {
    if(column>=N)return;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float sum[2][R];for(uint c=0;c<2;++c)for(uint r=0;r<R;++r)sum[c][r]=0;
    for(uint base=lane*64;base<K;base+=512) {
        float scale[2],bias[2];
        for(uint c=0;c<2;++c) {
            ulong group=ulong(min(column+c,N-1))*(K/64)+base/64;
            scale[c]=float(scales[group]);bias[c]=float(biases[group]);
        }
        #pragma unroll
        for(uint word=0;word<8;++word) {
            float4 lo[2],hi[2];
            for(uint c=0;c<2;++c) {
                ulong at=ulong(min(column+c,N-1))*K+base+word*8;
                uint bits=reinterpret_cast<device const uint*>(w)[at/8];
                lo[c]=fma(float4((uint4(bits)>>uint4(0,4,8,12))&15),float4(scale[c]),float4(bias[c]));
                hi[c]=fma(float4((uint4(bits)>>uint4(16,20,24,28))&15),float4(scale[c]),float4(bias[c]));
            }
            #pragma unroll
            for(uint r=0;r<R;++r) {
                device const float* at=x+ulong(min(r,valid-1))*K+base+word*8;
                float4 a=mlx_bf(*reinterpret_cast<device const float4*>(at));
                float4 b=mlx_bf(*reinterpret_cast<device const float4*>(at+4));
                for(uint c=0;c<2;++c)sum[c][r]=mlx_affine_contract8(a,b,lo[c],hi[c],sum[c][r]);
            }
        }
    }
    for(uint c=0;c<2;++c)for(uint r=0;r<R;++r) {
        float v=kquant_sum<8>(sum[c][r]);
        if(lane==0 && r<valid && column+c<N)out[ulong(r)*N+column+c]=mlx_bf(v);
    }
}
#define MLX_STABLE_DIRECT(R) \
kernel void mlx_affine_direct##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
uint g=group.x,n0=(p[1]+15)/16,n1=(p[2]+15)/16,N=p[1];device const uchar* w=w0;device float* out=o0; \
if(g>=n0){g-=n0;N=p[2];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];w=w2;out=o2;}} \
mlx_affine_direct_rows<R>(w,x+ulong(group.y)*R*p[0],out+ulong(group.y)*R*N,p[0],N,g*16+tid/8*2,tid%8,min(uint(R),p[4]-group.y*R));}
MLX_STABLE_DIRECT(1)
MLX_STABLE_DIRECT(2)
MLX_STABLE_DIRECT(3)
MLX_STABLE_DIRECT(4)
MLX_STABLE_DIRECT(5)
#undef MLX_STABLE_DIRECT

// Single-token bandwidth path: a complete SIMD group shares sixteen input
// values across four independent output columns. Fewer scale/bias loads and
// shorter K loops than the small-batch eight-lane layout above.
kernel void mlx_affine_single(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]],
    device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint g=group,n0=(p[1]+15)/16,n1=(p[2]+15)/16,N=p[1],K=p[0];device const uchar* w=w0;device float* out=o0;
    if(g>=n0){g-=n0;N=p[2];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];w=w2;out=o2;}}
    uint n=g*16+(tid/32)*4,lane=tid%32;
    if(n>=N)return;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float sums[4]={0,0,0,0};
    for(uint k=lane*16;k<K;k+=512) {
        float4 a[4];float bias_sum=0;
        for(uint j=0;j<4;++j) {
            a[j]=*reinterpret_cast<device const float4*>(x+k+j*4);
            bias_sum+=mlx_bf(mlx_bf(mlx_bf(a[j].x+a[j].y)+a[j].z)+a[j].w);
        }
        for(uint c=0;c<4;++c) {
            ulong i=ulong(min(n+c,N-1))*K+k;
            uint2 bits=*reinterpret_cast<device const uint2*>(w+i/2);
            float sum=0;
            for(uint j=0;j<4;++j) {
                float4 codes=float4((uint4(bits[j/2])>>(uint4(0,4,8,12)+uint4((j%2)*16)))&15);
                sum+=dot(codes,a[j]);
            }
            sums[c]+=sum*float(scales[i/64])+bias_sum*float(biases[i/64]);
        }
    }
    for(uint c=0;c<4;++c) {
        float sum=simd_sum(sums[c]);
        if(lane==0 && n+c<N)out[n+c]=mlx_bf(sum);
    }
}


// Bias operands depend only on the activation, not the output column. Prepare
// the exact single-stream BF16 quartet reductions once for all fused planes.
kernel void mlx_affine_bias(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]/16)return;
    float sum=0;
    for(uint j=0;j<4;++j) {
        float4 a=*reinterpret_cast<device const float4*>(x+ulong(i)*16+j*4);
        sum+=mlx_bf(mlx_bf(mlx_bf(a.x+a.y)+a.z)+a.w);
    }
    out[i]=sum;
}

// Two/three/four speculative positions share packed weights without changing
// single-stream decode's accumulation tree. Narrow adaptive rounds must not
// execute duplicate positions merely to fill a four-row tile.
template<uint R,uint C=4>
inline void mlx_affine_verify_rows(device const uchar* w0,device const uchar* w1,device const uchar* w2,
    device const float* x,device float* o0,device float* o1,device float* o2,
    device const float* bias_input,constant uint* p,uint2 group,uint tid) {
    constexpr uint Columns=4*C;
    uint g=group.x,n0=(p[1]+Columns-1)/Columns,n1=(p[2]+Columns-1)/Columns,N=p[1],K=p[0];device const uchar* w=w0;device float* out=o0;
    if(g>=n0){g-=n0;N=p[2];w=w1;out=o1;if(g>=n1){g-=n1;N=p[3];w=w2;out=o2;}}
    uint n=g*Columns+(tid/32)*C,lane=tid%32,first=group.y*R,valid=min(R,p[4]-first);
    if(n>=N)return;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float sums[R][C];for(uint r=0;r<R;++r)for(uint c=0;c<C;++c)sums[r][c]=0;
    for(uint k=lane*16;k<K;k+=512) {
        float4 a[R][4];float bias_sum[R];
        for(uint r=0;r<R;++r) {
            ulong base=ulong(first+min(r,valid-1))*K+k;
            bias_sum[r]=bias_input[base/16];
            for(uint j=0;j<4;++j)a[r][j]=*reinterpret_cast<device const float4*>(x+base+j*4);
        }
        for(uint c=0;c<C;++c) {
            ulong i=ulong(min(n+c,N-1))*K+k;
            uint2 bits=*reinterpret_cast<device const uint2*>(w+i/2);
            float sum[R];for(uint r=0;r<R;++r)sum[r]=0;
            for(uint j=0;j<4;++j) {
                float4 codes=float4((uint4(bits[j/2])>>(uint4(0,4,8,12)+uint4((j%2)*16)))&15);
                for(uint r=0;r<R;++r)sum[r]+=dot(codes,a[r][j]);
            }
            for(uint r=0;r<R;++r)sums[r][c]+=sum[r]*float(scales[i/64])+bias_sum[r]*float(biases[i/64]);
        }
    }
    for(uint r=0;r<R;++r)for(uint c=0;c<C;++c) {
        float sum=simd_sum(sums[r][c]);
        if(lane==0 && n+c<N && r<valid)out[ulong(first+r)*N+n+c]=mlx_bf(sum);
    }
}
#define MLX_VERIFY(R) \
kernel void mlx_affine_verify_single##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
    device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
    device const float* bias_input [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    mlx_affine_verify_rows<R>(w0,w1,w2,x,o0,o1,o2,bias_input,p,group,tid); }
MLX_VERIFY(2)
MLX_VERIFY(3)
MLX_VERIFY(4)
#undef MLX_VERIFY

#define MLX_VERIFY_COLUMNS(NAME,R,C) \
kernel void NAME(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const float* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
device const float* bias_input [[buffer(7)]],constant uint* p [[buffer(8)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
mlx_affine_verify_rows<R,C>(w0,w1,w2,x,o0,o1,o2,bias_input,p,group,tid);}
MLX_VERIFY_COLUMNS(mlx_affine_verify_narrow2,2,1)
MLX_VERIFY_COLUMNS(mlx_affine_verify_narrow3,3,1)
MLX_VERIFY_COLUMNS(mlx_affine_verify_narrow4,4,1)
MLX_VERIFY_COLUMNS(mlx_affine_verify_half2,2,2)
MLX_VERIFY_COLUMNS(mlx_affine_verify_half3,3,2)
#undef MLX_VERIFY_COLUMNS


// Bounded BF16 tile staging; M5 TensorOps executes the batched contraction.
// BK=128 keeps the staging footprint bounded without a full weight expansion.
template<uint BM,uint Padding=0,uint Values=8,bool BitStore=false,bool Compact=false,uint BK=128>
inline void mlx_affine_tile(device const uchar* w,device bfloat* x,device float* out,
    uint K,uint N,uint M,uint2 g,uint tid,threadgroup bfloat* tile) {
    constexpr uint BN=32;
    static_assert(BK%64==0 && 64%Values==0 && Padding%8==0);
    uint pitch=(K+127)/128*128,padded=(M+127)/128*128,n=g.x*BN,m=g.y*BM;
    auto a=tensor(x,dextents<int,2>(pitch,padded),array<int,2>{1,int(pitch)});
    auto b=tensor(tile,extents<int,BK,BN>(),array<int,2>{1,BK+Padding});
    auto dst=tensor(out,dextents<int,2>(N,M),array<int,2>{1,int(N)});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<K;base+=BK) {
        for(uint i=tid*Values;i<BK*BN;i+=128*Values) {
            uint col=n+i/BK,k=base+i%BK;
            uint at=(i/BK)*(BK+Padding)+i%BK;
            if(col<N && k<K) {
                if constexpr(Values==8)mlx_affine_stage8(w,K,N,ulong(col)*K+k,tile+at);
                else {
                    ulong index=ulong(col)*K+k;
                    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
                    float scale=float(scales[index/64]),bias=float(scales[ulong(K)*N/64+index/64]);
                    #pragma unroll
                    for(uint j=0;j<Values;j+=8) {
                        uint bits=reinterpret_cast<device const uint*>(w)[(index+j)/8];
                        float4 lo=float4((uint4(bits)>>uint4(0,4,8,12))&15)*scale+bias;
                        float4 hi=float4((uint4(bits)>>uint4(16,20,24,28))&15)*scale+bias;
                        *reinterpret_cast<threadgroup vec<bfloat,4>*>(tile+at+j)=vec<bfloat,4>(lo);
                        *reinterpret_cast<threadgroup vec<bfloat,4>*>(tile+at+j+4)=vec<bfloat,4>(hi);
                    }
                }
            } else {for(uint j=0;j<Values;++j)tile[at+j]=bfloat(0);}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input_tile=a.slice<BK,BM>(base,m);
        op.run(input_tile,b,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if constexpr(Compact) {
        auto converted=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),bfloat>();
        #pragma unroll
        for(ushort i=0;i<acc.get_capacity();++i)converted[i]=bfloat(acc[i]);
        auto target=tensor(reinterpret_cast<device bfloat*>(out),dextents<int,2>(N,M),array<int,2>{1,int(N)});
        converted.store(target.slice(n,m));
    } else if constexpr(BitStore) {
        // Explicit integer RNE survives the cooperative F32 store even under
        // GPU instrumentation. The native BF16 cast can be optimized away here.
        #pragma unroll
        for(ushort i=0;i<acc.get_capacity();++i) {
            uint bits=as_type<uint>(acc[i]);
            uint rounded=(bits&0x7fffffff)>0x7f800000 ? (bits&0xffff0000)|0x00400000 :
                (bits+0x7fff+((bits>>16)&1))&0xffff0000;
            acc[i]=as_type<float>(rounded);
        }
        acc.store(dst.slice(n,m));
    } else if constexpr(BM==256) {
        // Materialize rounding at each valid store. A wide cooperative F32
        // store can otherwise bypass the preceding in-place BF16 cast under
        // shader instrumentation (observed on a ragged 511-row tile).
        for(auto it=acc.begin();it!=acc.end();++it) {
            auto ij=it.get_multidimensional_index();
            if(it.is_valid_element() && n+uint(ij[0])<N && m+uint(ij[1])<M)
                out[ulong(m+ij[1])*N+n+ij[0]]=mlx_bf(*it);
        }
    } else {
        for(uint i=0;i<acc.get_capacity();++i)acc[i]=mlx_bf(acc[i]);
        acc.store(dst.slice(n,m));
    }
}
#define MLX_AFFINE_TILE(BM) \
kernel void mlx_affine_tile##BM(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup bfloat tile[128*32];uint n0=(p[1]+31)/32,n1=(p[2]+31)/32,N=p[1];device const uchar* w=w0;device float* out=o0; \
if(g.x>=n0){g.x-=n0;N=p[2];w=w1;out=o1;if(g.x>=n1){g.x-=n1;N=p[3];w=w2;out=o2;}} \
mlx_affine_tile<BM>(w,x,out,p[0],N,p[4],g,tid,tile);}
MLX_AFFINE_TILE(32)
MLX_AFFINE_TILE(64)
#undef MLX_AFFINE_TILE

// Shared-memory padding offsets neighbouring columns from the same banks.
// Only the storage stride changes: BF16 values and K reduction stay identical.
kernel void mlx_affine_prefill64(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]],
    device bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat tile[136*32];
    uint n0=(p[1]+31)/32,n1=(p[2]+31)/32,N=p[1];device const uchar* w=w0;device float* out=o0;
    if(g.x>=n0){g.x-=n0;N=p[2];w=w1;out=o1;if(g.x>=n1){g.x-=n1;N=p[3];w=w2;out=o2;}}
    mlx_affine_tile<64,8>(w,x,out,p[0],N,p[4],g,tid,tile);
}

#define MLX_PREFILL_LOAD(NAME,ROWS) \
kernel void NAME(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup bfloat tile[136*32];uint n0=(p[1]+31)/32,n1=(p[2]+31)/32,N=p[1];device const uchar* w=w0;device float* out=o0; \
if(g.x>=n0){g.x-=n0;N=p[2];w=w1;out=o1;if(g.x>=n1){g.x-=n1;N=p[3];w=w2;out=o2;}} \
mlx_affine_tile<ROWS,8,32>(w,x,out,p[0],N,p[4],g,tid,tile);}
MLX_PREFILL_LOAD(mlx_affine_prefill_load32,64)
MLX_PREFILL_LOAD(mlx_affine_prefill_load32_m32,32)
MLX_PREFILL_LOAD(mlx_affine_prefill_rows256,256)
#undef MLX_PREFILL_LOAD

#ifndef PADDOCK_APPLE9
// Wider K staging amortizes synchronization while a smaller output tile
// bounds accumulator pressure. The BF16 operands, FP32 contraction and
// output rounding are unchanged. These routes require whole K tiles: padding
// weights with zero would not make an out-of-bounds/NaN input read safe.
#define MLX_PREFILL_WIDE(NAME,STEP,PAD,COMPACT) \
kernel void NAME(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
    device bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
    constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    if(p[0]%STEP!=0)return; \
    threadgroup bfloat tile[(STEP+PAD)*32]; \
    uint n0=(p[1]+31)/32,n1=(p[2]+31)/32,N=p[1];device const uchar* w=w0;device float* out=o0; \
    if(g.x>=n0){g.x-=n0;N=p[2];w=w1;out=o1;if(g.x>=n1){g.x-=n1;N=p[3];w=w2;out=o2;}} \
    mlx_affine_tile<128,PAD,32,true,COMPACT,STEP>(w,x,out,p[0],N,p[4],g,tid,tile); \
}
MLX_PREFILL_WIDE(mlx_affine_prefill_wide128,256,8,false)
MLX_PREFILL_WIDE(mlx_affine_prefill_deep128,384,8,false)
MLX_PREFILL_WIDE(mlx_affine_prefill_compact128,256,8,true)
#undef MLX_PREFILL_WIDE

kernel void mlx_affine_prefill_compact256(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]],
    device bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat tile[136*32];
    uint n0=(p[1]+31)/32,n1=(p[2]+31)/32,N=p[1];device const uchar* w=w0;device float* out=o0;
    if(g.x>=n0){g.x-=n0;N=p[2];w=w1;out=o1;if(g.x>=n1){g.x-=n1;N=p[3];w=w2;out=o2;}}
    mlx_affine_tile<256,8,32,false,true>(w,x,out,p[0],N,p[4],g,tid,tile);
}

kernel void mlx_affine_prefill_store256(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]],
    device bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat tile[136*32];
    uint n0=(p[1]+31)/32,n1=(p[2]+31)/32,N=p[1];device const uchar* w=w0;device float* out=o0;
    if(g.x>=n0){g.x-=n0;N=p[2];w=w1;out=o1;if(g.x>=n1){g.x-=n1;N=p[3];w=w2;out=o2;}}
    mlx_affine_tile<256,8,32,true>(w,x,out,p[0],N,p[4],g,tid,tile);
}
#endif

// Split contractions preserve the BF16 partial-result boundary used by the
// native checkpoint reference. The caller reserves a disjoint workspace tail.
template<uint Padding=0,uint Values=8>
inline void mlx_affine_partial(device const uchar* w,device bfloat* scratch,
    constant uint* p,uint3 g,uint tid,threadgroup bfloat* weights) {
    constexpr uint BK=64,BM=32,BN=32;
    uint K=p[0],N=p[1],M=p[2],span=K/p[3],n=g.x*BN,m=g.y*BM;
    uint pitch=(K+127)/128*128,rows=(M+127)/128*128;
    auto a=tensor(scratch,dextents<int,2>(pitch,rows),array<int,2>{1,int(pitch)});
    auto b=tensor(weights,extents<int,BK,BN>(),array<int,2>{1,BK+Padding});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto sum=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<sum.get_capacity();++i)sum[i]=0;
    for(uint base=g.z*span;base<(g.z+1)*span;base+=BK) {
        for(uint i=tid*Values;i<BK*BN;i+=128*Values) {
            uint col=n+i/BK,k=base+i%BK;
            uint at=(i/BK)*(BK+Padding)+i%BK;
            if(col<N) {
                #pragma unroll
                for(uint j=0;j<Values;j+=8)mlx_affine_stage8(w,K,N,ulong(col)*K+k+j,weights+at+j);
            } else {for(uint j=0;j<Values;++j)weights[at+j]=bfloat(0);}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto at=a.slice<BK,BM>(base,m);op.run(at,b,sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=sum.begin();it!=sum.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && n+uint(ij[0])<N && m+uint(ij[1])<M)
            scratch[ulong(p[4])+ulong(g.z)*M*N+ulong(m+ij[1])*N+n+ij[0]]=bfloat(*it);
    }
}
kernel void mlx_affine_parts(device const uchar* w [[buffer(0)]],device bfloat* scratch [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat weights[64*32];
    mlx_affine_partial(w,scratch,p,g,tid,weights);
}
kernel void mlx_affine_parts_padded(device const uchar* w [[buffer(0)]],device bfloat* scratch [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat weights[72*32];
    mlx_affine_partial<8,16>(w,scratch,p,g,tid,weights);
}
kernel void mlx_affine_join(device const bfloat* scratch [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint count=p[1]*p[2],parts=p[3];
    // The reference reduction has BF16 running totals, not just BF16 input
    // and output. Widening the complete join silently changed narrow alpha/
    // beta projections, especially at 256+ prefill rows. Preserve its two
    // deterministic reduction trees without copying its generic reducer.
    if(parts>=32) {
        uint column=i/32,lane=i%32;
        if(column>=count)return;
        bfloat sum=bfloat(0);
        for(uint part=lane;part<parts;part+=32)
            sum=bfloat(float(sum)+float(scratch[ulong(p[4])+ulong(part)*count+column]));
        bfloat result=bfloat(simd_sum(float(sum)));
        if(lane==0)out[column]=float(result);
    } else {
        if(i>=count)return;
        uint lanes=min(parts,8u);float total=0;
        for(uint lane=0;lane<lanes;++lane) {
            float subtotal=0;
            for(uint part=lane;part<parts;part+=lanes)
                subtotal=mlx_bf(subtotal+float(scratch[ulong(p[4])+ulong(part)*count+i]));
            total=mlx_bf(total+subtotal);
        }
        out[i]=total;
    }
}
