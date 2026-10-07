// Original small-row kernels, motivated by llama.cpp #29869's finding
// that verification needs its own contraction shape even on M5. No upstream
// kernel code is used. The packed-loader path preserves our FMA tree;
// alternative matrix/lane mappings are test-only, not production elections.
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
template<uint R,uint C>
inline void mlx_few_vector(device const uchar* w,device const bfloat* x,device float* out,
    uint K,uint N,uint M,uint2 group,uint tid) {
    uint column=group.x*(8*C)+(tid/8)*C,lane=tid%8,first=group.y*R;
    if(column>=N)return;
    device const uint* codes=reinterpret_cast<device const uint*>(w);
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float sums[R][C];for(uint r=0;r<R;++r)for(uint c=0;c<C;++c)sums[r][c]=0;
    for(uint base=lane*64;base<K;base+=512) {
        float s[C],b[C];
        for(uint c=0;c<C;++c) {
            ulong at=ulong(min(column+c,N-1))*(K/64)+base/64;
            s[c]=float(scales[at]);b[c]=float(biases[at]);
        }
        #pragma unroll
        for(uint word=0;word<8;++word) {
            float4 lo[C],hi[C];
            for(uint c=0;c<C;++c) {
                uint bits=codes[ulong(min(column+c,N-1))*(K/8)+base/8+word];
                lo[c]=fma(float4((uint4(bits)>>uint4(0,4,8,12))&15),float4(s[c]),float4(b[c]));
                hi[c]=fma(float4((uint4(bits)>>uint4(16,20,24,28))&15),float4(s[c]),float4(b[c]));
            }
            #pragma unroll
            for(uint r=0;r<R;++r) {
                device const bfloat* at=x+ulong(min(first+r,M-1))*K+base+word*8;
                float4 a=float4(*reinterpret_cast<device const vec<bfloat,4>*>(at));
                float4 z=float4(*reinterpret_cast<device const vec<bfloat,4>*>(at+4));
                for(uint c=0;c<C;++c)sums[r][c]=mlx_affine_contract8(a,z,lo[c],hi[c],sums[r][c]);
            }
        }
    }
    for(uint r=0;r<R;++r)for(uint c=0;c<C;++c) {
        float value=kquant_sum<8>(sums[r][c]);
        if(lane==0 && first+r<M && column+c<N)out[ulong(first+r)*N+column+c]=mlx_bf(value);
    }
}
#define MLX_FEW_VECTOR(R,C) \
kernel void mlx_few_r##R##c##C(device const uchar* w [[buffer(0)]],device const bfloat* x [[buffer(1)]], \
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
mlx_few_vector<R,C>(w,x,out,p[0],p[1],p[2],group,tid); }
MLX_FEW_VECTOR(2,2)
MLX_FEW_VECTOR(3,2)
MLX_FEW_VECTOR(4,2)
MLX_FEW_VECTOR(5,2)
MLX_FEW_VECTOR(8,1)
MLX_FEW_VECTOR(8,2)
#undef MLX_FEW_VECTOR

// Fold the eight logical reduction lanes onto fewer physical lanes. Keep
// independent accumulators for the displaced lanes, then perform exactly
// their original 4,2,1 reduction. This exposes independent FMA chains without
// changing a single eight-product word or the group-scale rounding.
template<uint R,uint L>
inline void mlx_few_folded(device const uchar* w,device const bfloat* x,device float* out,
    uint K,uint N,uint M,uint2 group,uint tid) {
    constexpr uint P=8/L;
    uint column=group.x*(64/L)+tid/L,lane=tid%L,first=group.y*R;
    if(column>=N)return;
    device const uint* codes=reinterpret_cast<device const uint*>(w)+ulong(column)*(K/8);
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2)+ulong(column)*(K/64);
    device const bfloat* biases=scales+ulong(K)*N/64;
    float sums[P][R];for(uint p=0;p<P;++p)for(uint r=0;r<R;++r)sums[p][r]=0;
    for(uint base=0;base<K;base+=512) {
        float s[P],b[P];
        for(uint p=0;p<P;++p) {uint k=base+(lane+p*L)*64;s[p]=float(scales[k/64]);b[p]=float(biases[k/64]);}
        #pragma unroll
        for(uint word=0;word<8;++word) {
            #pragma unroll
            for(uint p=0;p<P;++p) {
                uint k=base+(lane+p*L)*64+word*8,bits=codes[k/8];
                float4 lo=fma(float4((uint4(bits)>>uint4(0,4,8,12))&15),float4(s[p]),float4(b[p]));
                float4 hi=fma(float4((uint4(bits)>>uint4(16,20,24,28))&15),float4(s[p]),float4(b[p]));
                #pragma unroll
                for(uint r=0;r<R;++r) {
                    device const bfloat* at=x+ulong(min(first+r,M-1))*K+k;
                    float4 a=float4(*reinterpret_cast<device const vec<bfloat,4>*>(at));
                    float4 z=float4(*reinterpret_cast<device const vec<bfloat,4>*>(at+4));
                    sums[p][r]=mlx_affine_contract8(a,z,lo,hi,sums[p][r]);
                }
            }
        }
    }
    for(uint r=0;r<R;++r) {
        for(uint step=P/2;step>0;step/=2)for(uint p=0;p<step;++p)sums[p][r]+=sums[p+step][r];
        float v=kquant_sum<L>(sums[0][r]);
        if(lane==0 && first+r<M)out[ulong(first+r)*N+column]=mlx_bf(v);
    }
}
#define MLX_FEW_FOLD(R,L) \
kernel void mlx_few_fold##R##l##L(device const uchar* w [[buffer(0)]],device const bfloat* x [[buffer(1)]], \
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
mlx_few_folded<R,L>(w,x,out,p[0],p[1],p[2],group,tid); }
MLX_FEW_FOLD(2,4)
MLX_FEW_FOLD(3,4)
MLX_FEW_FOLD(4,4)
MLX_FEW_FOLD(5,4)
MLX_FEW_FOLD(2,2)
MLX_FEW_FOLD(3,2)
MLX_FEW_FOLD(4,2)
MLX_FEW_FOLD(5,2)
#undef MLX_FEW_FOLD
#endif

// Coalesce each lane's complete affine group into two 128-bit code loads.
// Activations are loaded as one 128-bit BF16 word; the original contraction
// helper still owns every FMA and rounding boundary.
template<uint R,uint C=8,uint StaticK=0,bool Share=false>
inline void mlx_few_packed(device const uchar* w,device const bfloat* x,device float* out,
    uint runtimeK,uint N,uint M,uint2 group,uint tid) {
    const uint K=StaticK?StaticK:runtimeK;
    uint column=group.x*C+tid/8,lane=tid%8,first=group.y*R;
    if(column>=N)return;
    device const uint* codes=reinterpret_cast<device const uint*>(w)+ulong(column)*(K/8);
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2)+ulong(column)*(K/64);
    device const bfloat* biases=scales+ulong(K)*N/64;
    device const bfloat* input[R];float sum[R];
    for(uint r=0;r<R;++r) {input[r]=x+ulong(min(first+r,M-1))*K;sum[r]=0;}
    for(uint base=lane*64;base<K;base+=512) {
        uint4 bits0=*reinterpret_cast<device const uint4*>(codes+base/8);
        uint4 bits1=*reinterpret_cast<device const uint4*>(codes+base/8+4);
        float s=float(scales[base/64]),b=float(biases[base/64]);
        #pragma unroll
        for(uint word=0;word<8;++word) {
            uint bits=word<4?bits0[word%4]:bits1[word%4];
            float4 lo=fma(float4((uint4(bits)>>uint4(0,4,8,12))&15),float4(s),float4(b));
            float4 hi=fma(float4((uint4(bits)>>uint4(16,20,24,28))&15),float4(s),float4(b));
            #pragma unroll
            for(uint r=0;r<R;++r) {
                uint4 raw;
                if constexpr(Share) {
                    raw=uint4(0);
                    if(tid%32<8)raw=*reinterpret_cast<device const uint4*>(input[r]+base+word*8);
                    raw=simd_shuffle(raw,ushort(lane));
                } else {
                    raw=*reinterpret_cast<device const uint4*>(input[r]+base+word*8);
                }
                float4 a=float4(as_type<vec<bfloat,4>>(raw.xy));
                float4 z=float4(as_type<vec<bfloat,4>>(raw.zw));
                sum[r]=mlx_affine_contract8(a,z,lo,hi,sum[r]);
            }
        }
    }
    for(uint r=0;r<R;++r) {
        float v=kquant_sum<8>(sum[r]);
        if(lane==0 && first+r<M)out[ulong(first+r)*N+column]=mlx_bf(v);
    }
}
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
#define MLX_FEW_PACKED(R) \
kernel void mlx_few_packed##R(device const uchar* w [[buffer(0)]],device const bfloat* x [[buffer(1)]], \
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
mlx_few_packed<R>(w,x,out,p[0],p[1],p[2],group,tid); }
MLX_FEW_PACKED(1)
MLX_FEW_PACKED(2)
MLX_FEW_PACKED(3)
MLX_FEW_PACKED(4)
MLX_FEW_PACKED(5)
MLX_FEW_PACKED(8)
#undef MLX_FEW_PACKED

// Occupancy probe: vary independent columns/workgroup without changing the
// eight logical reduction lanes or their order. Static K is a separate
// experiment, not an assumption that compiler unrolling will be faster.
#define MLX_FEW_SHAPE(R,C,K) \
kernel void mlx_few_shape_r##R##c##C##k##K(device const uchar* w [[buffer(0)]],device const bfloat* x [[buffer(1)]], \
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
mlx_few_packed<R,C,K>(w,x,out,p[0],p[1],p[2],group,tid); }
#define MLX_FEW_SHAPES(R) \
MLX_FEW_SHAPE(R,4,0) \
MLX_FEW_SHAPE(R,16,0) \
MLX_FEW_SHAPE(R,32,0) \
MLX_FEW_SHAPE(R,8,5120) \
MLX_FEW_SHAPE(R,8,6144) \
MLX_FEW_SHAPE(R,8,17408)
MLX_FEW_SHAPES(2)
MLX_FEW_SHAPES(3)
MLX_FEW_SHAPES(4)
MLX_FEW_SHAPES(5)
#define MLX_FEW_SHARED(R) \
kernel void mlx_few_shared##R(device const uchar* w [[buffer(0)]],device const bfloat* x [[buffer(1)]], \
device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
mlx_few_packed<R,8,0,true>(w,x,out,p[0],p[1],p[2],group,tid); }
MLX_FEW_SHARED(2)
MLX_FEW_SHARED(3)
MLX_FEW_SHARED(4)
MLX_FEW_SHARED(5)
#undef MLX_FEW_SHARED
#undef MLX_FEW_SHAPES
#undef MLX_FEW_SHAPE
#endif

#define MLX_FEW_FUSED(R) \
kernel void mlx_affine_packed##R(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],device const uchar* w2 [[buffer(2)]], \
device const bfloat* x [[buffer(3)]],device float* o0 [[buffer(4)]],device float* o1 [[buffer(5)]],device float* o2 [[buffer(6)]], \
constant uint* p [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
uint n0=(p[1]+7)/8,n1=(p[2]+7)/8,N=p[1];device const uchar* w=w0;device float* out=o0; \
if(group.x>=n0){group.x-=n0;N=p[2];w=w1;out=o1;if(group.x>=n1){group.x-=n1;N=p[3];w=w2;out=o2;}} \
mlx_few_packed<R>(w,x,out,p[0],N,p[4],group,tid); }
MLX_FEW_FUSED(3)
MLX_FEW_FUSED(4)
MLX_FEW_FUSED(5)
#undef MLX_FEW_FUSED

#ifdef PADDOCK_KERNEL_DIAGNOSTICS

// Bounded F32 matrix candidate: four independent K ranges, shared 8x64
// activation/weight tiles and one final reduction. BF16 source activations,
// F32 dequantization/accumulation. Its reduction is intentionally NOT assumed
// equivalent to the fixed vector tree; the GPU oracle reports every mismatch.
kernel void mlx_few_mma(device const uchar* w [[buffer(0)]],device const bfloat* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float atile[4*512],btile[4*512],partial[4*64];
    uint K=p[0],N=p[1],M=p[2],sg=tid/32,lane=tid%32;
    device const uint* codes=reinterpret_cast<device const uint*>(w);
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    device const bfloat* biases=scales+ulong(K)*N/64;
    auto acc=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint base=0;base<K;base+=256) {
        for(uint i=lane;i<512;i+=32) {
            uint r=group.y*8+i/64,c=group.x*8+i/64,k=base+sg*64+i%64;
            atile[sg*512+i]=(r<M && k<K)?float(x[ulong(r)*K+k]):0.0f;
            float v=0;
            if(c<N && k<K) {
                ulong index=ulong(c)*K+k;
                uint q=(codes[index/8]>>((k%8)*4))&15;
                v=fma(float(q),float(scales[index/64]),float(biases[index/64]));
            }
            btile[sg*512+i]=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<64;k+=8) {
            simdgroup_float8x8 a,b;
            simdgroup_load(a,atile+sg*512+k,64);
            simdgroup_load(b,btile+sg*512+k,64,ulong2(0),true);
            simdgroup_multiply_accumulate(acc,a,b,acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,partial+sg*64,8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<64) {
        uint r=group.y*8+tid/8,c=group.x*8+tid%8;
        float sum=(partial[tid]+partial[64+tid])+(partial[128+tid]+partial[192+tid]);
        if(r<M && c<N)out[ulong(r)*N+c]=mlx_bf(sum);
    }
}
// Register-fed matrix experiment. Each SIMD handles the same 64-weight
// partition as one baseline vector lane. Separate eight-product matrices
// retain word boundaries; whether hardware retains the scalar FMA chain is
// established by the oracle, not assumed. Lane coordinates are the Apple
// 8x8 fragment layout (hardware ABI), not an imported dequantization kernel.
kernel void mlx_few_register(device const uchar* w [[buffer(0)]],device const bfloat* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8*64];
    uint K=p[0],N=p[1],M=p[2],sg=tid/32,lane=tid%32;
    uint rr=((lane&16)>>2)|((lane&6)>>1),cc=((lane&8)>>1)|((lane&1)<<1);
    uint query=group.y*8+rr,c0=group.x*8+cc,c1=c0+1;
    device const uint* codes=reinterpret_cast<device const uint*>(w);
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    device const bfloat* biases=scales+ulong(K)*N/64;
    auto acc=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint base=sg*64;base<K;base+=512) {
        ulong i0=ulong(min(c0,N-1))*(K/64)+base/64,i1=ulong(min(c1,N-1))*(K/64)+base/64;
        float s0=float(scales[i0]),s1=float(scales[i1]),b0=float(biases[i0]),b1=float(biases[i1]);
        for(uint word=0;word<8;++word) {
            uint k=base+word*8;
            simdgroup_float8x8 a,b;
            a.thread_elements()[0]=query<M?float(x[ulong(query)*K+k+cc]):0.0f;
            a.thread_elements()[1]=query<M?float(x[ulong(query)*K+k+cc+1]):0.0f;
            uint q0=codes[ulong(min(c0,N-1))*(K/8)+k/8],q1=codes[ulong(min(c1,N-1))*(K/8)+k/8];
            b.thread_elements()[0]=fma(float((q0>>(rr*4))&15),s0,b0);
            b.thread_elements()[1]=fma(float((q1>>(rr*4))&15),s1,b1);
            auto v=make_filled_simdgroup_matrix<float,8>(0.0f);
            simdgroup_multiply_accumulate(v,a,b,v);
            acc.thread_elements()[0]+=v.thread_elements()[0];
            acc.thread_elements()[1]+=v.thread_elements()[1];
        }
    }
    simdgroup_store(acc,partial+sg*64,8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<64) {
        // Same xor 4,2,1 reduction tree as kquant_sum<8>.
        float a=(partial[tid]+partial[256+tid])+(partial[128+tid]+partial[384+tid]);
        float b=(partial[64+tid]+partial[320+tid])+(partial[192+tid]+partial[448+tid]);
        uint r=group.y*8+tid/8,c=group.x*8+tid%8;
        if(r<M && c<N)out[ulong(r)*N+c]=mlx_bf(a+b);
    }
}
#endif
