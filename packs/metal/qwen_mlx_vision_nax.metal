// M5-only register fragments adapted from MLX v0.32.3's steel/attn/nax.h
// and steel_attention_nax.h. Copyright © 2024–2026 Apple Inc. MIT license;
// see qwen_mlx_vision_nax.NOTICE.md. No MLX runtime or weight dependency.
// The register mapping is Apple10-specific. Earlier GPUs use the bounded MPP
// implementation, whose cooperative tensors choose their own mapping.
#ifndef PADDOCK_APPLE9
namespace qmlx_nax {
#define QMLX_UNROLL _Pragma("clang loop unroll(full)")

template<bool Aligned>
__attribute__((always_inline)) inline vec<bfloat,8> load(
    device const bfloat* src,uint stride,uint valid,uint lane) {
    uint r=(lane&16)/4+(lane&6)/2,c=(lane&8)+(lane&1)*4;
    vec<bfloat,8> values;
    QMLX_UNROLL for(short i=0;i<2;++i) {
        QMLX_UNROLL for(short j=0;j<4;++j) {
            values[i*4+j]=Aligned || r+uint(i)*8<valid ? src[(r+i*8)*stride+c+j] : bfloat(0);
        }
    }
    return values;
}

template<typename A,bool Transpose>
__attribute__((always_inline)) inline void mma(thread vec<float,8>& c0,thread vec<float,8>& c1,
    thread const vec<A,8>& a,thread const vec<bfloat,8>& b0,thread const vec<bfloat,8>& b1) {
    constexpr auto desc=matmul2d_descriptor(16,32,16,false,Transpose,true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroup> op;
    auto ca=op.template get_left_input_cooperative_tensor<A,bfloat,float>();
    auto cb=op.template get_right_input_cooperative_tensor<A,bfloat,float>();
    auto cc=op.template get_destination_cooperative_tensor<decltype(ca),decltype(cb),float>();
    QMLX_UNROLL for(short i=0;i<8;++i){ca[i]=a[i];cb[i]=b0[i];cb[i+8]=b1[i];cc[i]=c0[i];cc[i+8]=c1[i];}
    op.run(ca,cb,cc);
    QMLX_UNROLL for(short i=0;i<8;++i){c0[i]=cc[i];c1[i]=cc[i+8];}
}

template<uint Heads,bool Aligned>
__attribute__((always_inline)) inline void attention(device const bfloat* q,device const bfloat* k,
    device const bfloat* v,device float* out,uint4 tile,uint head,uint tid) {
    // MLX's SDPA preserves the scalar softmax update's multiply/add cuts.
    // Contracting den*factor + sum changes BF16 ties; those small errors grow
    // through the tower. Keep this local: MPP contractions still use tensor
    // hardware and other model families retain their qualified math modes.
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint first=tid/32*16,lane=tid%32;
    if(first>=tile.y)return;
    uint count=min(16u,tile.y-first),stride=Heads*64;
    uint r=(lane&16)/4+(lane&6)/2,c=(lane&8)+(lane&1)*4;
    q+=(ulong(tile.x+first)*Heads+head)*64;
    k+=(ulong(tile.z)*Heads+head)*64;v+=(ulong(tile.z)*Heads+head)*64;
    vec<float,8> acc[4];
    QMLX_UNROLL for(short i=0;i<4;++i)acc[i]=vec<float,8>(0);
    float2 hi(-INFINITY),den(0);
    for(uint base=0;base<tile.w;base+=32) {
        vec<float,8> score[2]={vec<float,8>(0),vec<float,8>(0)};
        #pragma clang loop unroll_count(4)
        for(short d=0;d<4;++d) {
            auto qr=load<Aligned>(q+d*16,stride,count,lane);
            auto k0=load<Aligned>(k+d*16,stride,min(16u,tile.w-base),lane);
            // Do not form a pointer beyond the input for a final <16-row tile.
            auto k1=load<Aligned>(k+(tile.w-base>16?16:0)*stride+d*16,stride,
                tile.w-base>16?min(16u,tile.w-base-16):0,lane);
            mma<bfloat,true>(score[0],score[1],qr,k0,k1);
        }
        float2 nh=hi;
        QMLX_UNROLL for(short f=0;f<2;++f) {
            QMLX_UNROLL for(short i=0;i<2;++i) {
                QMLX_UNROLL for(short j=0;j<4;++j) {
                    score[f][i*4+j]=Aligned || base+f*16+c+j<tile.w?
                        score[f][i*4+j]*(.125f*1.44269504089f):-INFINITY;
                }
                float m=max(max(score[f][i*4],score[f][i*4+1]),max(score[f][i*4+2],score[f][i*4+3]));
                m=max(m,simd_shuffle_xor(m,ushort(1)));m=max(m,simd_shuffle_xor(m,ushort(8)));
                nh[i]=max(nh[i],m);
            }
        }
        float2 factor;
        QMLX_UNROLL for(short i=0;i<2;++i){factor[i]=fast::exp2(hi[i]-nh[i]);hi[i]=nh[i];den[i]*=factor[i];}
        QMLX_UNROLL for(short f=0;f<2;++f) {
            QMLX_UNROLL for(short i=0;i<2;++i) {
                QMLX_UNROLL for(short j=0;j<4;++j)score[f][i*4+j]=fast::exp2(score[f][i*4+j]-hi[i]);
                float sum=(score[f][i*4]+score[f][i*4+1])+(score[f][i*4+2]+score[f][i*4+3]);
                sum+=simd_shuffle_xor(sum,ushort(1));sum+=simd_shuffle_xor(sum,ushort(8));den[i]+=sum;
            }
        }
        QMLX_UNROLL for(short d=0;d<4;++d) {
            QMLX_UNROLL for(short i=0;i<8;++i)acc[d][i]*=factor[i/4];
        }
        simdgroup_barrier(mem_flags::mem_none);
        QMLX_UNROLL for(short d=0;d<4;d+=2) {
            QMLX_UNROLL for(short f=0;f<2;++f) {
                uint remaining=tile.w-base>uint(f)*16?min(16u,tile.w-base-f*16):0;
                auto vp=v+(remaining?uint(f)*16:0)*stride+d*16;
                auto v0=load<Aligned>(vp,stride,remaining,lane),v1=load<Aligned>(vp+16,stride,remaining,lane);
                mma<float,false>(acc[d],acc[d+1],score[f],v0,v1);
            }
        }
        if(base+32<tile.w){k+=32*stride;v+=32*stride;}
    }
    // Relaxed library math may approximate division even with FP contraction
    // disabled. Preserve this final cut as well, including ragged-row ties.
    float2 inv=precise::divide(float2(1.0f),den);
    QMLX_UNROLL for(short d=0;d<4;++d) {
        QMLX_UNROLL for(short i=0;i<2;++i)if(r+uint(i)*8<count) {
            QMLX_UNROLL for(short j=0;j<4;++j) {
                out[(ulong(tile.x+first+r+i*8)*Heads+head)*64+d*16+c+j]=mlx_bf(acc[d][i*4+j]*inv[i]);
            }
        }
    }
}
#undef QMLX_UNROLL
} // namespace qmlx_nax
#endif
