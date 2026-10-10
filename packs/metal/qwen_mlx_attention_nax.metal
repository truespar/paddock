// Paged head-256 contraction adapted from MLX v0.32.3's
// steel_attention_nax.h (head-dimension split). Copyright © 2024–2026
// Apple Inc., MIT; see qwen_mlx_vision_nax.NOTICE.md. Native paged addressing,
// bounded scheduler tiles and F32 output; no MLX runtime.
#ifndef PADDOCK_APPLE9
#define QMLX_UNROLL _Pragma("clang loop unroll(full)")
[[kernel, max_total_threads_per_threadgroup(128)]] void mlx_attention_prefill_nax(
    device const bfloat* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],
    device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],
    device const uint* tiles [[buffer(6)]],device const uint* limits [[buffer(7)]],constant uint* p [[buffer(8)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint first=tiles[2*g.y],count=tiles[2*g.y+1],slot=meta[2*first];
    uint lane=tid%32,sg=tid/32,row_group=sg/2,part=sg%2;
    uint local=row_group*16,valid=count>local?min(16u,count-local):0;
    uint r=(lane&16)/4+(lane&6)/2,c=(lane&8)+(lane&1)*4;
    uint heads=p[0],kvheads=p[1],kh=g.x/(heads/kvheads),stride=kvheads*256,last=limits[first+count-1];
    threadgroup float exchange[4*32*16];
    vec<bfloat,8> queries[8];
    QMLX_UNROLL for(short d=0;d<8;++d)
        queries[d]=qmlx_nax::load<false>(q+(ulong(first+(valid?local:0))*heads+g.x)*256+part*128+d*16,heads*256,valid,lane);
    vec<float,8> acc[8];QMLX_UNROLL for(short d=0;d<8;++d)acc[d]=vec<float,8>(0);
    float2 hi(-INFINITY),den(0);
    for(uint base=0;base<=last;base+=32) {
        uint n0=min(16u,last-base+1),n1=last>=base+16?min(16u,last-base-15):0;
        ulong at0=ulong(pages[slot*p[2]+base/16]*16)*stride+kh*256+part*128;
        ulong at1=n1?ulong(pages[slot*p[2]+base/16+1]*16)*stride+kh*256+part*128:at0;
        vec<float,8> score[2]={vec<float,8>(0),vec<float,8>(0)};
        QMLX_UNROLL for(short d=0;d<8;++d) {
            auto k0=qmlx_nax::load<false>(k+at0+d*16,stride,n0,lane);
            auto k1=qmlx_nax::load<false>(k+at1+d*16,stride,n1,lane);
            qmlx_nax::mma<bfloat,true>(score[0],score[1],queries[d],k0,k1);
        }
        QMLX_UNROLL for(short f=0;f<2;++f)QMLX_UNROLL for(short i=0;i<8;++i)
            exchange[(sg*32+lane)*16+f*8+i]=score[f][i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        QMLX_UNROLL for(short f=0;f<2;++f)QMLX_UNROLL for(short i=0;i<8;++i)
            score[f][i]=exchange[(row_group*2*32+lane)*16+f*8+i]+exchange[((row_group*2+1)*32+lane)*16+f*8+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 nh=hi;
        QMLX_UNROLL for(short f=0;f<2;++f)QMLX_UNROLL for(short i=0;i<2;++i) {
            uint qr=local+r+i*8;
            uint limit=qr<count?limits[first+qr]:0;
            QMLX_UNROLL for(short j=0;j<4;++j)
                score[f][i*4+j]=qr<count && base+f*16+c+j<=limit?
                    score[f][i*4+j]*(as_type<float>(p[3])*1.44269504089f):-INFINITY;
            float m=max(max(score[f][i*4],score[f][i*4+1]),max(score[f][i*4+2],score[f][i*4+3]));
            m=max(m,simd_shuffle_xor(m,ushort(1)));m=max(m,simd_shuffle_xor(m,ushort(8)));
            nh[i]=max(nh[i],m);
        }
        float2 factor;
        QMLX_UNROLL for(short i=0;i<2;++i) {
            // Inactive query lanes still participate in the exchange barriers.
            if(!isfinite(nh[i]))nh[i]=0;
            factor[i]=fast::exp2(hi[i]-nh[i]);hi[i]=nh[i];den[i]*=factor[i];
        }
        QMLX_UNROLL for(short f=0;f<2;++f)QMLX_UNROLL for(short i=0;i<2;++i) {
            QMLX_UNROLL for(short j=0;j<4;++j)score[f][i*4+j]=fast::exp2(score[f][i*4+j]-hi[i]);
            float sum=(score[f][i*4]+score[f][i*4+1])+(score[f][i*4+2]+score[f][i*4+3]);
            sum+=simd_shuffle_xor(sum,ushort(1));sum+=simd_shuffle_xor(sum,ushort(8));den[i]+=sum;
        }
        QMLX_UNROLL for(short d=0;d<8;++d)QMLX_UNROLL for(short i=0;i<8;++i)acc[d][i]*=factor[i/4];
        simdgroup_barrier(mem_flags::mem_none);
        QMLX_UNROLL for(short d=0;d<8;d+=2)QMLX_UNROLL for(short f=0;f<2;++f) {
            ulong at=f?at1:at0;uint n=f?n1:n0;
            auto v0=qmlx_nax::load<false>(v+at+d*16,stride,n,lane);
            auto v1=qmlx_nax::load<false>(v+at+d*16+16,stride,n,lane);
            qmlx_nax::mma<float,false>(acc[d],acc[d+1],score[f],v0,v1);
        }
    }
    // Every active causal row includes itself, so its softmax denominator is
    // at least one. Preserve the reference's direct division; inactive rows
    // never store. A redundant denominator clamp also miscompiled under GPU
    // instrumentation on the tested Apple10 toolchain (ragged-page gate).
    float2 inv=precise::divide(float2(1),den);
    QMLX_UNROLL for(short d=0;d<8;++d)QMLX_UNROLL for(short i=0;i<2;++i)if(r+uint(i)*8<valid)
        QMLX_UNROLL for(short j=0;j<4;++j)
            out[(ulong(first+local+r+i*8)*heads+g.x)*256+part*128+d*16+c+j]=mlx_bf(acc[d][i*4+j]*inv[i]);
}
#undef QMLX_UNROLL
#endif
