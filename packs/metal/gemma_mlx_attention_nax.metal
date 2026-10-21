// Gemma paged/ring attention using the existing MLX-derived NAX fragment
// helpers. Copyright © 2024–2026 Apple Inc., MIT; see
// qwen_mlx_vision_nax.NOTICE.md. No MLX runtime or persistent cache expansion.
// M5 dense Gemma 31B text prefill. Vision, speculation, MoE and diffusion
// retain their existing paths. Query/KV storage is BF16; online-softmax
// state and accumulators are F32. Three-component probability contraction
// compensates BF16 matrix arithmetic; it is not bit-identical to the old
// F32 contraction. Tests compare to an independent F64 oracle and oMLX.
#ifndef PADDOCK_APPLE9
#define GMLX_UNROLL _Pragma("clang loop unroll(full)")
// Do not read stale ring entries outside the oldest query's window: even a
// masked probability of zero cannot neutralize a NaN in the value cache.
inline vec<bfloat,8> gmlx_nax_load(device const bfloat* source,uint stride,
    uint valid,uint skip,uint lane) {
    uint row=(lane&16)/4+(lane&6)/2,col=(lane&8)+(lane&1)*4;
    vec<bfloat,8> result;
    GMLX_UNROLL for(short i=0;i<2;++i)GMLX_UNROLL for(short j=0;j<4;++j) {
        uint r=row+uint(i)*8;
        result[i*4+j]=r>=skip && r<valid?source[r*stride+col+j]:bfloat(0);
    }
    return result;
}
// Unlike the shared relaxed helper, retain full F32 probabilities here.
inline void gmlx_nax_pv(thread vec<float,8>& c0,thread vec<float,8>& c1,
    thread const vec<float,8>& a,thread const vec<bfloat,8>& b0,thread const vec<bfloat,8>& b1) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    // Three residual BF16 components retain F32 probability significance
    // without the strict mixed-operand fragment layout's scalar shuffle cost.
    vec<bfloat,8> hi,mid,lo;
    GMLX_UNROLL for(short i=0;i<8;++i) {
        hi[i]=bfloat(a[i]);float residual=a[i]-float(hi[i]);
        mid[i]=bfloat(residual);lo[i]=bfloat(residual-float(mid[i]));
    }
    qmlx_nax::mma<bfloat,false>(c0,c1,lo,b0,b1);
    qmlx_nax::mma<bfloat,false>(c0,c1,mid,b0,b1);
    qmlx_nax::mma<bfloat,false>(c0,c1,hi,b0,b1);
}
template<uint HD>
inline void gmlx_nax_attention(device const bfloat* q,device const bfloat* k,device const bfloat* v,
    device const uint* meta,device const uint* pages,device float* out,device const uint* tiles,
    constant uint* p,uint3 g,uint tid,threadgroup float* exchange) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    constexpr uint Parts=HD/128,QueryRows=64/Parts;
    uint original_count=tiles[2*g.y+1],shift=g.z*QueryRows;
    if(shift>=original_count)return;
    uint first=tiles[2*g.y]+shift,count=min(QueryRows,original_count-shift),slot=meta[2*first];
    uint lane=tid%32,sg=tid/32,row_group=sg/Parts,part=sg%Parts;
    uint local=row_group*16,valid=count>local?min(16u,count-local):0;
    uint r=(lane&16)/4+(lane&6)/2,c=(lane&8)+(lane&1)*4;
    uint heads=p[0],kvheads=p[1],kh=g.x/(heads/kvheads),stride=kvheads*HD,last=meta[2*(first+count-1)+1];
    uint firstpos=meta[2*first+1],low=p[3] && firstpos+1>p[3]?firstpos+1-p[3]:0;
    vec<bfloat,8> queries[8];
    GMLX_UNROLL for(short d=0;d<8;++d)
        queries[d]=qmlx_nax::load<false>(q+(ulong(first+(valid?local:0))*heads+g.x)*HD+part*128+d*16,heads*HD,valid,lane);
    vec<float,8> acc[8];GMLX_UNROLL for(short d=0;d<8;++d)acc[d]=vec<float,8>(0);
    float2 hi(-INFINITY),den(0);
    for(uint base=low/32*32;base<=last;base+=32) {
        uint n0=min(16u,last-base+1),n1=last>=base+16?min(16u,last-base-15):0;
        uint skip0=low>base?min(16u,low-base):0;
        uint skip1=low>base+16?min(16u,low-base-16):0;
        ulong at0=ulong(gemma_physical(pages,slot,base,p))*stride+kh*HD+part*128;
        ulong at1=n1?ulong(gemma_physical(pages,slot,base+16,p))*stride+kh*HD+part*128:at0;
        vec<float,8> score[2]={vec<float,8>(0),vec<float,8>(0)};
        GMLX_UNROLL for(short d=0;d<8;++d) {
            auto k0=gmlx_nax_load(k+at0+d*16,stride,n0,skip0,lane),k1=gmlx_nax_load(k+at1+d*16,stride,n1,skip1,lane);
            qmlx_nax::mma<bfloat,true>(score[0],score[1],queries[d],k0,k1);
        }
        GMLX_UNROLL for(short f=0;f<2;++f)GMLX_UNROLL for(short i=0;i<8;++i)
            exchange[(sg*32+lane)*16+f*8+i]=score[f][i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        GMLX_UNROLL for(short f=0;f<2;++f)GMLX_UNROLL for(short i=0;i<8;++i) {
            float sum=exchange[(row_group*Parts*32+lane)*16+f*8+i];
            GMLX_UNROLL for(short s=1;s<Parts;++s)sum+=exchange[((row_group*Parts+s)*32+lane)*16+f*8+i];
            score[f][i]=sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 nh=hi;
        GMLX_UNROLL for(short f=0;f<2;++f)GMLX_UNROLL for(short i=0;i<2;++i) {
            uint qr=local+r+i*8,pos=qr<count?meta[2*(first+qr)+1]:0,lo=p[3] && pos+1>p[3]?pos+1-p[3]:0;
            GMLX_UNROLL for(short j=0;j<4;++j) {
                uint key=base+f*16+c+j;
                score[f][i*4+j]=qr<count && key<=pos && key>=lo?score[f][i*4+j]*1.44269504089f:-INFINITY;
            }
            float m=max(max(score[f][i*4],score[f][i*4+1]),max(score[f][i*4+2],score[f][i*4+3]));
            m=max(m,simd_shuffle_xor(m,ushort(1)));m=max(m,simd_shuffle_xor(m,ushort(8)));nh[i]=max(nh[i],m);
        }
        float2 factor;
        GMLX_UNROLL for(short i=0;i<2;++i) {
            factor[i]=isfinite(hi[i])?fast::exp2(hi[i]-nh[i]):0;
            hi[i]=nh[i];den[i]*=factor[i];
        }
        GMLX_UNROLL for(short f=0;f<2;++f)GMLX_UNROLL for(short i=0;i<2;++i) {
            GMLX_UNROLL for(short j=0;j<4;++j)score[f][i*4+j]=isfinite(hi[i])?fast::exp2(score[f][i*4+j]-hi[i]):0;
            float sum=(score[f][i*4]+score[f][i*4+1])+(score[f][i*4+2]+score[f][i*4+3]);
            sum+=simd_shuffle_xor(sum,ushort(1));sum+=simd_shuffle_xor(sum,ushort(8));den[i]+=sum;
        }
        GMLX_UNROLL for(short d=0;d<8;++d)GMLX_UNROLL for(short i=0;i<8;++i)acc[d][i]*=factor[i/4];
        simdgroup_barrier(mem_flags::mem_none);
        GMLX_UNROLL for(short d=0;d<8;d+=2)GMLX_UNROLL for(short f=0;f<2;++f) {
            ulong at=f?at1:at0;uint n=f?n1:n0,skip=f?skip1:skip0;
            auto v0=gmlx_nax_load(v+at+d*16,stride,n,skip,lane),v1=gmlx_nax_load(v+at+d*16+16,stride,n,skip,lane);
            gmlx_nax_pv(acc[d],acc[d+1],score[f],v0,v1);
        }
    }
    float2 inv=precise::divide(float2(1),den);
    GMLX_UNROLL for(short d=0;d<8;++d)GMLX_UNROLL for(short i=0;i<2;++i)if(r+uint(i)*8<valid)
        GMLX_UNROLL for(short j=0;j<4;++j)
            out[(ulong(first+local+r+i*8)*heads+g.x)*HD+part*128+d*16+c+j]=mlx_bf(acc[d][i*4+j]*inv[i]);
}
#define GMLX_NAX(HD) \
[[kernel, max_total_threads_per_threadgroup(128)]] void gmlx_nax_prefill##HD(device const bfloat* q [[buffer(0)]],device const bfloat* k [[buffer(1)]],device const bfloat* v [[buffer(2)]],device const uint* meta [[buffer(3)]],device const uint* pages [[buffer(4)]],device float* out [[buffer(5)]],device const uint* tiles [[buffer(6)]],constant uint* p [[buffer(7)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float exchange[4*32*16];gmlx_nax_attention<HD>(q,k,v,meta,pages,out,tiles,p,g,tid,exchange);}
GMLX_NAX(256)
GMLX_NAX(512)
#undef GMLX_NAX
#undef GMLX_UNROLL
#endif
