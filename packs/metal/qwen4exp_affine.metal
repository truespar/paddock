// Original compressed-affine kernels. U32 codes, then BF16 scales, then
// BF16 biases; offsets use 64-bit arithmetic (PLE is larger than 4 GiB).
// This file does not alter the dense-Qwen affine/group64 implementation.
inline float mlx_sigmoid_bf(float x);
kernel void q4a_small(device const bfloat* x [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]) {float v=float(x[i]);y[i]=p[1]==1 ? -exp(v) : p[1]==2 ? 1.0f+v : v;}
}
template<uint Bits,uint Group>
inline float q4a_value_t(device const uchar* w,uint K,uint N,ulong i) {
    constexpr uint pack=32/Bits;
    uint code=(reinterpret_cast<device const uint*>(w)[i/pack]>>((i%pack)*Bits))&((1u<<Bits)-1);
    device const bfloat* s=reinterpret_cast<device const bfloat*>(w+ulong(K)*N*Bits/8);
    return float(code)*float(s[i/Group])+float(s[ulong(K)*N/Group+i/Group]);
}
inline float q4a_value(device const uchar* w,uint K,uint N,ulong i,uint bits,uint group) {
    // Both supported layouts must specialize division/modulo at compile
    // time. Runtime 64-bit division in every decoded value is prohibitive.
    return bits==4 ? q4a_value_t<4,32>(w,K,N,i) : q4a_value_t<8,64>(w,K,N,i);
}
kernel void q4a_gather(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[2])return;
    uint row=ids[i/p[0]];
    y[i]=row<p[1] ? mlx_bf(q4a_value(w,p[0],p[1],ulong(row)*p[0]+i%p[0],p[3],p[4])) : NAN;
}

// Four outputs share a SIMD's activation loads. BF16 bias subtotals are
// observable in MLX's singleton affine contraction; retain that boundary.
template<uint bits,uint group,uint step,uint Outputs=4,uint Post=0>
inline void q4a_vector_step(device const uchar* w,device const float* x,device float* y,
    uint K,uint totalN,uint N,uint col,uint lane,ulong expert,threadgroup float* tile=nullptr) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if(col>=N)return; // uniform within a SIMD; no workgroup barriers here
    float sums[Outputs];for(uint c=0;c<Outputs;++c)sums[c]=0;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*totalN*bits/8);
    device const bfloat* biases=scales+ulong(K)*totalN/group;
    for(uint k=lane*step;k<K;k+=32*step) {
        // Compile-time step keeps activations in registers and lets the
        // compiler unroll the packed-word contractions. Runtime-sized local
        // array indexing was still present after specializing quant layout.
        float a[step];float bias_sum=0;
        #pragma unroll
        for(uint j=0;j<step;j+=4) {
            // Loader guarantees K is group-aligned, and step divides the
            // group. Every lane's visited step is complete, with no K tail.
            float4 av=*reinterpret_cast<device const float4*>(x+k+j);
            a[j]=av.x;a[j+1]=av.y;a[j+2]=av.z;a[j+3]=av.w;
            if(bits==4) {
                bias_sum+=mlx_bf(mlx_bf(mlx_bf(a[j]+a[j+1])+a[j+2])+a[j+3]);
                // Compensate fixed nibble positions once per shared input,
                // leaving weight decoding as masks, without per-code shifts.
                a[j+1]*=0.0625f;a[j+2]*=0.00390625f;a[j+3]*=0.000244140625f;
            }
            else {for(uint t=0;t<4;++t)bias_sum+=a[j+t];}
        }
        #pragma unroll
        for(uint c=0;c<Outputs;++c) {
            if(col+c>=N)continue;
            ulong first=(expert*N+col+c)*K+k;float dot=0;
            #pragma unroll
            for(uint j=0;j<step;j+=4) {
                float sub=0;
                if(bits==4) {
                    // One aligned packed load per four codes, not four
                    // independently addressed U32 loads and 64-bit indices.
                    uint codes=reinterpret_cast<device const ushort*>(w)[(first+j)/4];
                    sub=float(codes&15)*a[j];
                    sub+=float(codes&0x00f0)*a[j+1];
                    sub+=float(codes&0x0f00)*a[j+2];
                    sub+=float(codes&0xf000)*a[j+3];
                } else {
                    uint codes=reinterpret_cast<device const uint*>(w)[(first+j)/4];
                    dot+=float(codes&255)*a[j];
                    dot+=float((codes>>8)&255)*a[j+1];
                    dot+=float((codes>>16)&255)*a[j+2];
                    dot+=float(codes>>24)*a[j+3];
                }
                if(bits==4)dot+=sub;
            }
            // The affine group contracts scale*dot + bias*xsum, but adding
            // that group to the running accumulator is a separate rounding.
            // Leaving both implicit let tail-check removal move the FMA
            // boundary and change whole-model choices. Keep it explicit.
            float term=fma(dot,float(scales[first/group]),bias_sum*float(biases[first/group]));
            sums[c]+=term;
        }
    }
    for(uint c=0;c<Outputs;++c) {
        float sum=simd_sum(sums[c]);
        if(lane==0 && col+c<N) {
            float value=mlx_bf(sum);
            if(Post==1) {value=mlx_bf(value*0.25f);value=mlx_bf(value*mlx_sigmoid_bf(value));}
            if(Post==2)value=mlx_bf(2.0f*mlx_sigmoid_bf(mlx_bf(value*0.25f)));
            y[col+c]=value;
            if(Post==3)tile[c]=value;
        }
    }
}
template<uint bits,uint group>
inline void q4a_vector(device const uchar* w,device const float* x,device float* y,
    uint K,uint totalN,uint N,uint col,uint lane,ulong expert) {
    constexpr uint pack=32/bits;
    if(K%(pack*64)==0 && N>=8 && N%8==0)
        q4a_vector_step<bits,group,pack*2>(w,x,y,K,totalN,N,col,lane,expert);
    else q4a_vector_step<bits,group,pack>(w,x,y,K,totalN,N,col,lane,expert);
}
// Small-batch arithmetic uses affine F32 decoded values, not the singleton
// bias contraction or BF16 tile staging. Eight lanes reduce independent
// quantization groups. Activation values are already BF16 in F32 storage.
kernel void q4a_wide(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    uint n=g.x*8+tid/8,lane=tid%8;if(n>=p[1])return;
    float sum=0;
    for(uint base=lane*p[4];base<p[0];base+=8*p[4])for(uint j=0;j<p[4];j+=8) {
        float sub=0;
        for(uint t=0;t<8;++t)sub+=x[ulong(g.y)*p[0]+base+j+t]*q4a_value(w,p[0],p[1],ulong(n)*p[0]+base+j+t,p[3],p[4]);
        sum+=sub;
    }
    sum=kquant_sum<8>(sum);if(lane==0)y[ulong(g.y)*p[1]+n]=mlx_bf(sum);
}
// Keep the small-prompt eight-lane contraction, but specialize the packed
// format and hoist its group scale/bias. The generic entry above stays an
// independent oracle: no BF16 weight staging or singleton bias sums here.
template<uint Bits,uint Group>
inline void q4a_wide_packed(device const uchar* w,device const float* x,device float* y,
    constant uint* p,uint2 g,uint tid) {
    #pragma clang fp reassociate(off)
    constexpr uint Pack=32/Bits,Mask=(1u<<Bits)-1;
    uint K=p[0],N=p[1],n=g.x*8+tid/8,lane=tid%8;
    if(n>=N)return;
    x+=ulong(p[6]+g.y)*K;y+=ulong(p[6]+g.y)*N;
    device const uint* codes=reinterpret_cast<device const uint*>(w);
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N*Bits/8);
    float sum=0;
    for(uint base=lane*Group;base<K;base+=8*Group) {
        ulong first=ulong(n)*K+base;
        float scale=float(scales[first/Group]),bias=float(scales[ulong(K)*N/Group+first/Group]);
        #pragma unroll
        for(uint j=0;j<Group;j+=8) {
            float sub=0;
            #pragma unroll
            for(uint t=0;t<8;++t) {
                uint code=(codes[(first+j+t)/Pack]>>(((j+t)%Pack)*Bits))&Mask;
                float value=float(code)*scale+bias;
                sub+=x[base+j+t]*value;
            }
            sum+=sub;
        }
    }
    sum=kquant_sum<8>(sum);if(lane==0)y[n]=mlx_bf(sum);
}
#define Q4A_WIDE_PACKED(Name,Bits,Group) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]], \
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    q4a_wide_packed<Bits,Group>(w,x,y,p,g,tid); \
}
Q4A_WIDE_PACKED(q4a_wide4_packed,4,32)
Q4A_WIDE_PACKED(q4a_wide8_packed,8,64)
#undef Q4A_WIDE_PACKED
kernel void q4a_mv(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    if(p[3]==4)q4a_vector<4,32>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1],p[0],p[1],p[1],g.x*16+tid/32*4,tid%32,0);
    else q4a_vector<8,64>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1],p[0],p[1],p[1],g.x*16+tid/32*4,tid%32,0);
}
// Separate pipeline entries also specialize register allocation and remove
// uniform 4/8-bit and step branches from the hot decode program.
#define Q4A_MV_ENTRY(Name,Bits,Group,Step) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]], \
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1]; \
    q4a_vector_step<Bits,Group,Step>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1], \
        p[0],p[1],p[1],g.x*16+tid/32*4,tid%32,0); \
}
Q4A_MV_ENTRY(q4a_mv4,4,32,8)
Q4A_MV_ENTRY(q4a_mv4_fast,4,32,16)
Q4A_MV_ENTRY(q4a_mv8,8,64,4)
Q4A_MV_ENTRY(q4a_mv8_fast,8,64,8)
#undef Q4A_MV_ENTRY

// HC's four-wide injection otherwise activates only one SIMD, serializing
// four outputs over a 10K-wide input. Give each output its own SIMD without
// changing any lane's K order or its final reduction.
kernel void q4a_mv4_narrow(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    q4a_vector_step<4,32,8,1>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1],
        p[0],p[1],p[1],g.x*4+tid/32,tid%32,0);
}
// The 320-wide HC down projection has too few 16-column workgroups to
// occupy M5 Max. Two SIMDs per group double independent scheduling units.
kernel void q4a_mv4_fast2(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    q4a_vector_step<4,32,16>(w,x+ulong(g.y)*p[0],y+ulong(g.y)*p[1],
        p[0],p[1],p[1],g.x*8+tid/32*4,tid%32,0);
}
// Join launches, not reductions: HC down retains its 16-wide lane walk,
// and each of the four injection outputs retains its 8-wide lane walk.
// The post-ops have the same explicit BF16 boundaries as their old kernels.
kernel void q4a_hc_down_vector(device const uchar* down [[buffer(0)]],device const uchar* inject [[buffer(1)]],
    device const float* x [[buffer(2)]],device float* low [[buffer(3)]],device float* gain [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    x+=ulong(g.y)*10240;
    if(g.x<40)q4a_vector_step<4,32,16,4,1>(down,x,low+ulong(g.y)*320,10240,320,320,g.x*8+tid/32*4,tid%32,0);
    else q4a_vector_step<4,32,8,1,2>(inject,x,gain+ulong(g.y)*4,10240,4,4,(g.x-40)*2+tid/32,tid%32,0);
}
// Four SIMDs own the same four coordinates in the four residual streams.
// Preserve each up projection, then fold its rounded gate in stream order.
// Keep the full gate plane for traces; the consumer reads the on-chip copy.
kernel void q4a_hc_up_mix_vector(device const uchar* w [[buffer(0)]],device const float* low [[buffer(1)]],
    device const float* norm [[buffer(2)]],device float* gate [[buffer(3)]],device float* mixed [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    threadgroup float tile[16];uint sg=tid/32;
    q4a_vector_step<4,32,8,4,3>(w,low+ulong(g.y)*320,gate+ulong(g.y)*10240,
        320,10240,10240,sg*2560+g.x*4,tid%32,0,tile+sg*4);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<4) {
        uint d=g.x*4+tid;float sum=0;
        for(uint s=0;s<4;++s)sum=mlx_bf(sum+mlx_bf(norm[ulong(g.y)*10240+s*2560+d]*mlx_sigmoid_bf(tile[s*4+tid])));
        mixed[ulong(g.y)*2560+d]=mlx_bf(sum*0.25f);
    }
}
kernel void q4a_expert_mv(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint expert=ids[g.y];if(expert>=p[4])return;
    q4a_vector<4,32>(w,x+ulong(p[3] ? g.y : g.y/10)*p[0],y+ulong(g.y)*p[1],
        p[0],p[1]*p[4],p[1],g.x*16+tid/32*4,tid%32,expert);
}

// Singleton affine contracts, including the BF16 bias subtotals, are the
// same as q4a_vector_step<4,32,16>. Share activation loads across gate/up
// and consume their rounded results immediately. Each group owns one routed
// or shared expert; routing order and the later weighted fold are unchanged.
kernel void q4a_expert_gate_up_vector(device const uchar* gw [[buffer(0)]],device const uchar* uw [[buffer(1)]],
    device const uchar* sgw [[buffer(2)]],device const uchar* suw [[buffer(3)]],
    device const float* x [[buffer(4)]],device const uint* ids [[buffer(5)]],
    device float* act [[buffer(6)]],device float* shared_act [[buffer(7)]],constant uint* p [[buffer(8)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint2 nt [[threads_per_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint K=p[0],N=p[1],E=p[3],row=g.y/11,slot=g.y%11,lane=tid%32;
    uint col=g.x*(nt.x/32*4)+tid/32*4;
    if(col>=N || row>=p[2])return;
    bool shared=slot==10;uint expert=shared ? 0 : ids[row*10+slot];
    if(!shared && expert>=E)return;
    device const uchar* weights[2]={shared ? sgw : gw,shared ? suw : uw};
    ulong totalN=shared ? N : ulong(N)*E;
    device const bfloat* scales[2]={reinterpret_cast<device const bfloat*>(weights[0]+ulong(K)*totalN/2),
                                  reinterpret_cast<device const bfloat*>(weights[1]+ulong(K)*totalN/2)};
    float sums[2][4]={{0,0,0,0},{0,0,0,0}};
    x+=ulong(row)*K;
    for(uint k=lane*16;k<K;k+=512) {
        float a[16];float bias_sum=0;
        #pragma unroll
        for(uint j=0;j<16;j+=4) {
            float4 av=*reinterpret_cast<device const float4*>(x+k+j);
            a[j]=av.x;a[j+1]=av.y;a[j+2]=av.z;a[j+3]=av.w;
            bias_sum+=mlx_bf(mlx_bf(mlx_bf(a[j]+a[j+1])+a[j+2])+a[j+3]);
            a[j+1]*=0.0625f;a[j+2]*=0.00390625f;a[j+3]*=0.000244140625f;
        }
        #pragma unroll
        for(uint plane=0;plane<2;++plane) {
            #pragma unroll
            for(uint c=0;c<4;++c) {
                if(col+c>=N)continue;
                ulong first=(ulong(expert)*N+col+c)*K+k;float dot=0;
                #pragma unroll
                for(uint j=0;j<16;j+=4) {
                    uint codes=reinterpret_cast<device const ushort*>(weights[plane])[(first+j)/4];
                    float sub=float(codes&15)*a[j];
                    sub+=float(codes&0x00f0)*a[j+1];sub+=float(codes&0x0f00)*a[j+2];sub+=float(codes&0xf000)*a[j+3];
                    dot+=sub;
                }
                float term=fma(dot,float(scales[plane][first/32]),bias_sum*float(scales[plane][ulong(K)*totalN/32+first/32]));
                sums[plane][c]+=term;
            }
        }
    }
    device float* out=shared ? shared_act+ulong(row)*N : act+ulong(row*10+slot)*N;
    for(uint c=0;c<4;++c) {
        float gate=mlx_bf(simd_sum(sums[0][c])),up=mlx_bf(simd_sum(sums[1][c]));
        if(lane==0 && col+c<N)out[col+c]=mlx_bf(mlx_bf(gate*mlx_sigmoid_bf(gate))*up);
    }
}

// Stable, bounded GPU permutation only. The original routing IDs and their
// accumulation order never change. Scratch reuses the GGUF alignment list.
// 128 tokens * top10 fit in a 2048-key sorting network (8 KiB shared).
kernel void q4a_expert_order(device const uint* ids [[buffer(0)]],device uint* order [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint keys[2048];
    for(uint i=tid;i<2048;i+=256)
        keys[i]=i<p[0] && ids[i]<512 ? ids[i]*2048+i : UINT_MAX;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint width=2;width<=2048;width*=2)for(uint stride=width/2;stride>0;stride/=2) {
        for(uint i=tid;i<2048;i+=256) {
            uint other=i^stride;
            if(other>i) {
                uint a=keys[i],b=keys[other];bool ascending=(i&width)==0;
                keys[i]=ascending ? min(a,b) : max(a,b);
                keys[other]=ascending ? max(a,b) : min(a,b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=tid;i<p[0];i+=256)order[i]=keys[i]==UINT_MAX ? UINT_MAX : keys[i]%2048;
}

// Schedule nearby expert entries together at each output tile. Every SIMD
// retains the original singleton arithmetic and scatters directly to its
// token/top-k destination: no activation gather or floating-point atomics.
kernel void q4a_expert_ordered(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],device const uint* order [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint sorted=g.y*8+g.x%8;if(sorted>=p[2])return;
    uint entry=order[sorted];if(entry>=p[2])return;
    uint expert=ids[entry];if(expert>=p[4])return;
    q4a_vector<4,32>(w,x+ulong(p[3] ? entry : entry/10)*p[0],y+ulong(entry)*p[1],
        p[0],p[1]*p[4],p[1],g.x/8*16+tid/32*4,tid%32,expert);
}

// Separate expert steps matter for register allocation just as much as for
// dense projections. A uniform branch still reserves the larger live set.
#define Q4A_EXPERT_ENTRY(Name,Step,Ordered) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],device const uint* order [[buffer(4)]], \
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    uint index=Ordered ? g.y*8+g.x%8 : g.y;if(index>=p[2])return; \
    uint entry=Ordered ? order[index] : index;if(entry>=p[2])return; \
    uint expert=ids[entry];if(expert>=p[4])return; \
    uint tile=Ordered ? g.x/8 : g.x; \
    q4a_vector_step<4,32,Step>(w,x+ulong(p[3] ? entry : entry/10)*p[0],y+ulong(entry)*p[1], \
        p[0],p[1]*p[4],p[1],tile*16+tid/32*4,tid%32,expert); \
}
Q4A_EXPERT_ENTRY(q4a_expert4,8,false)
Q4A_EXPERT_ENTRY(q4a_expert4_fast,16,false)
Q4A_EXPERT_ENTRY(q4a_expert4_ordered,8,true)
Q4A_EXPERT_ENTRY(q4a_expert4_fast_ordered,16,true)
#undef Q4A_EXPERT_ENTRY

// Two adjacent routed rows can share packed weights, scales and biases.
// Keep each row's singleton lane/reduction/FMA contract: unlike choosing
// matrix arithmetic by expert occupancy, reuse cannot change a neighbour's
// logits when another request joins. Only integer routing is shared.
template<uint Step>
inline void q4a_expert_pair(device const uchar* w,device const float* x,device float* y,
    uint K,uint N,uint col,uint lane,uint expert,uint first,uint second,bool per_entry) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float sums[2][4];
    for(uint r=0;r<2;++r)for(uint c=0;c<4;++c)sums[r][c]=0;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N*512/2);
    device const bfloat* biases=scales+ulong(K)*N*512/32;
    for(uint k=lane*Step;k<K;k+=32*Step) {
        float a[2][Step],bias_sum[2]={0,0};
        #pragma unroll
        for(uint r=0;r<2;++r) {
            uint entry=r==0 ? first : second;
            device const float* row=x+ulong(per_entry ? entry : entry/10)*K;
            #pragma unroll
            for(uint j=0;j<Step;j+=4) {
                float4 av=*reinterpret_cast<device const float4*>(row+k+j);
                bias_sum[r]+=mlx_bf(mlx_bf(mlx_bf(av.x+av.y)+av.z)+av.w);
                a[r][j]=av.x;a[r][j+1]=av.y*0.0625f;
                a[r][j+2]=av.z*0.00390625f;a[r][j+3]=av.w*0.000244140625f;
            }
        }
        #pragma unroll
        for(uint c=0;c<4;++c) {
            ulong at=(ulong(expert)*N+col+c)*K+k;
            float dot[2]={0,0};
            #pragma unroll
            for(uint j=0;j<Step;j+=4) {
                uint codes=reinterpret_cast<device const ushort*>(w)[(at+j)/4];
                #pragma unroll
                for(uint r=0;r<2;++r) {
                    float sub=float(codes&15)*a[r][j];
                    sub+=float(codes&0x00f0)*a[r][j+1];
                    sub+=float(codes&0x0f00)*a[r][j+2];
                    sub+=float(codes&0xf000)*a[r][j+3];
                    dot[r]+=sub;
                }
            }
            float scale=float(scales[at/32]),bias=float(biases[at/32]);
            for(uint r=0;r<2;++r)sums[r][c]+=fma(dot[r],scale,bias_sum[r]*bias);
        }
    }
    for(uint r=0;r<2;++r)for(uint c=0;c<4;++c) {
        float value=simd_sum(sums[r][c]);
        if(lane==0)y[ulong(r==0 ? first : second)*N+col+c]=mlx_bf(value);
    }
}

#define Q4A_EXPERT_PAIR(Name,Step) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],device const uint* order [[buffer(4)]], \
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    uint index=(g.y*4+g.x%4)*2;if(index>=p[2])return; \
    uint first=order[index],second=index+1<p[2] ? order[index+1] : UINT_MAX; \
    if(first>=p[2])return;uint expert=ids[first];if(expert>=512)return; \
    uint col=g.x/4*16+tid/32*4,lane=tid%32; \
    if(second<p[2] && ids[second]==expert) { \
        q4a_expert_pair<Step>(w,x,y,p[0],p[1],col,lane,expert,first,second,p[3]!=0); \
    } else { \
        q4a_vector_step<4,32,Step>(w,x+ulong(p[3] ? first : first/10)*p[0],y+ulong(first)*p[1], \
            p[0],p[1]*512,p[1],col,lane,expert); \
        if(second<p[2] && ids[second]<512)q4a_vector_step<4,32,Step>(w,x+ulong(p[3] ? second : second/10)*p[0], \
            y+ulong(second)*p[1],p[0],p[1]*512,p[1],col,lane,ids[second]); \
    } \
}
Q4A_EXPERT_PAIR(q4a_expert4_pair,8)
Q4A_EXPERT_PAIR(q4a_expert4_fast_pair,16)
#undef Q4A_EXPERT_PAIR

// Mixed batches retain singleton arithmetic for decoding and short logical
// prompt chunks. The mask is per token, never a route-occupancy election.
kernel void q4a_expert_vector_masked(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint row=g.y/10;
    if((p[5+row/32]>>(row%32))&1)return;
    uint expert=ids[g.y];if(expert>=512)return;
    q4a_vector<4,32>(w,x+ulong(p[3] ? g.y : row)*p[0],y+ulong(g.y)*p[1],
        p[0],p[1]*512,p[1],g.x*16+tid/32*4,tid%32,expert);
}

// Grouped affine prefill: one expert's routed rows share a
// bounded BF16 tile, with direct scatter to token/top-k destinations.
// The matrix contraction has a different numerical contract from singleton
// decode. The logical-chunk mask owns that election, not physical batch size.
// Retain 32x32x32 as a GPU exactness baseline; 32x64x64 uses 12 KiB staging.
inline void q4a_unpack_group32(uint4 words,float scale,float bias,threadgroup bfloat* dst) {
    #pragma unroll
    for(uint word=0;word<4;++word) {
        uint codes=words[word];
        float4 lo=float4(codes&15,(codes>>4)&15,(codes>>8)&15,(codes>>12)&15);
        float4 hi=float4((codes>>16)&15,(codes>>20)&15,(codes>>24)&15,codes>>28);
        *reinterpret_cast<threadgroup bfloat4*>(dst+word*8)=bfloat4(fma(lo,float4(scale),float4(bias)));
        *reinterpret_cast<threadgroup bfloat4*>(dst+word*8+4)=bfloat4(fma(hi,float4(scale),float4(bias)));
    }
}
inline void q4a_unpack_group32_masked(uint4 words,float scale,float bias,threadgroup bfloat* dst) {
    float4 scales=float4(scale,scale*0.0625f,scale,scale*0.0625f);
    #pragma unroll
    for(uint word=0;word<4;++word) {
        uchar4 bytes=as_type<uchar4>(words[word]);
        #pragma unroll
        for(uint j=0;j<2;++j) {
            uint a=bytes[j*2],b=bytes[j*2+1];
            float4 codes=float4(a&15,a&240,b&15,b&240);
            *reinterpret_cast<threadgroup bfloat4*>(dst+word*8+j*4)=bfloat4(fma(codes,scales,float4(bias)));
        }
    }
}
template<uint BM,uint BN,uint BK,bool GroupLoad=false,uint Pad=0,bool PackedInput=false,uint LoadValues=32,uint Threads=128,bool Masked=false>
inline void q4a_expert_matrix(device const uchar* w,device const float* x,
    device const uint* lists,device const uint* counts,device const uint* tiles,
    device float* y,constant uint* p,uint2 g,uint tid,threadgroup bfloat* a,threadgroup bfloat* b) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    uint count=min(BM,counts[expert]-first);
    auto at=tensor(a,extents<int,BK,BM>(),array<int,2>{1,BK+Pad});
    auto bt=tensor(b,extents<int,BK,BN>(),array<int,2>{1,BK+Pad});
    constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<Threads/32>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=BK) {
        if(BK==32) {
            for(uint i=tid;i<1024;i+=Threads) {
                uint r=i/32,k=base+i%32,col=g.x*BN+r;
                uint entry=r<count ? lists[expert*p[2]+first+r] : p[2];
                a[i]=bfloat(entry<p[2] ? x[ulong(p[3] ? entry : entry/10)*p[0]+k] : 0);
                b[i]=bfloat(col<p[1] ? q4a_value_t<4,32>(w,p[0],p[1]*512,(ulong(expert)*p[1]+col)*p[0]+k) : 0);
            }
        } else {
            // Four-value packed staging: each code word and scale/bias
            // pair is loaded once, then reused for four matrix operands.
            for(uint i=tid*4;i<BM*BK;i+=Threads*4) {
                uint r=i/BK,k=base+i%BK;
                uint entry=r<count ? lists[expert*p[2]+first+r] : p[2];
                ulong offset=ulong(p[3] ? entry : entry/10)*p[0]+k;
                bfloat4 v=0;
                if(entry<p[2]) {
                    if(PackedInput)v=*reinterpret_cast<device const bfloat4*>(reinterpret_cast<device const bfloat*>(x)+offset);
                    else v=bfloat4(*reinterpret_cast<device const float4*>(x+offset));
                }
                *reinterpret_cast<threadgroup bfloat4*>(a+r*(BK+Pad)+i%BK)=v;
            }
            if(GroupLoad) {
                // Own an entire affine group, not four values from each of
                // eight different groups. One packed 16-byte load and one
                // scale/bias pair produce the same 32 BF16 tensor operands.
                // No change to tensor shape, K order or the output boundary.
                for(uint i=tid*LoadValues;i<BN*BK;i+=Threads*LoadValues) {
                    uint col=g.x*BN+i/BK,k=base+i%BK;
                    ulong at=(ulong(expert)*p[1]+col)*p[0]+k;
                    uint4 words=0;float scale=0,bias=0;
                    if(col<p[1]) {
                        if(LoadValues==32)words=*reinterpret_cast<device const uint4*>(w+at/2);
                        else if(LoadValues==8)words.x=*reinterpret_cast<device const uint*>(w+at/2);
                        else words.x=*reinterpret_cast<device const ushort*>(w+at/2);
                        device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(p[0])*p[1]*512/2);
                        scale=float(scales[at/32]);bias=float(scales[ulong(p[0])*p[1]*512/32+at/32]);
                    }
                    auto dst=b+(i/BK)*(BK+Pad)+i%BK;
                    if(LoadValues==32) {
                        if(Masked)q4a_unpack_group32_masked(words,scale,bias,dst);
                        else q4a_unpack_group32(words,scale,bias,dst);
                    }
                    else {
                        float4 lo=float4(words.x&15,(words.x>>4)&15,(words.x>>8)&15,(words.x>>12)&15);
                        *reinterpret_cast<threadgroup bfloat4*>(dst)=bfloat4(fma(lo,float4(scale),float4(bias)));
                        if(LoadValues==8) {
                            float4 hi=float4((words.x>>16)&15,(words.x>>20)&15,(words.x>>24)&15,words.x>>28);
                            *reinterpret_cast<threadgroup bfloat4*>(dst+4)=bfloat4(fma(hi,float4(scale),float4(bias)));
                        }
                    }
                }
            } else for(uint i=tid*4;i<BN*BK;i+=Threads*4) {
                uint col=g.x*BN+i/BK,k=base+i%BK;float4 v=0;
                if(col<p[1]) {
                    ulong at=(ulong(expert)*p[1]+col)*p[0]+k;
                    uint code=reinterpret_cast<device const ushort*>(w)[at/4];
                    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(p[0])*p[1]*512/2);
                    float s=float(scales[at/32]),bias=float(scales[ulong(p[0])*p[1]*512/32+at/32]);
                    float4 codes=float4(code&15,(code>>4)&15,(code>>8)&15,code>>12);
                    v=fma(codes,float4(s),float4(bias));
                }
                *reinterpret_cast<threadgroup bfloat4*>(b+(i/BK)*(BK+Pad)+i%BK)=bfloat4(v);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*BN+ij[0];
        if(it.is_valid_element() && ij[1]<count && col<p[1]) {
            uint entry=lists[expert*p[2]+first+ij[1]];
            uint row=entry/10;
            if((p[5+row/32]>>(row%32))&1)y[ulong(entry)*p[1]+col]=mlx_bf(*it);
        }
    }
}
#define Q4A_EXPERT_MATRIX(Name,BN,BK) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]], \
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup bfloat a[32*BK],b[BN*BK]; \
    q4a_expert_matrix<32,BN,BK>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
}
Q4A_EXPERT_MATRIX(q4a_expert_mm,32,32)
Q4A_EXPERT_MATRIX(q4a_expert_mm_wide,64,64)
#undef Q4A_EXPERT_MATRIX

// A compact tail uses fewer tensor rows, without altering the routed-entry
// order, K accumulation, output mask or BF16 boundary. Full tiles keep 32 rows.
kernel void q4a_expert_mm_tail(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    threadgroup bfloat a[32*64],b[64*64];
    if(counts[expert]-first<=8)
        q4a_expert_matrix<8,64,64>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else if(counts[expert]-first<=16)
        q4a_expert_matrix<16,64,64>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else
        q4a_expert_matrix<32,64,64>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
}

kernel void q4a_expert_mm_group32(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    threadgroup bfloat a[32*64],b[64*64];
    if(counts[expert]-first<=8)
        q4a_expert_matrix<8,64,64,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else if(counts[expert]-first<=16)
        q4a_expert_matrix<16,64,64,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else
        q4a_expert_matrix<32,64,64,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
}

// Padded group-wise staging; also the fallback when device-input scratch is
// unavailable. Keep the same contraction and vary only local row pitch.
kernel void q4a_expert_mm_group32_pad(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    threadgroup bfloat a[32*72],b[64*72];
    if(counts[expert]-first<=8)
        q4a_expert_matrix<8,64,64,true,8>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else if(counts[expert]-first<=16)
        q4a_expert_matrix<16,64,64,true,8>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else
        q4a_expert_matrix<32,64,64,true,8>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
}

// The projection input is already BF16 in numerical value. Stage it once
// outside the N tiles so each tile reads two-byte values instead of loading
// F32 and converting the same activation repeatedly. The allocation is
// shared with dense projection scratch, never retained per expert/layer.
kernel void q4a_expert_mm_group32_packed(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    threadgroup bfloat a[32*72],b[64*72];
    if(counts[expert]-first<=8)
        q4a_expert_matrix<8,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else if(counts[expert]-first<=16)
        q4a_expert_matrix<16,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else
        q4a_expert_matrix<32,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
}

#ifdef PADDOCK_KERNEL_DIAGNOSTICS
// Expert-major 64-row schedule for gate/up weight-reuse diagnostics.
kernel void q4a_expert_tiles64(device const uint* counts [[buffer(0)]],device uint* tiles [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[16],offsets[16];
    uint num=(counts[tid]+63)/64,off=simd_prefix_exclusive_sum(num),sum=simd_sum(num);
    if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        uint v=lane<16 ? sums[lane] : 0,prefix=simd_prefix_exclusive_sum(v);
        uint total=simd_sum(v);
        if(lane<16)offsets[lane]=prefix;
        if(lane==0)tiles[0]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<num;++j){uint out=offsets[sg]+off+j;tiles[1+2*out]=tid;tiles[2+2*out]=j*64;}
}
kernel void q4a_expert_mm_rows64(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    threadgroup bfloat a[64*72],b[64*72];
    if(counts[expert]-first<=8)q4a_expert_matrix<8,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else if(counts[expert]-first<=16)q4a_expert_matrix<16,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else if(counts[expert]-first<=32)q4a_expert_matrix<32,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
    else q4a_expert_matrix<64,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b);
}

// Register-resident operand experiment: unpack directly into MPP fragments,
// avoiding shared-memory staging and its two barriers per K step.
inline void q4a_expert_register(device const uchar* w,device const bfloat* x,
    device const uint* lists,device const uint* counts,device const uint* tiles,
    device float* y,constant uint* p,uint2 g,uint sg) {
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y]+(sg/2)*16;
    if(first>=counts[expert])return;
    uint count=min(16u,counts[expert]-first),column=g.x*64+(sg%2)*32;
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroup> op;
    auto a=op.template get_left_input_cooperative_tensor<bfloat,bfloat,float>();
    auto b=op.template get_right_input_cooperative_tensor<bfloat,bfloat,float>();
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(p[0])*p[1]*512/2);
    device const bfloat* biases=scales+ulong(p[0])*p[1]*512/32;
    for(uint base=0;base<p[0];base+=32) {
        for(auto it=a.begin();it!=a.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();
            uint entry=ij[1]<count ? lists[expert*p[2]+first+ij[1]] : p[2];
            *it=entry<p[2] ? x[ulong(p[3] ? entry : entry/10)*p[0]+base+ij[0]] : bfloat(0);
        }
        for(auto it=b.begin();it!=b.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();uint col=column+ij[1];
            bfloat value=0;
            if(col<p[1]) {
                ulong pos=(ulong(expert)*p[1]+col)*p[0]+base+ij[0];
                uint code=(reinterpret_cast<device const uint*>(w)[pos/8]>>((pos%8)*4))&15;
                value=bfloat(fma(float(code),float(scales[pos/32]),float(biases[pos/32])));
            }
            *it=value;
        }
        op.run(a,b,acc);
    }
    for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();uint col=column+ij[0];
        if(ij[1]<count && col<p[1]) {
            uint entry=lists[expert*p[2]+first+ij[1]],row=entry/10;
            if((p[5+row/32]>>(row%32))&1)y[ulong(entry)*p[1]+col]=mlx_bf(*it);
        }
    }
}
kernel void q4a_expert_mm_register(device const uchar* w [[buffer(0)]],device const bfloat* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    q4a_expert_register(w,x,lists,counts,tiles,y,p,g,sg);
}

// Fewer participating SIMD groups give each group a larger output tile.
#define Q4A_EXPERT_GROUPS(Name,T,P,Masked) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]], \
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    if(g.y>=tiles[0])return; \
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y]; \
    if(expert>=512 || first>=counts[expert])return; \
    threadgroup bfloat a[32*(64+P)],b[64*(64+P)]; \
    if(counts[expert]-first<=8)q4a_expert_matrix<8,64,64,true,P,true,32,T,Masked>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
    else if(counts[expert]-first<=16)q4a_expert_matrix<16,64,64,true,P,true,32,T,Masked>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
    else q4a_expert_matrix<32,64,64,true,P,true,32,T,Masked>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
}
Q4A_EXPERT_GROUPS(q4a_expert_mm_sg1,32,8,false)
Q4A_EXPERT_GROUPS(q4a_expert_mm_sg2,64,8,false)
Q4A_EXPERT_GROUPS(q4a_expert_mm_pad4,128,4,false)
Q4A_EXPERT_GROUPS(q4a_expert_mm_pad16,128,16,false)
Q4A_EXPERT_GROUPS(q4a_expert_mm_masked,128,8,true)
#undef Q4A_EXPERT_GROUPS

// Vary only the cooperative B-loader's ownership; matrix shape/order stay fixed.
#define Q4A_EXPERT_LOAD(Name,L) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]], \
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    if(g.y>=tiles[0])return; \
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y]; \
    if(expert>=512 || first>=counts[expert])return; \
    threadgroup bfloat a[32*72],b[64*72]; \
    if(counts[expert]-first<=8)q4a_expert_matrix<8,64,64,true,8,true,L>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
    else if(counts[expert]-first<=16)q4a_expert_matrix<16,64,64,true,8,true,L>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
    else q4a_expert_matrix<32,64,64,true,8,true,L>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
}
Q4A_EXPERT_LOAD(q4a_expert_mm_load4,4)
Q4A_EXPERT_LOAD(q4a_expert_mm_load8,8)
#undef Q4A_EXPERT_LOAD

// Candidate only: widen K to amortize staging barriers. Keep routing, row
// masks and BF16 boundaries; exact arithmetic still requires qualification.
#define Q4A_EXPERT_K128(Name,BN) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]], \
    device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    if(g.y>=tiles[0])return; \
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y]; \
    if(expert>=512 || first>=counts[expert])return; \
    threadgroup bfloat a[32*136],b[BN*136]; \
    if(counts[expert]-first<=8)q4a_expert_matrix<8,BN,128,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
    else if(counts[expert]-first<=16)q4a_expert_matrix<16,BN,128,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
    else q4a_expert_matrix<32,BN,128,true,8,true>(w,x,lists,counts,tiles,y,p,g,tid,a,b); \
}
Q4A_EXPERT_K128(q4a_expert_mm_k128_n32,32)
Q4A_EXPERT_K128(q4a_expert_mm_k128_n64,64)
#undef Q4A_EXPERT_K128
#endif

// Compact sorted inputs let TensorOps read A directly from device memory.
// Only B is staged per output tile. Offsets are an exclusive scan of the
// actual counts (no padded expert rows and no CPU routing synchronization).
kernel void q4a_expert_offsets(device const uint* counts [[buffer(0)]],device uint* offsets [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[16],prefixes[16];
    uint count=counts[tid],prefix=simd_prefix_exclusive_sum(count);
    uint sum=simd_sum(count);
    if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        uint v=lane<16 ? sums[lane] : 0;
        uint off=simd_prefix_exclusive_sum(v);
        if(lane<16)prefixes[lane]=off;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    offsets[tid]=prefixes[sg]+prefix;
}
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
// Rejected serving candidate: retained only for exact cost diagnostics.
// Build compact input offsets and a 64-row schedule together. Only the down
// projection uses this schedule; gate/up keep their existing 32-row tiles.
kernel void q4a_expert_plan64(device const uint* counts [[buffer(0)]],device uint* offsets [[buffer(1)]],
    device uint* tiles [[buffer(2)]],constant uint* p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint2 sums[16],prefixes[16];
    // A 33..48-row remainder is cheaper as 32 + 8/16 than a padded 64.
    // Use 64 only when it replaces two full 32-row matrix operations.
    uint count=counts[tid],full=count/64,tail=count%64;
    uint num=full+(tail>48 ? 1 : (tail+31)/32);
    uint2 prefix=uint2(simd_prefix_exclusive_sum(count),simd_prefix_exclusive_sum(num));
    uint2 sum=uint2(simd_sum(count),simd_sum(num));
    if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        uint2 v=lane<16 ? sums[lane] : uint2(0);
        uint2 off=uint2(simd_prefix_exclusive_sum(v.x),simd_prefix_exclusive_sum(v.y));
        uint total=simd_sum(v.y);
        if(lane<16)prefixes[lane]=off;
        if(lane==0)tiles[0]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix+=prefixes[sg];offsets[tid]=prefix.x;
    for(uint j=0;j<num;++j) {
        uint out=prefix.y+j;tiles[1+2*out]=tid;
        tiles[2+2*out]=j<full ? j*64 : full*64+(j-full)*32;
    }
}
#endif
kernel void q4a_expert_pack(device const float* x [[buffer(0)]],device const uint* lists [[buffer(1)]],
    device const uint* counts [[buffer(2)]],device const uint* tiles [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],device bfloat* sorted [[buffer(5)]],
    constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],k=g.x*512+tid*4;
    if(expert>=512 || first>=counts[expert] || k>=p[0])return;
    uint count=min(32u,counts[expert]-first);
    for(uint r=0;r<count;++r) {
        uint entry=lists[expert*p[2]+first+r];
        if(entry>=p[2])continue;
        ulong src=ulong(p[3] ? entry : entry/10)*p[0]+k;
        ulong dst=ulong(offsets[expert]+first+r)*p[0]+k;
        *reinterpret_cast<device bfloat4*>(sorted+dst)=bfloat4(*reinterpret_cast<device const float4*>(x+src));
    }
}
#ifdef PADDOCK_KERNEL_DIAGNOSTICS
kernel void q4a_expert_pack64(device const float* x [[buffer(0)]],device const uint* lists [[buffer(1)]],
    device const uint* counts [[buffer(2)]],device const uint* tiles [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],device bfloat* sorted [[buffer(5)]],
    constant uint* p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],k=g.x*512+tid*4;
    if(expert>=512 || first>=counts[expert] || k>=p[0])return;
    uint remain=counts[expert]-first,count=min(remain<=48 ? 32u : 64u,remain);
    for(uint r=0;r<count;++r) {
        uint entry=lists[expert*p[2]+first+r];
        if(entry>=p[2])continue;
        ulong src=ulong(p[3] ? entry : entry/10)*p[0]+k;
        ulong dst=ulong(offsets[expert]+first+r)*p[0]+k;
        *reinterpret_cast<device bfloat4*>(sorted+dst)=bfloat4(*reinterpret_cast<device const float4*>(x+src));
    }
}
#endif
template<uint BM>
inline void q4a_expert_direct(device const uchar* w,device bfloat* x,
    device const uint* lists,device const uint* counts,device const uint* tiles,
    device const uint* offsets,device float* y,constant uint* p,uint2 g,uint tid,threadgroup bfloat* b) {
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(BM,counts[expert]-first);
    auto at=tensor(x+ulong(offsets[expert])*p[0],extents<int,dynamic_extent,dynamic_extent>(p[0],counts[expert]),array<int,2>{1,int(p[0])});
    auto bt=tensor(b,extents<int,64,64>(),array<int,2>{1,72});
    constexpr auto desc=matmul2d_descriptor(BM,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<p[0];base+=64) {
        for(uint i=tid*32;i<64*64;i+=128*32) {
            uint col=g.x*64+i/64,k=base+i%64;
            ulong pos=(ulong(expert)*p[1]+col)*p[0]+k;
            uint4 words=0;float scale=0,bias=0;
            if(col<p[1]) {
                words=*reinterpret_cast<device const uint4*>(w+pos/2);
                device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(p[0])*p[1]*512/2);
                scale=float(scales[pos/32]);bias=float(scales[ulong(p[0])*p[1]*512/32+pos/32]);
            }
            q4a_unpack_group32(words,scale,bias,b+(i/64)*72+i%64);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input=at.slice<64,BM>(base,first);op.run(input,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*64+ij[0];
        if(it.is_valid_element() && ij[1]<count && col<p[1]) {
            uint entry=lists[expert*p[2]+first+ij[1]],row=entry/10;
            if((p[5+row/32]>>(row%32))&1)y[ulong(entry)*p[1]+col]=mlx_bf(*it);
        }
    }
}
kernel void q4a_expert_mm_direct(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device const uint* offsets [[buffer(5)]],device float* y [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    threadgroup bfloat b[64*72];
    if(counts[expert]-first<=8)q4a_expert_direct<8>(w,x,lists,counts,tiles,offsets,y,p,g,tid,b);
    else if(counts[expert]-first<=16)q4a_expert_direct<16>(w,x,lists,counts,tiles,offsets,y,p,g,tid,b);
    else q4a_expert_direct<32>(w,x,lists,counts,tiles,offsets,y,p,g,tid,b);
}

#ifdef PADDOCK_KERNEL_DIAGNOSTICS
kernel void q4a_expert_mm_direct64(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],
    device const uint* lists [[buffer(2)]],device const uint* counts [[buffer(3)]],device const uint* tiles [[buffer(4)]],
    device const uint* offsets [[buffer(5)]],device float* y [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    threadgroup bfloat b[64*72];
    if(counts[expert]-first<=8)q4a_expert_direct<8>(w,x,lists,counts,tiles,offsets,y,p,g,tid,b);
    else if(counts[expert]-first<=16)q4a_expert_direct<16>(w,x,lists,counts,tiles,offsets,y,p,g,tid,b);
    else if(counts[expert]-first<=48)q4a_expert_direct<32>(w,x,lists,counts,tiles,offsets,y,p,g,tid,b);
    else q4a_expert_direct<64>(w,x,lists,counts,tiles,offsets,y,p,g,tid,b);
}

// Defined in the shared MLX operations section of the same Metal library.
inline float mlx_sigmoid_bf(float v);
// Original paired contraction: gate/up retain separate compressed weight
// planes but share one activation tile and K traversal. The two accumulator
// fragments have the exact unfused shape/order. Retain BOTH BF16 projection
// boundaries before SwiGLU; only the intermediate device writes disappear.
// Dispatch fusion keeps the original per-thread resource footprint. Gate/up
// tiles share input packing and one launch, without duplicating model weights.
kernel void q4a_expert_gate_up_dispatch(device const uchar* gate [[buffer(0)]],device const uchar* up [[buffer(1)]],
    device const float* x [[buffer(2)]],device const uint* lists [[buffer(3)]],device const uint* counts [[buffer(4)]],
    device const uint* tiles [[buffer(5)]],device float* yg [[buffer(6)]],device float* yu [[buffer(7)]],
    constant uint* p [[buffer(8)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    device const uchar* w=g.z==0 ? gate : up;device float* y=g.z==0 ? yg : yu;
    threadgroup bfloat a[32*72],b[64*72];
    if(counts[expert]-first<=8)q4a_expert_matrix<8,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g.xy,tid,a,b);
    else if(counts[expert]-first<=16)q4a_expert_matrix<16,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g.xy,tid,a,b);
    else q4a_expert_matrix<32,64,64,true,8,true>(w,x,lists,counts,tiles,y,p,g.xy,tid,a,b);
}
template<uint BM>
inline void q4a_expert_gate_up(device const uchar* gate,device const uchar* up,
    device const bfloat* x,device const uint* lists,device const uint* counts,device const uint* tiles,
    device float* y,constant uint* p,uint2 g,uint tid,threadgroup bfloat* a,threadgroup bfloat* b) {
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y],count=min(BM,counts[expert]-first);
    auto at=tensor(a,extents<int,64,BM>(),array<int,2>{1,72});
    auto bt=tensor(b,extents<int,64,64>(),array<int,2>{1,72});
    auto ut=tensor(b+64*72,extents<int,64,64>(),array<int,2>{1,72});
    constexpr auto desc=matmul2d_descriptor(BM,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto ag=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    auto au=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<ag.get_capacity();++i){ag[i]=0;au[i]=0;}
    for(uint base=0;base<p[0];base+=64) {
        for(uint i=tid*4;i<BM*64;i+=512) {
            uint r=i/64,k=base+i%64;
            uint entry=r<count ? lists[expert*p[2]+first+r] : p[2];
            bfloat4 v=0;
            if(entry<p[2])v=*reinterpret_cast<device const bfloat4*>(x+ulong(entry/10)*p[0]+k);
            *reinterpret_cast<threadgroup bfloat4*>(a+r*72+i%64)=v;
        }
        // Stage both B tiles before either contraction. Two barriers per K
        // step, matching the separate kernel, instead of serializing uploads
        // between gate and up. Scratch is 22.5 KiB at the largest row tile.
        for(uint which=0;which<2;++which) {
            device const uchar* w=which==0 ? gate : up;
            for(uint i=tid*32;i<64*64;i+=128*32) {
                uint col=g.x*64+i/64,k=base+i%64;
                ulong index=(ulong(expert)*p[1]+col)*p[0]+k;
                uint4 words=0;float scale=0,bias=0;
                if(col<p[1]) {
                    words=*reinterpret_cast<device const uint4*>(w+index/2);
                    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(p[0])*p[1]*512/2);
                    scale=float(scales[index/32]);bias=float(scales[ulong(p[0])*p[1]*512/32+index/32]);
                }
                q4a_unpack_group32(words,scale,bias,b+which*64*72+(i/64)*72+i%64);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(at,bt,ag);op.run(at,ut,au);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto gi=ag.begin();auto ui=au.begin();
    for(;gi!=ag.end();++gi,++ui) {
        auto ij=gi.get_multidimensional_index();uint col=g.x*64+ij[0];
        if(gi.is_valid_element() && ij[1]<count && col<p[1]) {
            uint entry=lists[expert*p[2]+first+ij[1]],row=entry/10;
            if((p[5+row/32]>>(row%32))&1) {
                float gv=mlx_bf(*gi),uv=mlx_bf(*ui);
                y[ulong(entry)*p[1]+col]=mlx_bf(mlx_bf(gv*mlx_sigmoid_bf(gv))*uv);
            }
        }
    }
}
kernel void q4a_expert_gate_up_packed(device const uchar* gate [[buffer(0)]],device const uchar* up [[buffer(1)]],
    device const bfloat* x [[buffer(2)]],device const uint* lists [[buffer(3)]],device const uint* counts [[buffer(4)]],
    device const uint* tiles [[buffer(5)]],device float* y [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    if(g.y>=tiles[0])return;
    uint expert=tiles[1+2*g.y],first=tiles[2+2*g.y];
    if(expert>=512 || first>=counts[expert])return;
    threadgroup bfloat a[32*72],b[128*72];
    if(counts[expert]-first<=8)q4a_expert_gate_up<8>(gate,up,x,lists,counts,tiles,y,p,g,tid,a,b);
    else if(counts[expert]-first<=16)q4a_expert_gate_up<16>(gate,up,x,lists,counts,tiles,y,p,g,tid,a,b);
    else q4a_expert_gate_up<32>(gate,up,x,lists,counts,tiles,y,p,g,tid,a,b);
}
// Matrix rows above already contain their activation. Only singleton/vector
// rows in a mixed pass still need SwiGLU after their original F32 contractions.
kernel void q4a_expert_swiglu_masked(device float* gate [[buffer(0)]],device const float* up [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[2]*p[1] || ((p[5+i/p[1]/10/32]>>((i/p[1]/10)%32))&1))return;
    gate[i]=mlx_bf(mlx_bf(gate[i]*mlx_sigmoid_bf(gate[i]))*up[i]);
}
#endif

// 32x32 output tiles stage only 4 KiB of BF16 operands. No full-plane
// dequantization and no padded workspace allocation. This is the initial
// correctness path; grouped expert prefill / shape elections remain gated.
kernel void q4a_mm(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    threadgroup bfloat a[1024],b[1024];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    auto total=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    auto subtotal=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<total.get_capacity();++i)total[i]=0;
    uint parts=p[5],lanes=parts>=32 ? 32 : min(parts,8u),span=p[0]/parts;
    // Keep the reference's BF16 partial/join boundary. This sequential
    // per-tile baseline is a correctness seam; parallel split-K is still
    // a performance target, not an established SOTA election.
    for(uint lane=0;lane<lanes;++lane) {
      for(uint i=0;i<subtotal.get_capacity();++i)subtotal[i]=0;
      for(uint part=lane;part<parts;part+=lanes) {
       for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
       for(uint base=part*span;base<(part+1)*span;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint k=base+i%32,row=g.y*32+i/32,col=g.x*32+i/32;
            a[i]=bfloat(row<p[2] ? x[ulong(row)*p[0]+k] : 0);
            b[i]=bfloat(col<p[1] ? q4a_value(w,p[0],p[1],ulong(col)*p[0]+k,p[3],p[4]) : 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
       }
       for(uint i=0;i<subtotal.get_capacity();++i)subtotal[i]=mlx_bf(subtotal[i]+mlx_bf(acc[i]));
      }
      for(uint i=0;i<total.get_capacity();++i)total[i]=parts>=32 ? total[i]+subtotal[i] : mlx_bf(total[i]+subtotal[i]);
    }
    for(auto it=total.begin();it!=total.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*32+ij[0],row=g.y*32+ij[1];
        if(it.is_valid_element() && col<p[1] && row<p[2])y[ulong(row)*p[1]+col]=mlx_bf(*it);
    }
}

// Original packed staging for the single-part dense contraction. Keep the
// baseline's 4 KiB operand footprint and 32x32 compute shape; larger dense
// tiles regressed the complete-model workload despite microbenchmark wins.
template<uint Bits,uint Reuse>
inline void q4a_dense_packed(device const uchar* w,device const float* x,device float* y,
    constant uint* p,uint2 g,uint tid,threadgroup bfloat* a,threadgroup bfloat* b) {
    // Reordered entry is retained only as a GPU test comparator; it did not
    // improve the full-model measurement over the packed baseline.
    if(Reuse>1) {
        uint nx=(p[1]+31)/32,ny=(p[2]+31)/32,index=g.y*nx+g.x;
        uint first=index/(nx*Reuse)*Reuse,height=min(Reuse,ny-first);
        uint local=index-first*nx;
        g=uint2(local/height,first+local%height);
    }
    x+=ulong(p[6])*p[0];y+=ulong(p[6])*p[1];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(p[0])*p[1]*Bits/8);
    device const bfloat* biases=scales+ulong(p[0])*p[1]/(Bits==4 ? 32 : 64);
    for(uint base=0;base<p[0];base+=32) {
        for(uint i=tid*4;i<1024;i+=512) {
            uint row=g.y*32+i/32,col=g.x*32+i/32,k=base+i%32;
            float4 av=row<p[2] ? *reinterpret_cast<device const float4*>(x+ulong(row)*p[0]+k) : float4(0);
            *reinterpret_cast<threadgroup bfloat4*>(a+i)=bfloat4(av);
            float4 bv=0;
            if(col<p[1]) {
                ulong index=ulong(col)*p[0]+k;float4 codes;
                if(Bits==4) {
                    uint code=reinterpret_cast<device const ushort*>(w)[index/4];
                    codes=float4(code&15,(code>>4)&15,(code>>8)&15,code>>12);
                } else {
                    uint code=reinterpret_cast<device const uint*>(w)[index/4];
                    codes=float4(code&255,(code>>8)&255,(code>>16)&255,code>>24);
                }
                ulong group=index/(Bits==4 ? 32 : 64);
                bv=fma(codes,float4(float(scales[group])),float4(float(biases[group])));
            }
            *reinterpret_cast<threadgroup bfloat4*>(b+i)=bfloat4(bv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*32+ij[0],row=g.y*32+ij[1];
        if(it.is_valid_element() && col<p[1] && row<p[2])y[ulong(row)*p[1]+col]=mlx_bf(*it);
    }
}
#define Q4A_DENSE_PACKED(Name,Bits,Reuse) \
kernel void Name(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]], \
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup bfloat a[1024],b[1024];q4a_dense_packed<Bits,Reuse>(w,x,y,p,g,tid,a,b); \
}
Q4A_DENSE_PACKED(q4a_mm4_packed,4,1)
Q4A_DENSE_PACKED(q4a_mm8_packed,8,1)
Q4A_DENSE_PACKED(q4a_mm4_reuse,4,4)
Q4A_DENSE_PACKED(q4a_mm8_reuse,8,4)
#undef Q4A_DENSE_PACKED

// One BF16 input conversion is shared by all output tiles. The checkpoint
// already imposes this rounding on every matrix operand; staging changes only
// reuse, not precision. Pad physical rows so a tensor slice never reads OOB.
kernel void q4a_input(device const float* x [[buffer(0)]],device bfloat* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint padded=(p[2]+31)/32*32;
    if(i<padded*p[0])out[i]=bfloat(i<p[2]*p[0] ? x[ulong(p[6])*p[0]+i] : 0);
}
template<uint Bits,uint BK,bool GroupLoad=false,uint Pad=0,uint BN=32>
inline void q4a_dense_device(device const uchar* w,device bfloat* x,device float* y,
    constant uint* p,uint2 g,uint tid,threadgroup bfloat* b) {
    uint K=p[0],N=p[1],M=p[2];
    auto at=tensor(x,dextents<int,2>(K,(M+31)/32*32),array<int,2>{1,int(K)});
    auto bt=tensor(b,extents<int,BK,BN>(),array<int,2>{1,BK+Pad});
    constexpr auto desc=matmul2d_descriptor(32,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint first=g.y*32;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N*Bits/8);
    device const bfloat* biases=scales+ulong(K)*N/(Bits==4 ? 32 : 64);
    for(uint base=0;base<K;base+=BK) {
        if(GroupLoad) {
            for(uint i=tid*32;i<BN*BK;i+=128*32) {
                uint col=g.x*BN+i/BK,k=base+i%BK;
                ulong index=ulong(col)*K+k;
                uint4 words=0;float scale=0,bias=0;
                if(col<N) {
                    words=*reinterpret_cast<device const uint4*>(w+index/2);
                    scale=float(scales[index/32]);bias=float(biases[index/32]);
                }
                q4a_unpack_group32(words,scale,bias,b+i/BK*(BK+Pad)+i%BK);
            }
        } else for(uint i=tid*4;i<BN*BK;i+=512) {
            uint col=g.x*BN+i/BK,k=base+i%BK;float4 values=0;
            if(col<N) {
                ulong index=ulong(col)*K+k;float4 codes;
                if(Bits==4) {
                    uint code=reinterpret_cast<device const ushort*>(w)[index/4];
                    codes=float4(code&15,(code>>4)&15,(code>>8)&15,code>>12);
                } else {
                    uint code=reinterpret_cast<device const uint*>(w)[index/4];
                    codes=float4(code&255,(code>>8)&255,(code>>16)&255,code>>24);
                }
                ulong group=index/(Bits==4 ? 32 : 64);
                values=fma(codes,float4(float(scales[group])),float4(float(biases[group])));
            }
            *reinterpret_cast<threadgroup bfloat4*>(b+i/BK*(BK+Pad)+i%BK)=bfloat4(values);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input=at.slice<BK,32>(base,first);op.run(input,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    y+=ulong(p[6])*N;
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*BN+ij[0],row=first+ij[1];
        if(it.is_valid_element() && col<N && row<M)y[ulong(row)*N+col]=mlx_bf(*it);
    }
}
#define Q4A_DENSE_DEVICE(Name,Bits,BK,GroupLoad,Pad) \
kernel void Name(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device float* y [[buffer(2)]], \
    constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup bfloat b[32*(BK+Pad)];q4a_dense_device<Bits,BK,GroupLoad,Pad>(w,x,y,p,g,tid,b); }
Q4A_DENSE_DEVICE(q4a_mm4_device128,4,128,false,0)
Q4A_DENSE_DEVICE(q4a_mm8_device128,8,128,false,0)
Q4A_DENSE_DEVICE(q4a_mm4_device64,4,64,false,0)
Q4A_DENSE_DEVICE(q4a_mm8_device64,8,64,false,0)
Q4A_DENSE_DEVICE(q4a_mm4_device128_group32,4,128,true,0)
Q4A_DENSE_DEVICE(q4a_mm4_device64_group32,4,64,true,0)
Q4A_DENSE_DEVICE(q4a_mm4_device128_pad8,4,128,true,8)
Q4A_DENSE_DEVICE(q4a_mm4_device64_pad8,4,64,true,8)
Q4A_DENSE_DEVICE(q4a_mm4_device128_pad16,4,128,true,16)
Q4A_DENSE_DEVICE(q4a_mm4_device64_pad16,4,64,true,16)
#undef Q4A_DENSE_DEVICE

kernel void q4a_mm4_device128_wide(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device float* y [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat b[64*136];q4a_dense_device<4,128,true,8,64>(w,x,y,p,g,tid,b);
}

// A uniform workgroup selects one original packed plane. The matrix shape,
// BF16 operands and K loop are exactly those of the separate dense kernels.
inline uint2 q4a_pair_grid(uint2 g,uint columns,uint rows,uint group) {
    if(group==1)return g;
    uint linear=g.y*columns+g.x,first=(linear/(columns*group))*group;
    uint height=min(group,rows-first),offset=linear%(columns*group);
    return uint2(offset/height,first+offset%height);
}
#define Q4A_PAIR(Name,BN) \
kernel void Name(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]], \
    device bfloat* x [[buffer(2)]],device float* y0 [[buffer(3)]],device float* y1 [[buffer(4)]], \
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup bfloat b[BN*136];g=q4a_pair_grid(g,(p[1]+p[8])/BN,(p[2]+31)/32,p[16]); \
    uint cut=p[1]/BN;bool second=g.x>=cut; \
    if(second)g.x-=cut; \
    q4a_dense_device<4,128,true,8,BN>(second?w1:w0,x,second?y1:y0,second?p+7:p,g,tid,b); }
Q4A_PAIR(q4a_mm4_pair,32)
Q4A_PAIR(q4a_mm4_pair_wide,64)
#undef Q4A_PAIR

#ifdef PADDOCK_KERNEL_DIAGNOSTICS
// Isolated slab gains have not translated into a c=4 whole-model win.
// Keep this experiment out of the production pipeline set.
kernel void q4a_weight_pair_slab(device const uchar* w0 [[buffer(0)]],device const uchar* w1 [[buffer(1)]],
    device bfloat* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint K=p[0];if(i>=p[15]*(K/32))return;
    uint col=p[14]+i/(K/32),k=i%(K/32)*32;bool second=col>=p[1];
    uint N=second?p[8]:p[1];if(second)col-=p[1];
    device const uchar* w=second?w1:w0;
    ulong at=ulong(col)*K+k;
    uint4 words=*reinterpret_cast<device const uint4*>(w+at/2);
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
    float scale=float(scales[at/32]),bias=float(scales[ulong(K)*N/32+at/32]);
    #pragma unroll
    for(uint word=0;word<4;++word) {
        uint codes=words[word];
        float4 lo=float4(codes&15,(codes>>4)&15,(codes>>8)&15,(codes>>12)&15);
        float4 hi=float4((codes>>16)&15,(codes>>20)&15,(codes>>24)&15,codes>>28);
        *reinterpret_cast<device bfloat4*>(out+ulong(i)*32+word*8)=bfloat4(fma(lo,float4(scale),float4(bias)));
        *reinterpret_cast<device bfloat4*>(out+ulong(i)*32+word*8+4)=bfloat4(fma(hi,float4(scale),float4(bias)));
    }
}
kernel void q4a_mm4_pair_slab(device bfloat* w [[buffer(0)]],device bfloat* x [[buffer(1)]],
    device float* y0 [[buffer(2)]],device float* y1 [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]]) {
    uint K=p[0],M=p[2];g=q4a_pair_grid(g,p[15]/64,(M+63)/64,p[16]);
    uint first=p[14]+g.x*64;bool second=first>=p[1];
    uint N=second?p[8]:p[1];if(second)first-=p[1];
    device float* y=(second?y1:y0)+ulong(p[6])*N;
    auto at=tensor(x,dextents<int,2>(K,(M+31)/32*32),array<int,2>{1,int(K)}).slice(0,g.y*64);
    auto bt=tensor(w,dextents<int,2>(K,p[15]),array<int,2>{1,int(K)}).slice(0,g.x*64);
    constexpr auto desc=matmul2d_descriptor(64,64,dynamic_length_v<int>,false,true,false);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    op.run(at,bt,acc);
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=first+ij[0],row=g.y*64+ij[1];
        if(it.is_valid_element() && col<N && g.x*64+ij[0]<p[15] && row<M)y[ulong(row)*N+col]=mlx_bf(*it);
    }
}
#endif

// Decode a bounded output-column slab once for all physical prompt rows.
// The temporary remains in the model's reusable projection arena; packed
// checkpoint weights are unchanged. Pad columns for guarded tensor slices.
kernel void q4a_weight_slab(device const uchar* w [[buffer(0)]],device bfloat* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint K=p[0],N=p[1];
    if(i>=p[8]*(K/32))return;
    uint col=p[7]+i/(K/32),k=i%(K/32)*32;
    ulong at=ulong(col)*K+k;
    uint4 words=0;float scale=0,bias=0;
    if(col<N) {
        words=*reinterpret_cast<device const uint4*>(w+at/2);
        device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N/2);
        scale=float(scales[at/32]);bias=float(scales[ulong(K)*N/32+at/32]);
    }
    #pragma unroll
    for(uint word=0;word<4;++word) {
        uint codes=words[word];
        float4 lo=float4(codes&15,(codes>>4)&15,(codes>>8)&15,(codes>>12)&15);
        float4 hi=float4((codes>>16)&15,(codes>>20)&15,(codes>>24)&15,codes>>28);
        *reinterpret_cast<device bfloat4*>(out+ulong(i)*32+word*8)=bfloat4(fma(lo,float4(scale),float4(bias)));
        *reinterpret_cast<device bfloat4*>(out+ulong(i)*32+word*8+4)=bfloat4(fma(hi,float4(scale),float4(bias)));
    }
}
kernel void q4a_mm4_slab(device bfloat* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device float* y [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]]) {
    uint K=p[0],N=p[1],M=p[2];
    auto at=tensor(x,dextents<int,2>(K,(M+31)/32*32),array<int,2>{1,int(K)}).slice(0,g.y*64);
    auto bt=tensor(w,dextents<int,2>(K,p[8]),array<int,2>{1,int(K)}).slice(0,g.x*64);
    constexpr auto desc=matmul2d_descriptor(64,64,dynamic_length_v<int>,false,true,false);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    op.run(at,bt,acc);
    y+=ulong(p[6])*N;
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=p[7]+g.x*64+ij[0],row=g.y*64+ij[1];
        if(it.is_valid_element() && col<N && col<p[7]+p[8] && row<M)y[ulong(row)*N+col]=mlx_bf(*it);
    }
}

// The same affine group loader for split-K, without changing the immutable
// partition boundaries or the BF16 partial/join contract. Input staging and
// partials occupy disjoint regions of one bounded model-owned workspace.
template<uint BK,uint Pad=0,uint Bits=4>
inline void q4a_split_device(device const uchar* w,device bfloat* x,device bfloat* partial,
    constant uint* p,uint3 g,uint tid,threadgroup bfloat* b) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint K=p[0],N=p[1],M=p[2],span=K/p[5];
    auto at=tensor(x,dextents<int,2>(K,(M+31)/32*32),array<int,2>{1,int(K)});
    auto bt=tensor(b,extents<int,BK,32>(),array<int,2>{1,BK+Pad});
    constexpr auto desc=matmul2d_descriptor(32,32,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    device const bfloat* scales=reinterpret_cast<device const bfloat*>(w+ulong(K)*N*Bits/8);
    device const bfloat* biases=scales+ulong(K)*N/(Bits==4 ? 32 : 64);
    for(uint base=g.z*span;base<(g.z+1)*span;base+=BK) {
        if(Bits==4)for(uint i=tid*32;i<32*BK;i+=128*32) {
            uint col=g.x*32+i/BK,k=base+i%BK;
            ulong index=ulong(col)*K+k;
            uint4 words=0;float scale=0,bias=0;
            if(col<N) {
                words=*reinterpret_cast<device const uint4*>(w+index/2);
                scale=float(scales[index/32]);bias=float(biases[index/32]);
            }
            q4a_unpack_group32(words,scale,bias,b+i/BK*(BK+Pad)+i%BK);
        } else for(uint i=tid*4;i<32*BK;i+=512) {
            uint col=g.x*32+i/BK,k=base+i%BK;float4 values=0;
            if(col<N) {
                ulong index=ulong(col)*K+k;
                uint code=reinterpret_cast<device const uint*>(w)[index/4];
                float4 codes=float4(code&255,(code>>8)&255,(code>>16)&255,code>>24);
                values=fma(codes,float4(float(scales[index/64])),float4(float(biases[index/64])));
            }
            *reinterpret_cast<threadgroup bfloat4*>(b+i/BK*(BK+Pad)+i%BK)=bfloat4(values);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input=at.slice<BK,32>(base,g.y*32);op.run(input,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*32+ij[0],row=g.y*32+ij[1];
        if(it.is_valid_element() && col<N && row<M)
            partial[(ulong(g.z)*M+row)*N+col]=bfloat(*it);
    }
}
#define Q4A_SPLIT_DEVICE(Name,BK,Pad) \
kernel void Name(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device bfloat* partial [[buffer(2)]], \
    constant uint* p [[buffer(3)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup bfloat b[32*(BK+Pad)];q4a_split_device<BK,Pad>(w,x,partial,p,g,tid,b); }
Q4A_SPLIT_DEVICE(q4a_mm4_split_device128_group32,128,0)
Q4A_SPLIT_DEVICE(q4a_mm4_split_device64_group32,64,0)
Q4A_SPLIT_DEVICE(q4a_mm4_split_device32_group32,32,0)
Q4A_SPLIT_DEVICE(q4a_mm4_split_device128_pad8,128,8)
Q4A_SPLIT_DEVICE(q4a_mm4_split_device64_pad8,64,8)
Q4A_SPLIT_DEVICE(q4a_mm4_split_device32_pad8,32,8)
#undef Q4A_SPLIT_DEVICE

#define Q4A_SPLIT8_DEVICE(Name,BK,Pad) \
kernel void Name(device const uchar* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device bfloat* partial [[buffer(2)]], \
    constant uint* p [[buffer(3)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup bfloat b[32*(BK+Pad)];q4a_split_device<BK,Pad,8>(w,x,partial,p,g,tid,b); }
Q4A_SPLIT8_DEVICE(q4a_mm8_split_device128,128,0)
Q4A_SPLIT8_DEVICE(q4a_mm8_split_device64,64,0)
Q4A_SPLIT8_DEVICE(q4a_mm8_split_device128_pad8,128,8)
Q4A_SPLIT8_DEVICE(q4a_mm8_split_device64_pad8,64,8)
#undef Q4A_SPLIT8_DEVICE

// Original parallel split-K: one workgroup owns one immutable partition.
// The join below preserves the sequential path's BF16 partials and reduction
// order exactly. Only 1 MiB of model-owned scratch is needed at most.
kernel void q4a_mm_split(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device bfloat* partial [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    x+=ulong(p[6])*p[0];
    threadgroup bfloat a[1024],b[1024];
    auto at=tensor(a,extents<int,32,32>(),array<int,2>{1,32});
    auto bt=tensor(b,extents<int,32,32>(),array<int,2>{1,32});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(at),decltype(bt),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint span=p[0]/p[5];
    for(uint base=g.z*span;base<(g.z+1)*span;base+=32) {
        for(uint i=tid;i<1024;i+=128) {
            uint k=base+i%32,row=g.y*32+i/32,col=g.x*32+i/32;
            a[i]=bfloat(row<p[2] ? x[ulong(row)*p[0]+k] : 0);
            b[i]=bfloat(col<p[1] ? q4a_value(w,p[0],p[1],ulong(col)*p[0]+k,p[3],p[4]) : 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(at,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=g.x*32+ij[0],row=g.y*32+ij[1];
        if(it.is_valid_element() && col<p[1] && row<p[2])
            partial[(ulong(g.z)*p[2]+row)*p[1]+col]=bfloat(*it);
    }
}

kernel void q4a_mm_join(device const bfloat* partial [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    y+=ulong(p[6])*p[1];
    ulong size=ulong(p[1])*p[2];if(i>=size)return;
    uint parts=p[5],lanes=parts>=32 ? 32 : min(parts,8u);float total=0;
    for(uint lane=0;lane<lanes;++lane) {
        float subtotal=0;
        for(uint part=lane;part<parts;part+=lanes)
            subtotal=mlx_bf(subtotal+float(partial[ulong(part)*size+i]));
        total=parts>=32 ? total+subtotal : mlx_bf(total+subtotal);
    }
    y[i]=mlx_bf(total);
}
