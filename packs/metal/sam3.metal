// Original SAM 3 Metal seams. Arithmetic/layout contract: Paddock's CUDA
// sam3/{vit,neck,image}.cuh and Meta's image/video processors (see sam3/).
// Image and video normalization are deliberately different. No pixel math
// goes through the host, and attention never materializes a score matrix.
kernel void sam3_patch(device const uchar* px [[buffer(0)]], device half* out [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    uint side=p[0],patch=p[1],win=p[2],kp=p[3],grid=side/patch;
    uint row=i/kp,col=i%kp;if(row>=grid*grid)return;
    if(col>=3*patch*patch){out[i]=half(0);return;}
    uint wi=row/(win*win),at=row%(win*win),nw=grid/win;
    uint y=(wi/nw*win+at/win)*patch+(col%(patch*patch))/patch;
    uint x=(wi%nw*win+at%win)*patch+col%patch;
    float b=float(px[(y*side+x)*3+col/(patch*patch)]);
    if(p[4]) {
        // Explicit stores to half retain all three video rounding boundaries.
        half a=half(precise::divide(b,255.f));half d=half(float(a)-0.5f);
        out[i]=half(float(d)*2.f);
    } else {
        float a=b*(1.f/255.f);out[i]=half((a-0.5f)*2.f);
    }
}

// F16 contractions, F32 accumulation. All operation boundaries are explicit:
// 0 raw F32; 1 raw F16; 2 bias/F16; 3 bias+tanh-GELU/F16; 4 bias/F32.
kernel void sam3_mm(device half* w [[buffer(0)]],device half* x [[buffer(1)]],
    device uint* out [[buffer(2)]],device const float* bias [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {
    uint K=p[0],N=p[1],M=p[2],m=g.y*32,n=g.x*64;
    auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
    auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
    constexpr auto desc=matmul2d_descriptor(32,64,dynamic_length_v<int>,false,true,false);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();op.run(a,b,acc);
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint c=n+ij[0],r=m+ij[1];
        if(it.is_valid_element() && r<M && c<N) {
            float v=*it;if(p[3]>=2)v+=bias[c];if(p[3]==3)v=vis_gelu(v);
            ulong ix=ulong(r)*N+c;
            if(p[3]==0 || p[3]==4)reinterpret_cast<device float*>(out)[ix]=v;
            else reinterpret_cast<device half*>(out)[ix]=half(v);
        }
    }
}

kernel void sam3_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]+=pos[i%p[1]];
}

// Two-pass F32 LayerNorm, CUDA's explicit down-shuffle reduction geometry.
// Four owned components per thread stay in registers through both passes.
// Mode bit 0 adds the half projection + F32 bias; bit 1 lands F32 (ln_pre).
kernel void sam3_norm(device float* x [[buffer(0)]],device const half* add [[buffer(1)]],
    device const float* bias [[buffer(2)]],device const float* w [[buffer(3)]],
    device const float* b [[buffer(4)]],device half* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    threadgroup float sums[8];uint D=p[0],lane=tid%32,sg=tid/32;float sum=0,values[4];
    for(uint u=0;u<4;++u){uint c=tid+u*256;ulong i=ulong(row)*D+c;
        float z=x[i];if(p[1]&1){z+=float(add[i])+bias[c];x[i]=z;}values[u]=z;sum+=z;}
    for(uint step=16;step;step/=2)sum+=simd_shuffle_down(sum,step);
    if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=0;for(uint j=0;j<8;++j)mean+=sums[j];mean/=float(D);
    threadgroup_barrier(mem_flags::mem_threadgroup);sum=0;
    for(uint u=0;u<4;++u){float v=values[u]-mean;sum=fma(v,v,sum);}
    for(uint step=16;step;step/=2)sum+=simd_shuffle_down(sum,step);
    if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float var=0;for(uint j=0;j<8;++j)var+=sums[j];float inv=precise::rsqrt(var/float(D)+1e-5f);
    for(uint u=0;u<4;++u){uint c=tid+u*256;ulong i=ulong(row)*D+c;
        float v=fma((values[u]-mean)*inv,w[c],b[c]);
        if(p[1]&2)reinterpret_cast<device float*>(out)[i]=v;else out[i]=half(v);}
}

// The input is unrounded QKV GEMM output, matching CUDA's fused producer
// election. Q/K weights have rotate-half rows; V retains checkpoint order.
kernel void sam3_qkv(device const float* in [[buffer(0)]],device const float* bias [[buffer(1)]],
    device const float2* rope [[buffer(2)]],device half* q [[buffer(3)]],
    device half* k [[buffer(4)]],device half* v [[buffer(5)]],constant uint* p [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    uint row=i/1024,c=i%1024;if(row>=p[0])return;
    uint j=c%64,other=(j+32)%64+c/64*64;float2 a=rope[(row%p[1])*32+j%32];
    float sn=j<32?-a.y:a.y;ulong base=ulong(row)*3072;
    float q0=in[base+c]+bias[c],q1=in[base+other]+bias[other];
    float k0=in[base+1024+c]+bias[1024+c],k1=in[base+1024+other]+bias[1024+other];
    q[i]=half(fma(q1,sn,q0*a.x)*0.125f);k[i]=half(fma(k1,sn,k0*a.x));
    v[i]=half(in[base+2048+c]+bias[2048+c]);
}
kernel void sam3_attention(device half* q [[buffer(0)]],device half* k [[buffer(1)]],
    device half* v [[buffer(2)]],device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<true,64,64,16>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
kernel void sam3_raster(device const float* x [[buffer(0)]],device half* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint r=i/p[2],c=i%p[2],grid=p[0],win=p[1];if(r>=grid*grid)return;
    uint y=r/grid,xx=r%grid,src=(y/win*(grid/win)+xx/win)*win*win+(y%win)*win+xx%win;
    out[i]=half(x[src*p[2]+c]);
}
// Non-overlapping k2/s2 transposed conv: GEMM tap rows -> raster, bias and
// (only the first x4 stage) erf-GELU, then a single half rounding. The shared
// original erf approximation is numerically tested; not claimed CUDA-bit-exact.
kernel void sam3_convt(device const float* x [[buffer(0)]],device const float* bias [[buffer(1)]],
    device half* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint c=i%p[1],r=i/p[1],side=p[0]*2;if(r>=side*side)return;
    uint y=r/side,xx=r%side,t=(y%2)*2+xx%2;
    float v=x[((y/2)*p[0]+xx/2)*4*p[1]+t*p[1]+c]+bias[c];
    if(p[2])v=mv_gelu_value(v);out[i]=half(v);
}
// Bounded implicit k3/s1 conv, tap-major weights: no 364.5 MiB im2row plane.
kernel void sam3_conv3(device half* w [[buffer(0)]],device const half* x [[buffer(1)]],
    device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint side=p[0],C=p[1],m=g.y*32,n=g.x*64,K=9*C;
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    threadgroup half gathered[32*64];
    auto a=tensor(gathered,extents<int,64,32>(),array<int,2>{1,64});
    auto b=tensor(w,dextents<int,2>(K,C),array<int,2>{1,int(K)});
    auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint j=0;j<acc.get_capacity();++j)acc[j]=0;
    for(uint base=0;base<K;base+=64) {
        for(uint i=tid;i<32*64;i+=128) {
            uint r=m+i/64,z=base+i%64,tap=z/C;
            int y=int(r/side)+int(tap/3)-1,xx=int(r%side)+int(tap%3)-1;
            gathered[i]=r<side*side && y>=0 && xx>=0 && y<int(side) && xx<int(side)?x[(uint(y)*side+uint(xx))*C+z%C]:half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto bt=b.slice(base,n);op.run(a,bt,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();uint r=m+ij[1],c=n+ij[0];
        if(r<side*side && c<C)out[r*C+c]=*it+bias[c];
    }
}
