// LightOn's MLX exports retain BF16 vision weights AND operation boundaries.
// F32 storage is shared with the ragged encoder, not permission to remove
// these cuts. GGUF/Bonsai/Splash keep their independently qualified graph.
inline float qmlx_tanh_gelu(float x) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    float cube=mlx_bf(pow(x,3.0f));
    float z=mlx_bf(x+mlx_bf(mlx_bf(0.044715f)*cube));
    z=mlx_bf(mlx_bf(0.7978845608028654f)*z);
    z=mlx_bf(1.0f+mlx_bf(precise::tanh(clamp(z,-10.0f,10.0f))));
    return mlx_bf(mlx_bf(0.5f*x)*z);
}

kernel void qmlx_vis_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint n=p[0],threads=((n+255)/256)*32,lane=tid%32;ulong base=ulong(row)*n;
    threadgroup float sums[32];float values[8],total=0;
    for(uint i=0;i<8;++i){uint col=tid*8+i;values[i]=col<n?x[base+col]:0.0f;total+=values[i];}
    total=simd_sum(total);if(lane==0)sums[tid/32]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=precise::divide(simd_sum(lane<threads/32?sums[lane]:0.0f),float(n));
    threadgroup_barrier(mem_flags::mem_threadgroup);total=0;
    // The reference variance is a fused sum of squares. The mean and final
    // affine BF16 boundaries above/below stay separate.
    for(uint i=0;i<8;++i){values[i]=tid*8+i<n?values[i]-mean:0.0f;total=fma(values[i],values[i],total);}
    total=simd_sum(total);if(lane==0)sums[tid/32]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var=precise::divide(simd_sum(lane<threads/32?sums[lane]:0.0f),float(n));
    float inv=precise::rsqrt(var+as_type<float>(p[1]));
    for(uint i=0;i<8;++i){uint col=tid*8+i;if(col<n)out[base+col]=mlx_bf(fma(mlx_bf(values[i]*inv),w[col],bias[col]));}
}

kernel void qmlx_vis_mm(device bfloat* w [[buffer(0)]],device float* x [[buffer(1)]],
    device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {
    uint K=p[0],N=p[1],M=p[2],m=g.y*64,n=g.x*64;
    auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
    auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
    constexpr auto desc=matmul2d_descriptor(64,64,dynamic_length_v<int>,false,true,true);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    op.run(a,b,acc);
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
        if(it.is_valid_element() && row<M && col<N) {
            ulong at=ulong(row)*N+col;
            float z=mlx_bf(*it+bias[col]);
            if(p[3]==2)z=mlx_bf(z+out[at]);
            if(p[3]==3)z=qmlx_tanh_gelu(z);
            if(p[3]==4)z=gmlx_erf_activation(z);
            out[at]=z;
        }
    }
}

kernel void qmlx_vis_patches(device const uchar* rgb [[buffer(0)]],device bfloat* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    // CPU Pillow has already produced this exact grid. Keep the contraction
    // operands BF16 as well as its values: mixed F16/BF16 MPP contractions
    // do not guarantee the same arithmetic as the BF16 reference convolution.
    uint pw=p[2]/16,ph=p[3]/16,row=i/1536,d=i%768;if(row>=pw*ph)return;
    uint py=(row/4/(pw/2))*2+(row%4)/2,px=(row/4%(pw/2))*2+row%2;
    uint x=px*16+(d/3)%16,y=py*16+d/48,c=d%3;
    // MLX-VLM's NumPy processor multiplies F32 by a rounded reciprocal,
    // then normalizes separately. FMA or replacing this with division changes
    // BF16 rounding for central gray values. Preserve both F32 boundaries.
    float v=float(rgb[(ulong(y)*p[2]+x)*3+c])*(1.0f/255.0f);
    out[ulong(p[4])*1536+i]=bfloat((v-0.5f)*2.0f);
}

kernel void qmlx_vis_patch_mm(device bfloat* w [[buffer(0)]],device bfloat* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    // Conv3d's RGB channels are padded to 16 by the reference. Its 8x8
    // accumulator advances once per spatial position. Flattening RGB into
    // a dense MPP GEMM changes rounding at BF16 ties on real OCR pages.
    // Reproduce that reduction without allocating padded weights/pixels:
    // the second all-zero 8-channel contraction is an identity and omitted.
    uint lane=tid%32,rr=((lane&16)>>2)|((lane&6)>>1),cc=((lane&8)>>1)|((lane&1)<<1);
    uint row=g.y*64+(tid/32)*16+rr,col=g.x*64+cc,N=p[0],M=p[1];
    simdgroup_float8x8 acc[2][8];
    #pragma clang loop unroll(full)
    for(short i=0;i<2;++i){
        #pragma clang loop unroll(full)
        for(short j=0;j<8;++j)acc[i][j]=make_filled_simdgroup_matrix<float,8>(0.0f);
    }
    for(uint pixel=0;pixel<512;++pixel){
        simdgroup_float8x8 a[2],b;
        #pragma clang loop unroll(full)
        for(short i=0;i<2;++i){
            a[i].thread_elements()[0]=row+i*8<M && cc<3?float(x[ulong(row+i*8)*1536+pixel*3+cc]):0.0f;
            a[i].thread_elements()[1]=row+i*8<M && cc+1<3?float(x[ulong(row+i*8)*1536+pixel*3+cc+1]):0.0f;
        }
        #pragma clang loop unroll(full)
        for(short j=0;j<8;++j){
            b.thread_elements()[0]=col+j*8<N && rr<3?float(w[ulong(col+j*8)*1536+pixel*3+rr]):0.0f;
            b.thread_elements()[1]=col+j*8+1<N && rr<3?float(w[ulong(col+j*8+1)*1536+pixel*3+rr]):0.0f;
            #pragma clang loop unroll(full)
            for(short i=0;i<2;++i)simdgroup_multiply_accumulate(acc[i][j],a[i],b,acc[i][j]);
        }
    }
    #pragma clang loop unroll(full)
    for(short i=0;i<2;++i){
        #pragma clang loop unroll(full)
        for(short j=0;j<8;++j){
            if(row+i*8<M && col+j*8<N)out[ulong(row+i*8)*N+col+j*8]=acc[i][j].thread_elements()[0];
            if(row+i*8<M && col+j*8+1<N)out[ulong(row+i*8)*N+col+j*8+1]=acc[i][j].thread_elements()[1];
        }
    }
}

kernel void qmlx_vis_position(device float* x [[buffer(0)]],device const float* temporal [[buffer(1)]],
    device const float* bias [[buffer(2)]],device const float* pos [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint pw=p[0],ph=p[1],e=p[3],s=p[4],row=i/e,d=i%e;if(row>=pw*ph)return;
    uint y=(row/4/(pw/2))*2+(row%4)/2,xx=(row/4%(pw/2))*2+row%2;
    // Match linspace's divide-then-multiply boundaries. Hoisting (s-1)/(ph-1)
    // into a single scale moves BF16 interpolation ties on real document grids.
    float sy=precise::divide(float(y),float(ph-1))*float(s-1);
    float sx=precise::divide(float(xx),float(pw-1))*float(s-1);
    uint y0=uint(sy),x0=uint(sx),y1=min(y0+1,s-1),x1=min(x0+1,s-1);
    float dy=sy-y0,dx=sx-x0;
    float a=mlx_bf(pos[(y0*s+x0)*e+d]*mlx_bf((1-dy)*(1-dx)));
    float b=mlx_bf(pos[(y0*s+x1)*e+d]*mlx_bf((1-dy)*dx));
    float c=mlx_bf(pos[(y1*s+x0)*e+d]*mlx_bf(dy*(1-dx)));
    float dd=mlx_bf(pos[(y1*s+x1)*e+d]*mlx_bf(dy*dx));
    float position=mlx_bf(mlx_bf(mlx_bf(a+b)+c)+dd);
    ulong at=ulong(p[2])*e+i;
    x[at]=mlx_bf(mlx_bf(mlx_bf(x[at])+bias[d])+position);
}

#define QMLX_QKV(HEADS) \
kernel void qmlx_vis_qkv_##HEADS(device const float* x [[buffer(0)]],device const uint2* xy [[buffer(1)]], \
device bfloat* q [[buffer(2)]],device bfloat* k [[buffer(3)]],device bfloat* v [[buffer(4)]], \
constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) { \
    _Pragma("clang fp contract(off)") \
    _Pragma("clang fp reassociate(off)") \
    uint row=i/(HEADS*64),h=i/64%HEADS,d=i%64;if(row>=p[0]+64)return; \
    float a=0,b=0,c=0;if(row<p[0]) { \
        ulong at=ulong(row)*HEADS*192+h*64;uint pair=d%32,other=(d+32)%64; \
        float inv=precise::divide(1.0f,precise::pow(10000.0f,float(pair%16)/16.0f)); \
        float theta=float(pair<16?xy[row].y:xy[row].x)*inv; \
        float cs=precise::cos(theta),sn=precise::sin(theta)*(d<32?-1.0f:1.0f); \
        a=x[at+d]*cs+x[at+other]*sn;b=x[at+HEADS*64+d]*cs+x[at+HEADS*64+other]*sn;c=x[at+HEADS*128+d]; \
    }q[i]=bfloat(a);k[i]=bfloat(b);v[i]=bfloat(c); }
QMLX_QKV(12)
QMLX_QKV(16)
#undef QMLX_QKV

// Explicit register operands keep the M5 tensor contractions on their native
// 16x32x16 tile. Q/K/V use bounded register fragments to avoid register-pressure
// spills. F32 probabilities and accumulators are never rounded to BF16.
// Scratch is O(tile), not O(image^2), and descriptors isolate ragged images.
// Complete tiles use fixed extents: avoid dynamic bounds on every fragment
// load. Ragged query/key tails keep the bounded implementation and identical
// F32 softmax/reduction order. No probability quantization or extra workspace.
template<uint Heads,bool Aligned=false>
__attribute__((always_inline)) inline void qmlx_attention(device bfloat* q,device bfloat* k,device bfloat* v,
    device float* out,device const uint4* tiles,uint2 g,uint tid,
    threadgroup float* scratch) {
    uint4 t=tiles[g.y];uint first=tid/32*16;if(first>=t.y)return;
    uint count=Aligned?16:min(16u,t.y-first),head=g.x;
    auto tq=tensor(q+ulong(t.x+first)*Heads*64+head*64,dextents<int,2>(64,count),array<int,2>{1,Heads*64});
    auto tk=tensor(k+ulong(t.z)*Heads*64+head*64,dextents<int,2>(64,t.w),array<int,2>{1,Heads*64});
    auto tv=tensor(v+ulong(t.z)*Heads*64+head*64,dextents<int,2>(64,t.w),array<int,2>{1,Heads*64});
    constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pvd=matmul2d_descriptor(16,32,32,false,false,true,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroup> qk;
    matmul2d<pvd,execution_simdgroup> pv;
    auto qr=qk.get_left_input_cooperative_tensor<bfloat,bfloat,float>();
    auto kr=qk.get_right_input_cooperative_tensor<bfloat,bfloat,float>();
    auto sc=qk.get_destination_cooperative_tensor<decltype(qr),decltype(kr),float>();
    auto pr=pv.get_left_input_cooperative_tensor<float,bfloat,float>();
    auto vr=pv.get_right_input_cooperative_tensor<float,bfloat,float>();
    auto a0=pv.get_destination_cooperative_tensor<decltype(pr),decltype(vr),float>();
    auto a1=pv.get_destination_cooperative_tensor<decltype(pr),decltype(vr),float>();
    for(uint i=0;i<a0.get_capacity();++i){a0[i]=0;a1[i]=0;}
    auto maximum=qk.get_row_reduction_destination_cooperative_tensor<decltype(qr),decltype(kr),float>();
    auto den=qk.get_row_reduction_destination_cooperative_tensor<decltype(qr),decltype(kr),float>();
    auto hi=qk.get_row_reduction_destination_cooperative_tensor<decltype(qr),decltype(kr),float>();
    auto sum=qk.get_row_reduction_destination_cooperative_tensor<decltype(qr),decltype(kr),float>();
    auto old=qk.get_row_reduction_destination_cooperative_tensor<decltype(qr),decltype(kr),float>();
    for(uint i=0;i<maximum.get_capacity();++i){maximum[i]=-INFINITY;den[i]=0;}
    auto plane=tensor(scratch+first*32,extents<int,32,16>());
    for(uint base=0;base<t.w;base+=32) {
        for(uint i=0;i<sc.get_capacity();++i)sc[i]=0;
        #pragma unroll
        for(uint d=0;d<4;++d){
            if constexpr(Aligned){
                auto qa=tensor(q+(ulong(t.x+first)*Heads+head)*64,extents<int,64,16>(),array<int,2>{1,Heads*64});
                auto ka=tensor(k+(ulong(t.z+base)*Heads+head)*64,extents<int,64,32>(),array<int,2>{1,Heads*64});
                qr.load(qa.slice(d*16,0));kr.load(ka.slice(d*16,0));
            }else{qr.load(tq.slice(d*16,0));kr.load(tk.slice(d*16,base));}
            qk.run(qr,kr,sc);
        }
        for(auto it=sc.begin();it!=sc.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();*it=Aligned || (ij[0]+base<t.w && ij[1]<count)?*it*(0.125f*1.44269504089f):-INFINITY;
        }
        reduce_rows(sc,hi,reduction_operation::max,-INFINITY);
        for(uint i=0;i<hi.get_capacity();++i){hi[i]=max(hi[i],maximum[i]);old[i]=exp2(maximum[i]-hi[i]);maximum[i]=hi[i];}
        for(auto it=sc.begin();it!=sc.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();*it=Aligned || (ij[0]+base<t.w && ij[1]<count)?exp2(*it-*hi.map_iterator(it)):0.0f;
        }
        reduce_rows(sc,sum);
        for(uint i=0;i<den.get_capacity();++i)den[i]=den[i]*old[i]+sum[i];
        for(auto it=a0.begin();it!=a0.end();++it)if(it.is_valid_element())*it*=*old.map_iterator(it);
        for(auto it=a1.begin();it!=a1.end();++it)if(it.is_valid_element())*it*=*old.map_iterator(it);
        auto load_v=[&](uint channel){
            if constexpr(Aligned){
                auto va=tensor(v+(ulong(t.z+base)*Heads+head)*64,extents<int,64,32>(),array<int,2>{1,Heads*64});
                vr.load(va.slice(channel,0));
            }else{vr.load(tv.slice(channel,base));}
        };
        if(pv.is_compatible_as_left_input<float,bfloat,float>(sc)) {
            auto probs=pv.get_left_input_cooperative_tensor<float,bfloat,float>(sc);
            load_v(0);pv.run(probs,vr,a0);
            load_v(32);pv.run(probs,vr,a1);
        } else {
            sc.store(plane);simdgroup_barrier(mem_flags::mem_threadgroup);pr.load(plane);
            load_v(0);pv.run(pr,vr,a0);
            load_v(32);pv.run(pr,vr,a1);
        }
    }
    for(auto it=a0.begin();it!=a0.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();if(ij[1]<count)out[(ulong(t.x+first+ij[1])*Heads+head)*64+ij[0]]=mlx_bf(*it*(1.0f/ *den.map_iterator(it)));
    }
    for(auto it=a1.begin();it!=a1.end();++it)if(it.is_valid_element()) {
        auto ij=it.get_multidimensional_index();if(ij[1]<count)out[(ulong(t.x+first+ij[1])*Heads+head)*64+32+ij[0]]=mlx_bf(*it*(1.0f/ *den.map_iterator(it)));
    }
}

#ifdef PADDOCK_APPLE9
#define QMLX_ELECTED(HEADS,ALIGNED) qmlx_attention<HEADS,ALIGNED>(q,k,v,out,tiles,g,tid,remap)
#else
#define QMLX_ELECTED(HEADS,ALIGNED) qmlx_nax::attention<HEADS,ALIGNED>(q,k,v,out,t,g.x,tid)
#endif
#define QMLX_ATTN(HEADS) \
[[max_total_threads_per_threadgroup(128)]] kernel void qmlx_vis_attention_##HEADS(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],device bfloat* v [[buffer(2)]], \
device float* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]], \
uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float remap[64*32]; \
uint4 t=tiles[g.y]; \
if(t.y%16==0 && t.w%32==0)QMLX_ELECTED(HEADS,true); \
else QMLX_ELECTED(HEADS,false); }
QMLX_ATTN(12)
QMLX_ATTN(16)
#undef QMLX_ATTN

#ifdef PADDOCK_KERNEL_DIAGNOSTICS
#ifdef PADDOCK_APPLE9
#define QMLX_CANDIDATE(HEADS,ALIGNED) qmlx_attention<HEADS,ALIGNED>(q,k,v,out,tiles,g,tid,remap)
#else
#define QMLX_CANDIDATE(HEADS,ALIGNED) qmlx_nax::attention<HEADS,ALIGNED>(q,k,v,out,t,g.x,tid)
#endif
// Diagnostic aliases retained for comparison with the earlier test-only
// attention captures. Release builds contain only the ordinary entry points.
#define QMLX_CANDIDATE_ATTN(HEADS,NAME,BOUNDED) \
[[max_total_threads_per_threadgroup(128)]] kernel void NAME( \
device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],device bfloat* v [[buffer(2)]], \
device float* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]], \
uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float remap[64*32];uint4 t=tiles[g.y]; \
if(!BOUNDED && t.y%16==0 && t.w%32==0)QMLX_CANDIDATE(HEADS,true); \
else QMLX_CANDIDATE(HEADS,false); }
QMLX_CANDIDATE_ATTN(12,qmlx_vis_attention_candidate_12,false)
QMLX_CANDIDATE_ATTN(16,qmlx_vis_attention_candidate_16,false)
QMLX_CANDIDATE_ATTN(12,qmlx_vis_attention_candidate_bounded_12,true)
QMLX_CANDIDATE_ATTN(16,qmlx_vis_attention_candidate_bounded_16,true)
#undef QMLX_CANDIDATE_ATTN
#undef QMLX_CANDIDATE
#define QMLX_BOUNDED_ATTN(HEADS) \
[[max_total_threads_per_threadgroup(128)]] kernel void qmlx_vis_attention_bounded_##HEADS( \
device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],device bfloat* v [[buffer(2)]], \
device float* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]], \
uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float remap[64*32];uint4 t=tiles[g.y];QMLX_ELECTED(HEADS,false); }
QMLX_BOUNDED_ATTN(12)
QMLX_BOUNDED_ATTN(16)
#undef QMLX_BOUNDED_ATTN
#endif
#undef QMLX_ELECTED
