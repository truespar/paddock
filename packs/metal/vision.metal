// Native M5 Qwen ViT. MPP contracts native BF16 weights with F32 activations;
// patches/QKV use F16. F32 residuals survive between blocks. No N x N score plane.
// Algorithm references: Apple MPP guide (2026-03), FlashAttention online
// softmax, Qwen3.8's published vision graph. All kernels are original.
kernel void vis_inject(device const float* image [[buffer(0)]],device const uint* meta [[buffer(1)]],
                       device float* x [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p[0],d=i%p[0];if(row>=p[1])return;
    uint pos=meta[row*2+1];if(meta[row*2]==p[2] && pos>=p[3] && pos<p[3]+p[4])x[i]=image[ulong(pos-p[3])*p[0]+d];
}
kernel void vis_cast(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                     device atomic_uint* bad [[buffer(2)]], constant uint* p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if(i>=p[0])return;
    float v=weight(src,p[1],i);
    if(!isfinite(v) || (p[2]==1 && abs(v)>65504.0f))atomic_store_explicit(bad,1u,memory_order_relaxed);
    if(p[2]==1)reinterpret_cast<device half*>(dst)[i]=half(v);
    else if(p[2]==30)reinterpret_cast<device bfloat*>(dst)[i]=bfloat(v);
    else reinterpret_cast<device float*>(dst)[i]=v;
}

kernel void vis_finite(device const float* x [[buffer(0)]],device atomic_uint* bad [[buffer(1)]],
                       constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0] && !isfinite(x[i]))atomic_store_explicit(bad,1u,memory_order_relaxed);
}

// Small GPU arithmetic oracle for layout tests, never selected by serving.
kernel void vis_attention_check(device const half* q [[buffer(0)]],device const half* k [[buffer(1)]],
                                device const half* v [[buffer(2)]],device float* out [[buffer(3)]],
                                device const uint2* bounds [[buffer(4)]],constant uint* p [[buffer(5)]],
                                uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g.y,h=g.x,hd=p[0],heads=p[1],pad=p[2];float acc[3]={0,0,0},maximum=-INFINITY,den=0;
    for(uint t=bounds[row].x;t<bounds[row].y;++t) {
        float score=0;for(uint d=lane;d<hd;d+=32)score+=float(q[(ulong(row)*heads+h)*pad+d])*float(k[(ulong(t)*heads+h)*pad+d]);
        score=simd_sum(score)/sqrt(float(hd));float next=max(score,maximum),old=exp(maximum-next),pr=exp(score-next);den=den*old+pr;maximum=next;
        for(uint j=0;j<3;++j){uint d=lane+j*32;if(d<hd)acc[j]=acc[j]*old+pr*float(v[(ulong(t)*heads+h)*pad+d]);}
    }
    for(uint j=0;j<3;++j){uint d=lane+j*32;if(d<hd)out[(ulong(row)*heads+h)*hd+d]=acc[j]/den;}
}

inline float vis_gelu(float x) {
    // Metal relaxed tanh overflows internally for large positive arguments
    // (observed at real ViT FFN inputs 11.8..34). Outside [-10,10] its
    // mathematical result already rounds to +/-1 in F32. Clamp the argument,
    // not the activation: positive outliers must retain their full magnitude.
    float a=0.7978845608028654f*x*(1.0f+0.044715f*x*x);
    return 0.5f*x*(1.0f+precise::tanh(clamp(a,-10.0f,10.0f)));
}

// Producer-side epilogues: 0 raw F32; 1 +bias F32; 2 (+bias)+residual
// F32; 3 bias+GELU in the activation type. Ragged N/K are dynamic tensor
// extents, never a static slice that can read the following matrix row.
template<uint BM,typename T,typename X,bool Relaxed=false,bool BF16Output=false>
inline void vis_project(device T* w,device X* x,device uint* out,
                        device const float* bias,constant uint* p,uint2 g) {
    uint K=p[0],N=p[1],M=p[2],m=g.y*BM,n=g.x*64;
    auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
    auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
    constexpr auto desc=matmul2d_descriptor(BM,64,dynamic_length_v<int>,false,true,Relaxed);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    op.run(a,b,acc);
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
        if(it.is_valid_element() && row<M && col<N) {
            float v=*it;if(p[3])v+=bias[col];
            ulong ix=ulong(row)*N+col;
            if(p[3]==3)reinterpret_cast<device X*>(out)[ix]=X(vis_gelu(v));
            else {
                if(p[3]==2)v+=reinterpret_cast<device float*>(out)[ix];
                reinterpret_cast<device float*>(out)[ix]=BF16Output?float(bfloat(v)):v;
            }
        }
    }
}
#define VIS_MM(NAME,BM,T,X) \
kernel void NAME(device T* w [[buffer(0)]],device X* x [[buffer(1)]], \
                       device uint* out [[buffer(2)]],device const float* bias [[buffer(3)]], \
                       constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {vis_project<BM,T,X>(w,x,out,bias,p,g);}
VIS_MM(vis_mm32,32,half,half)
VIS_MM(vis_mm64,64,half,half)
// Native F16 checkpoints share the F32 residual/norm graph. Only the patch
// producer supplies F16 activations; interpreting later F32 buffers as half
// corrupts both the matrix input and the fused GELU output stride.
VIS_MM(vis_hmm32,32,half,float)
VIS_MM(vis_hmm64,64,half,float)
VIS_MM(vis_bmm32,32,bfloat,float)
VIS_MM(vis_bmm64,64,bfloat,float)
#undef VIS_MM
// Distinct multiplication contract, not a silent change to vis_bmm's strict
// identity semantics. M5 same-weights tensor captures elect this mode for
// vision: first QKV samples agree within the reference's printed precision.
// Do not infer the hardware's internal operand dtype from this flag alone.
#define VIS_FAST(BM) \
kernel void vis_bmm_fast##BM(device bfloat* w [[buffer(0)]],device float* x [[buffer(1)]], \
                          device uint* out [[buffer(2)]],device const float* bias [[buffer(3)]], \
                          constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) { \
    vis_project<BM,bfloat,float,true>(w,x,out,bias,p,g); }
VIS_FAST(32)
VIS_FAST(64)
#undef VIS_FAST

// Upload RGB bytes only. Resize uses the mtmd align-corners/truncating-u8
// convention, including centered black PAD_CEIL, then normalize+patchify
// directly into F16. p: source W/H, target W/H, first patch row; mean/std bits.
kernel void vis_patches(device const uchar* rgb [[buffer(0)]],device half* patches [[buffer(1)]],
                        constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint pw=p[2]/16,ph=p[3]/16,row=i/768,d=i%768;if(row>=pw*ph)return;
    uint py=(row/4/(pw/2))*2+(row%4)/2,px=(row/4%(pw/2))*2+row%2;
    uint tx=px*16+d%16,ty=py*16+(d%256)/16,c=d/256;
    float scale=min(float(p[2])/p[0],float(p[3])/p[1]);
    uint nw=clamp(uint(ceil(float(p[0])*scale)),1u,p[2]);
    uint nh=clamp(uint(ceil(float(p[1])*scale)),1u,p[3]);
    uint ox=(p[2]-nw)/2,oy=(p[3]-nh)/2;float v=0;
    if(tx>=ox && tx<ox+nw && ty>=oy && ty<oy+nh) {
        float sx=float(tx-ox)*(nw>1?float(p[0]-1)/float(nw-1):0.0f);
        float sy=float(ty-oy)*(nh>1?float(p[1]-1)/float(nh-1):0.0f);
        uint x0=min(uint(sx),p[0]-1),y0=min(uint(sy),p[1]-1),x1=min(x0+1,p[0]-1),y1=min(y0+1,p[1]-1);
        float a=rgb[(ulong(y0)*p[0]+x0)*3+c],b=rgb[(ulong(y0)*p[0]+x1)*3+c];
        float e=rgb[(ulong(y1)*p[0]+x0)*3+c],f=rgb[(ulong(y1)*p[0]+x1)*3+c];
        float top=fma(b-a,sx-x0,a),bottom=fma(f-e,sx-x0,e);
        v=float(uchar(clamp(fma(bottom-top,sy-y0,top),0.0f,255.0f)));
    }
    patches[ulong(p[4])*768+i]=half((v/255.0f-as_type<float>(p[5+c]))/as_type<float>(p[8+c]));
}

// Dual temporal patch projections stay separate (no rounded weight sum).
// Current Qwen/mtmd learned-position interpolation is ALIGN_CORNERS.
kernel void vis_position(device float* x [[buffer(0)]],device const float* temporal [[buffer(1)]],
                         device const float* bias [[buffer(2)]],device const float* pos [[buffer(3)]],
                         constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    uint pw=p[0],ph=p[1],e=p[3],s=p[4],row=i/e,d=i%e;if(row>=pw*ph)return;
    uint y=(row/4/(pw/2))*2+(row%4)/2,xx=(row/4%(pw/2))*2+row%2;
    float sy=float(y)*float(s-1)/float(ph-1),sx=float(xx)*float(s-1)/float(pw-1);
    uint y0=uint(sy),x0=uint(sx),y1=min(y0+1,s-1),x1=min(x0+1,s-1);
    float a=pos[(y0*s+x0)*e+d],b=pos[(y0*s+x1)*e+d],c=pos[(y1*s+x0)*e+d],dd=pos[(y1*s+x1)*e+d];
    float top=a+(b-a)*(sx-x0),bot=c+(dd-c)*(sx-x0);
    ulong dst=ulong(p[2])*e+i;x[dst]=((x[dst]+temporal[dst])+bias[d])+(top+(bot-top)*(sy-y0));
}

// Two-pass variance avoids catastrophic cancellation on nearly constant
// rows. Keep affine results in F32; mixed contractions consume them directly
// with native BF16 weights, without an intervening activation quantization.
kernel void vis_ln(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
                   device const float* b [[buffer(2)]],device float* out [[buffer(3)]],
                   constant uint* p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[8];uint lane=tid%32,sg=tid/32;float sum=0;
    for(uint d=tid;d<p[0];d+=256)sum+=x[ulong(row)*p[0]+d];
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=0;for(uint j=0;j<8;++j)mean+=sums[j];mean/=float(p[0]);sum=0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<p[0];d+=256){float v=x[ulong(row)*p[0]+d]-mean;sum+=v*v;}
    sum=simd_sum(sum);if(lane==0)sums[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    float var=0;for(uint j=0;j<8;++j)var+=sums[j];float inv=rsqrt(var/float(p[0])+as_type<float>(p[1]));
    for(uint d=tid;d<p[0];d+=256){ulong ix=ulong(row)*p[0]+d;out[ix]=(x[ix]-mean)*inv*w[d]+b[d];}
}

// Fused QKV stays fused on disk and through GEMM. Each thread writes a
// disjoint rotated component; there is no in-place partner read/write race.
// Pad head72 to80 for matrix hardware, zero including the ragged row guard.
kernel void vis_qkv(device const float* x [[buffer(0)]],device const uint2* xy [[buffer(1)]],
                    device half* q [[buffer(2)]],device half* k [[buffer(3)]],device half* v [[buffer(4)]],
                    constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    uint row=i/(16*80),h=i/80%16,d=i%80;if(row>=p[0]+64)return;
    float qv=0,kv=0,vv=0;
    if(row<p[0] && d<72) {
        ulong src=ulong(row)*3456+h*72;uint pair=d%36,other=(d+36)%72;
        // Evaluate each frequency directly. Repeated multiplication rounds
        // once per preceding frequency and perturbs Q/K before the F16 cut.
        float theta=float(pair<18?xy[row].y:xy[row].x)*pow(10000.0f,-float(pair%18)/18.0f);
        float cs=cos(theta),sn=sin(theta)*(d<36?-1.0f:1.0f);
        qv=x[src+d]*cs+x[src+other]*sn;kv=x[src+1152+d]*cs+x[src+1152+other]*sn;vv=x[src+2304+d];
    }
    q[i]=half(qv);k[i]=half(kv);v[i]=half(vv);
}

// The smaller LightOn/Qwen towers have head64 and need no padded channels.
// Specialize the head count so addressing/division is compile-time constant.
template<uint Heads>
inline void vis_qkv64_impl(device const float* x,device const uint2* xy,
                          device half* q,device half* k,device half* v,
                          constant uint* p,uint i) {
    uint row=i/(Heads*64),h=i/64%Heads,d=i%64;if(row>=p[0]+64)return;
    float qv=0,kv=0,vv=0;
    if(row<p[0]) {
        ulong src=ulong(row)*Heads*64*3+h*64;uint pair=d%32,other=(d+32)%64;
        float theta=float(pair<16?xy[row].y:xy[row].x)*pow(10000.0f,-float(pair%16)/16.0f);
        float cs=cos(theta),sn=sin(theta)*(d<32?-1.0f:1.0f);
        qv=x[src+d]*cs+x[src+other]*sn;
        kv=x[src+Heads*64+d]*cs+x[src+Heads*64+other]*sn;
        vv=x[src+Heads*128+d];
    }
    q[i]=half(qv);k[i]=half(kv);v[i]=half(vv);
}
#define VIS_QKV64(HEADS) \
kernel void vis_qkv64_##HEADS(device const float* x [[buffer(0)]],device const uint2* xy [[buffer(1)]], \
device half* q [[buffer(2)]],device half* k [[buffer(3)]],device half* v [[buffer(4)]], \
constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {vis_qkv64_impl<HEADS>(x,xy,q,k,v,p,i);}
VIS_QKV64(12)
VIS_QKV64(16)
#undef VIS_QKV64

// Each tile is (first query,count,image first,image length). Ragged images
// share projections but never an attention domain. Unnormalized output stays
// in cooperative registers until the final division, not in device memory.
template<bool Gemma,uint HD=72,uint Padded=80,uint Heads=16,bool PadTail=false,bool Sam=false,typename T=half,bool Shaw=false,bool Causal=false,uint KVHeads=Heads,bool BF16Scaled=false,uint KeyTile=64>
inline void vis_attention_impl(device T* q,device T* k,device T* v,device ushort* out,
                          device const uint4* tiles,constant uint* p,uint2 g,uint tid,
                          threadgroup float* correction,threadgroup float* normalizer,threadgroup float* remap,
                          device const float* rh=nullptr,device const float* rw=nullptr,uint side=0,uint window=0) {
    // Split-Q ownership: each SIMD group owns 16 complete query rows.
    // Compatible score/probability layouts stay in registers; otherwise an
    // 8 KiB SIMD-private remap preserves F32 probabilities between the two
    // contractions. Row statistics use another 256 bytes. Neither path
    // synchronizes across SIMD groups (Apple MPP / WWDC 2026 TensorOps).
    uint4 t=tiles[g.y];uint head=g.x,first=(tid/32)*16;
    if(first>=t.y)return;
    uint count=min(16u,t.y-first);
    auto tq=tensor(q+ulong(t.x+first)*(Heads*Padded)+head*Padded,dextents<int,2>(Padded,count),array<int,2>{1,Heads*Padded});
    uint storage_rows=PadTail?((t.w+63)/64)*64:t.w;
    uint kvhead=head/(Heads/KVHeads);
    auto tk=tensor(k+ulong(t.z)*(KVHeads*Padded)+kvhead*Padded,dextents<int,2>(Padded,storage_rows),array<int,2>{1,KVHeads*Padded});
    auto tv=tensor(v+ulong(t.z)*(KVHeads*Padded)+kvhead*Padded,dextents<int,2>(Padded,storage_rows),array<int,2>{1,KVHeads*Padded});
    constexpr auto qkd=matmul2d_descriptor(16,KeyTile,Padded,false,true,false);
    constexpr auto pvd=matmul2d_descriptor(16,Padded,KeyTile,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroup> qk;
    matmul2d<pvd,execution_simdgroup> pv;
    auto sc=qk.template get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto qs=qk.template get_left_input_cooperative_tensor<T,T,float>();
    if constexpr(BF16Scaled) {
        // Reference BF16 SDPA rounds the base-2 scale and scaled query before
        // contraction. Only the aligner's explicit specialization opts in.
        qs.load(tq);
        T scale=T((HD==64?0.125f:0.08838834764831844f)*1.44269504089f);
        for(uint i=0;i<qs.get_capacity();++i)qs[i]=T(qs[i]*scale);
    }
    auto maximum=qk.template get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto denominator=qk.template get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto high=qk.template get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto sum=qk.template get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    auto old=qk.template get_row_reduction_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
    for(uint i=0;i<maximum.get_capacity();++i){maximum[i]=-INFINITY;denominator[i]=0;}
    auto pr=pv.template get_left_input_cooperative_tensor<float,T,float>();
    auto acc=pv.template get_destination_cooperative_tensor<decltype(pr),decltype(tv),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    auto tc=tensor(correction+first,extents<int,16>());
    auto tn=tensor(normalizer+first,extents<int,16>());
    uint key_first=window?(uint(max(0,int(t.x+first)-int(t.z)-int(window)))/KeyTile)*KeyTile:0;
    uint key_end=window?min(t.w,t.x+first+count-t.z+window):t.w;
    for(uint base=key_first;base<key_end;base+=KeyTile) {
        if constexpr(Causal) { if(t.z+base>=t.x+first+count)break; }
        auto kt=tk.slice(0,base);
        if constexpr(BF16Scaled)qk.run(qs,kt,sc);else qk.run(tq,kt,sc);
        for(auto it=sc.begin();it!=sc.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();
            if constexpr(Shaw) {
                // Q.R is contracted once for all relative offsets. Gather
                // only this key-query displacement, before the common scale.
                if(ij[0]+base<t.w && ij[1]<count) {
                    uint row=t.x+first+ij[1],key=t.z+base+ij[0];
                    *it+=rh[(ulong(row)*Heads+head)*401+int(row)-int(key)+200];
                }
            }
            *it=ij[0]+base<t.w && ij[1]<count?*it*((Gemma||BF16Scaled)?1.0f:(HD==64?0.125f:(HD==96?0.10206207261596575f:(HD==128?0.08838834764831844f:0.1178511301977579f)))):-INFINITY;
            if constexpr(Causal) { if(t.z+base+ij[0]>t.x+first+ij[1])*it=-INFINITY; }
            if(window && abs(int(t.z+base+ij[0])-int(t.x+first+ij[1]))>int(window))*it=-INFINITY;
            if constexpr(Sam) {
                if(ij[0]+base<t.w && ij[1]<count) {
                    ulong at=(ulong(t.x+first+ij[1])*Heads+head)*side;
                    *it+=rh[at+(base+ij[0])/side]+rw[at+(base+ij[0])%side];
                }
            }
        }
        reduce_rows(sc,high,reduction_operation::max,-INFINITY);
        for(uint i=0;i<maximum.get_capacity();++i) {
            high[i]=max(high[i],maximum[i]);
            old[i]=isfinite(maximum[i])?(BF16Scaled?exp2(maximum[i]-high[i]):exp(maximum[i]-high[i])):0.0f;
            maximum[i]=high[i];
        }
        for(auto it=sc.begin();it!=sc.end();++it)if(it.is_valid_element()) {
            auto ij=it.get_multidimensional_index();
            *it=ij[0]+base<t.w && ij[1]<count && (!window || isfinite(*high.map_iterator(it)))?(BF16Scaled?exp2(*it-*high.map_iterator(it)):exp(*it-*high.map_iterator(it))):0.0f;
        }
        reduce_rows(sc,sum);
        for(uint i=0;i<denominator.get_capacity();++i)denominator[i]=denominator[i]*old[i]+sum[i];
        old.store(tc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element())
            *it*=correction[first+it.get_multidimensional_index()[1]];
        auto vt=tv.slice(0,base);
        if(pv.template is_compatible_as_left_input<float,T,float>(sc)) {
            auto probs=pv.template get_left_input_cooperative_tensor<float,T,float>(sc);
            pv.run(probs,vt,acc);
        } else {
            auto plane=tensor(remap+first*KeyTile,extents<int,KeyTile,16>());
            sc.store(plane);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            pr.load(plane);
            pv.run(pr,vt,acc);
        }
    }
    denominator.store(tn);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(auto it=acc.begin();it!=acc.end();++it) {
        auto ij=it.get_multidimensional_index();
        if(it.is_valid_element() && ij[0]<HD && ij[1]<count) {
            ulong ix=(ulong(t.x+first+ij[1])*Heads+head)*HD+ij[0];
            float value=*it/normalizer[first+ij[1]];
            // MLX vision retains F32 scratch storage, but its SDPA result
            // crosses a BF16 operation boundary before the output projection.
            if(p[0]==31)value=float(bfloat(value));
            if(p[0]==30)reinterpret_cast<device bfloat*>(out)[ix]=bfloat(value);
            else if(p[0]==0 || p[0]==31)reinterpret_cast<device float*>(out)[ix]=value;
            else reinterpret_cast<device half*>(out)[ix]=half(value);
        }
    }
}
#define VIS_ATTN(NAME,GEMMA) \
kernel void NAME(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]], \
device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]], \
uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float correction[32],normalizer[32],remap[32*64]; \
vis_attention_impl<GEMMA>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap); }
VIS_ATTN(vis_attention,false)
VIS_ATTN(gv_attention,true)
#undef VIS_ATTN

#define VIS_ATTN64(HEADS) \
kernel void vis_attention64_##HEADS(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]], \
device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]], \
uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
threadgroup float correction[32],normalizer[32],remap[32*64]; \
vis_attention_impl<false,64,64,HEADS>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap); }
VIS_ATTN64(12)
VIS_ATTN64(16)
#undef VIS_ATTN64
