// Original Flash Next BF16 operation contracts. F32 scheduler buffers hold
// BF16 values; this does not silently change the existing GGUF F32 graph.
// Operation diagnostics only. These kernels are not elected by a model;
// they isolate observable reference rounding before a graph change is made.
kernel void q4b_trace_pointwise(device const float* norm [[buffer(0)]],device const float* gate [[buffer(1)]],
    device float* sg [[buffer(2)]],device float* prod [[buffer(3)]],device float* sum [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*2560)return;uint base=i/2560*10240+i%2560;float v[4];
    for(uint j=0;j<4;++j) {
        float g=gate[base+j*2560];float tail=1/(1+exp(abs(g)));float s=g<0?tail:1-tail;
        if(p[1]==0)s=mlx_sigmoid_bf(g);else if(p[1]==1)s=mlx_bf(s);
        sg[base+j*2560]=s;v[j]=norm[base+j*2560]*s;
        if(p[2]==0)v[j]=mlx_bf(v[j]);prod[base+j*2560]=v[j];
    }
    float total=0;
    if(p[3]==0) {for(uint j=0;j<4;++j)total+=v[j];}
    else if(p[3]==1) {for(uint j=0;j<4;++j)total=mlx_bf(total+v[j]);}
    else total=mlx_bf(mlx_bf(v[0]+v[1])+mlx_bf(v[2]+v[3]));
    sum[i]=mlx_bf(total);
}
kernel void q4b_trace_qk(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    ulong base=ulong(g.y)*10240+g.x*128;float4 v=*reinterpret_cast<device const float4*>(x+base+lane*4);
    float4 sq=mlx_bf(v*v);float sum=0;
    for(uint j=0;j<4;++j)sum=p[1] ? mlx_bf(sum+sq[j]) : sum+sq[j];
    sum=mlx_bf(simd_sum(sum));float inv=mlx_bf(precise::rsqrt(mlx_bf(sum+mlx_bf(1e-6f))));
    v=mlx_bf(v*inv);if(g.x<16)v=mlx_bf(v*(p[2] ? 0.08838834764831844f : mlx_bf(0.08838834764831844f)));
    *reinterpret_cast<device float4*>(y+base+lane*4)=v;
    if(g.x<16)for(uint j=0;j<3;++j)*reinterpret_cast<device float4*>(y+ulong(g.y)*10240+4096+g.x*384+j*128+lane*4)=
        *reinterpret_cast<device const float4*>(x+ulong(g.y)*10240+4096+g.x*384+j*128+lane*4);
}
kernel void q4b_index_split(device const float* x [[buffer(0)]],device float* q [[buffer(1)]],device float* k [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*640)return;uint row=i/640,d=i%640;
    if(d<512)q[ulong(row)*512+d]=x[i];else k[ulong(row)*128+d-512]=x[i];
}
inline float q4b_rope(float v,float partner,uint d,uint position) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if(d>=64)return v;
    float frequency=precise::exp2(-float(d%32)/32.0f*precise::log2(1e7f));
    float angle=float(position)*frequency,cs=fast::cos(angle),sn=fast::sin(angle);
    return mlx_bf(d<32 ? fma(v,cs,-partner*sn) : fma(partner,sn,v*cs));
}
kernel void q4b_norm_rope(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
    device const uint4* meta [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    threadgroup float normed[256];ulong src=ulong(g.y)*p[3]+g.x*p[2],dst=(ulong(g.y)*p[1]+g.x)*p[0];
    float sq=0;for(uint d=lane;d<p[0];d+=32)sq+=x[src+d]*x[src+d];
    float inv=precise::rsqrt(simd_sum(sq)/float(p[0])+as_type<float>(p[4]));
    for(uint d=lane;d<p[0];d+=32)normed[d]=mlx_bf((x[src+d]*inv)*w[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<p[0];d+=32)y[dst+d]=q4b_rope(normed[d],normed[d<32?d+32:d-32],d,meta[g.y].y);
}
kernel void q4b_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],
    device const uint4* meta [[buffer(2)]],device const uint* pages [[buffer(3)]],device bfloat* kc [[buffer(4)]],
    device bfloat* vc [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[2]*512)return;uint row=i/512,d=i%512;uint4 m=meta[row];
    uint physical=pages[m.x*p[0]+m.y/16]*16+m.y%16;
    kc[ulong(physical)*512+d]=bfloat(k[i]);vc[ulong(physical)*512+d]=bfloat(v[i]);
}
kernel void q4b_pool(device const float* raw [[buffer(0)]],device const float* ring [[buffer(1)]],
    device const float* norm [[buffer(2)]],device const uint4* meta [[buffer(3)]],device float* pooled [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint4 m=meta[row];if(m.y%4!=3)return;threadgroup float values[128];float sq=0;
    for(uint d=lane;d<128;d+=32) {
        float sum=0;for(uint j=0;j<4;++j) {
            int r=int(row)-3+int(j);uint pos=m.y-3+j;
            sum+=r>=int(m.z) ? raw[ulong(r)*128+d] : ring[(m.x*4+pos%4)*128+d];
        }
        float v=mlx_bf(sum*.25f);values[d]=v;sq+=v*v;
    }
    float inv=precise::rsqrt(simd_sum(sq)/128.0f+1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<128;d+=32)values[d]=mlx_bf((values[d]*inv)*norm[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong dst=(ulong(m.x)*p[1]+m.y/4)*128;
    for(uint d=lane;d<128;d+=32)pooled[dst+d]=q4b_rope(values[d],values[d<32?d+32:d-32],d,m.y-3);
}
// Cache-only terminal execution has no attention/FFN output through which to
// observe invalid values. Validate precisely the rows written, not stale or
// unallocated pages. Keep errors sticky until the host poisons the model.
kernel void q4b_cache_finite(device const bfloat* kc [[buffer(0)]],device const bfloat* vc [[buffer(1)]],
    device const float* raw [[buffer(2)]],device const float* pooled [[buffer(3)]],
    device const uint4* meta [[buffer(4)]],device const uint* pages [[buffer(5)]],
    device atomic_uint* bad [[buffer(6)]],constant uint* p [[buffer(7)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[2]*512)return;uint row=i/512,d=i%512;uint4 m=meta[row];
    ulong at=(ulong(pages[m.x*p[0]+m.y/16])*16+m.y%16)*512+d;
    bool ok=isfinite(float(kc[at])) && isfinite(float(vc[at]));
    if(d<128) {
        ok=ok && isfinite(raw[ulong(row)*128+d]);
        if(m.y%4==3)ok=ok && isfinite(pooled[(ulong(m.x)*p[1]+m.y/4)*128+d]);
    }
    if(!ok)atomic_fetch_or_explicit(bad,8u,memory_order_relaxed);
}
kernel void q4b_join_gate(device const float* parts [[buffer(0)]],device const float* qg [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint rh [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    ulong base=ulong(rh)*p[4]*258;float hi=-INFINITY,den=0;
    for(uint s=0;s<p[4];++s)hi=max(hi,parts[base+s*258+256]);
    float acc[8];for(uint j=0;j<8;++j)acc[j]=0;
    for(uint s=0;s<p[4];++s) {
        ulong src=base+s*258;float d=parts[src+257];if(d==0)continue;
        float factor=exp(parts[src+256]-hi);den+=d*factor;
        for(uint j=0;j<8;++j)acc[j]+=parts[src+lane+j*32]*factor;
    }
    for(uint j=0;j<8;++j){uint d=lane+j*32;y[ulong(rh)*256+d]=mlx_bf(mlx_bf(acc[j]/den)*mlx_sigmoid_bf(qg[ulong(rh)*512+256+d]));}
}
kernel void q4b_route(device const float* logits [[buffer(0)]],device uint* ids [[buffer(1)]],
    device float* weights [[buffer(2)]],device float* shared [[buffer(3)]],device uint* invalid [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float values[16];float hi=-INFINITY;bool bad=false;
    for(uint j=0;j<16;++j){float v=logits[row*512+j*32+lane];bad|=!isfinite(v);v=isfinite(v)?v:0;values[j]=v;hi=max(hi,v);}
    hi=simd_max(hi);float sum=0;
    for(uint j=0;j<16;++j){values[j]=exp(values[j]-hi);sum+=values[j];}
    sum=simd_sum(sum);
    for(uint j=0;j<16;++j)values[j]=mlx_bf(values[j]/sum);
    float selected[10];uint selected_ids[10];
    for(uint e=0;e<10;++e) {
        float best=-INFINITY;uint id=0xffffffffu;
        for(uint j=0;j<16;++j)if(values[j]>=best){best=values[j];id=j*32+lane;}
        best=simd_max(best);id=simd_max(best==values[id/32] ? id : 0u);
        selected[e]=best;selected_ids[e]=id;
        if(id%32==lane)values[id/32]=-INFINITY;
    }
    float norm=0;
    for(uint j=0;j<10;++j)norm=mlx_bf(norm+selected[9-j]);
    // MLX's stable ascending partition takes the final ten entries: ties
    // at the cutoff retain higher ids, and the contraction order is ascending.
    if(lane<10){ids[row*10+lane]=selected_ids[9-lane];weights[row*10+lane]=mlx_bf(selected[9-lane]/norm);}
    bad=simd_any(bad);
    if(lane==0){float v=shared[row];shared[row]=mlx_sigmoid_bf(v);invalid[row]=bad || !isfinite(v) || !isfinite(norm) || norm<=0;}
}
kernel void q4b_fold(device const float* routed [[buffer(0)]],device const float* weights [[buffer(1)]],
    device const float* shared [[buffer(2)]],device const float* scale [[buffer(3)]],device float* y [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if(i>=p[0]*2560)return;uint row=i/2560,d=i%2560;float sum=0;
    for(uint lane=0;lane<8;++lane) {
        float sub=0;for(uint e=lane;e<10;e+=8)sub=mlx_bf(sub+mlx_bf(routed[(ulong(row)*10+e)*2560+d]*weights[row*10+e]));
        sum=mlx_bf(sum+sub);
    }
    y[i]=mlx_bf(mlx_bf(sum)+mlx_bf(shared[i]*scale[row]));
}
kernel void q4b_dn_qk_norm(device float* qkv [[buffer(0)]],constant uint* p [[buffer(1)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    ulong base=ulong(g.y)*p[2]+g.x*128;float4 v=*reinterpret_cast<device float4*>(qkv+base+lane*4);
    float4 sq=mlx_bf(v*v);float sum=0;
    for(uint j=0;j<4;++j)sum=mlx_bf(sum+sq[j]);
    sum=mlx_bf(simd_sum(sum));
    float inv=mlx_bf(precise::rsqrt(mlx_bf(sum+mlx_bf(1e-6f))));
    v=mlx_bf(v*inv);if(g.x<p[0])v=mlx_bf(v*mlx_bf(0.08838834764831844f));
    *reinterpret_cast<device float4*>(qkv+base+lane*4)=v;
}
kernel void q4b_dn_gated_norm(device float* x [[buffer(0)]],device const float* z [[buffer(1)]],device const float* w [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    ulong base=(ulong(g.y)*p[1]+g.x)*128+lane*4;
    float4 v=*reinterpret_cast<device float4*>(x+base);
    float sum=v.x*v.x;sum+=v.y*v.y;sum+=v.z*v.z;sum+=v.w*v.w;
    float inv=precise::rsqrt(simd_sum(sum)/128.0f+as_type<float>(p[6]));
    float4 norm=mlx_bf(mlx_bf(v*inv)*(*reinterpret_cast<device const float4*>(w+lane*4)));
    float4 gate=*reinterpret_cast<device const float4*>(z+base);
    *reinterpret_cast<device float4*>(x+base)=mlx_bf(norm*mlx_sigmoid_f32(gate));
}
kernel void q4b_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    threadgroup float partial[8];uint width=p[0],base=(g.y*p[1]+g.x)*width;
    float sq=0;
    for(uint d=tid;d<width;d+=256)sq+=x[base+d]*x[base+d];
    sq=simd_sum(sq);if(tid%32==0)partial[tid/32]=sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sq=simd_sum(tid<8 ? partial[tid] : 0.0f);
    if(tid==0)partial[0]=precise::rsqrt(precise::divide(sq,float(width))+as_type<float>(p[2]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Qwen4's custom norm keeps both multiply operands in F32 until its
    // final cast. This is not Qwen3.5's earlier normalized-BF16 boundary.
    for(uint d=tid;d<width;d+=256)y[base+d]=mlx_bf((x[base+d]*partial[0])*w[g.x*width+d]);
}
kernel void q4b_scale_silu(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if(i<p[0]) {float v=mlx_bf(x[i]*0.25f);x[i]=mlx_bf(v*mlx_sigmoid_bf(v));}
}
kernel void q4b_hc_mix(device const float* norm [[buffer(0)]],device const float* gate [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if(i>=p[0]*p[1])return;uint base=i/p[0]*p[0]*4+i%p[0];float sum=0;
    for(uint s=0;s<4;++s)sum=mlx_bf(sum+mlx_bf(norm[base+s*p[0]]*mlx_sigmoid_bf(gate[base+s*p[0]])));
    y[i]=mlx_bf(sum*0.25f);
}
kernel void q4b_hc_combine(device float* h [[buffer(0)]],device const float* delta [[buffer(1)]],
    device const float* inject [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if(i>=p[0]*p[1]*4)return;
    float gain=inject[i/p[0]];
    h[i]=mlx_bf(h[i]+mlx_bf(delta[(i/(p[0]*4))*p[0]+i%p[0]]*gain));
}
kernel void q4b_injection(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0])x[i]=mlx_bf(2.0f*mlx_sigmoid_bf(mlx_bf(x[i]*0.25f)));
}
// Combine attention's residual and normalize for the FFN in one traversal.
// Keep both BF16 residual boundaries and the original eight-SIMD RMS sum.
kernel void q4b_hc_combine_norm(device float* h [[buffer(0)]],device const float* delta [[buffer(1)]],
    device const float* inject [[buffer(2)]],device const float* w [[buffer(3)]],
    device float* y [[buffer(4)]],constant uint* p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    threadgroup float partial[8];
    uint base=(g.y*4+g.x)*2560;float gain=inject[g.y*4+g.x],sq=0;
    float values[10];
    for(uint j=0;j<10;++j) {
        uint d=tid+j*256;
        float v=mlx_bf(h[base+d]+mlx_bf(delta[g.y*2560+d]*gain));
        values[j]=v;h[base+d]=v;sq+=v*v;
    }
    sq=simd_sum(sq);if(tid%32==0)partial[tid/32]=sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sq=simd_sum(tid<8 ? partial[tid] : 0.0f);
    if(tid==0)partial[0]=precise::rsqrt(precise::divide(sq,2560.0f)+as_type<float>(p[2]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<10;++j) {
        uint d=tid+j*256;y[base+d]=mlx_bf((values[j]*partial[0])*w[g.x*2560+d]);
    }
}
kernel void q4a_ple_gather(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*160)return;
    uint id=ids[i/160],shard=id/p[1],row=id%p[1];
    // Each row occupies 80 code bytes + 10 scale bytes + 10 bias bytes.
    y[i]=shard<p[2] ? mlx_bf(q4a_value(w+ulong(shard)*p[1]*100,160,p[1],ulong(row)*160+i%160,4,32)) : NAN;
}
// The GPU hash is authoritative: stale/misaddressed staged rows fail closed.
kernel void q4a_ple_staged(device const uchar* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*160)return;
    device const uchar* row=w+ulong(i/160)*104;
    y[i]=*reinterpret_cast<device const uint*>(row)==ids[i/160]
        ? mlx_bf(q4a_value(row+4,160,1,i%160,4,32)) : NAN;
}
kernel void q4b_ple_gate(device const float* key [[buffer(0)]],device const float* query [[buffer(1)]],
    device const float* value [[buffer(2)]],device float* gated [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    // A 2560-wide BF16 row reduction has 640 four-value subtotals,
    // each rounded at every add, then BF16 SIMD and cross-SIMD boundaries.
    threadgroup float partial[20];uint width=p[0],base=(g.y*4+g.x)*width;float sum=0;
    for(uint j=0;j<4;++j){uint d=tid*4+j;sum=mlx_bf(sum+mlx_bf(key[base+d]*query[base+d]));}
    sum=mlx_bf(simd_sum(sum));if(tid%32==0)partial[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);sum=mlx_bf(simd_sum(tid<20 ? partial[tid] : 0.0f));
    if(tid==0) {
        float v=mlx_bf(mlx_bf(sum)/mlx_bf(sqrt(float(width))));
        v=mlx_bf(sign(v)*mlx_bf(sqrt(max(abs(v),mlx_bf(1e-6f)))));
        partial[0]=mlx_sigmoid_bf(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<width;d+=640)gated[base+d]=mlx_bf(partial[0]*value[g.y*width+d]);
}
kernel void q4b_ple_conv(device const float* norm [[buffer(0)]],device const float* w [[buffer(1)]],
    device const float* ring [[buffer(2)]],device const uint4* meta [[buffer(3)]],
    device const float* gated [[buffer(4)]],device float* h [[buffer(5)]],
    constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if(i>=p[0]*p[1])return;uint row=i/p[0],d=i%p[0];uint4 m=meta[row];float sum=0;
    for(uint tap=0;tap<4;++tap) {
        uint back=(3-tap)*3;float v=0;
        if(m.y>=back)v=row-m.z>=back ? norm[(row-back)*p[0]+d] : ring[(ulong(m.x)*9+(m.y-back)%9)*p[0]+d];
        sum+=w[d*4+tap]*v;
    }
    sum=mlx_bf(sum);float delta=mlx_bf(gated[i]+mlx_bf(sum*mlx_sigmoid_bf(sum)));
    h[i]=mlx_bf(h[i]+delta);
}
// Developer-only logit evidence consumer (never selected by model serving).
// Two GPU argmax scans retain lowest-ID ties, plus the teacher token's score.
kernel void q4b_margin(device const float* x [[buffer(0)]],device uint* ids [[buffer(1)]],
    device float* scores [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[8];threadgroup uint indices[8],selected;
    if(tid==0)selected=UINT_MAX;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint rank=0;rank<2;++rank) {
        float best=-INFINITY;uint id=UINT_MAX;
        for(uint i=tid;i<p[0];i+=256)if(i!=selected) {
            float v=x[i];if(v>best || (v==best && i<id)){best=v;id=i;}
        }
        float high=simd_max(best);uint index=simd_min(best==high ? id : UINT_MAX);
        if(tid%32==0){values[tid/32]=high;indices[tid/32]=index;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0) {
            best=-INFINITY;id=UINT_MAX;
            for(uint j=0;j<8;++j)if(values[j]>best || (values[j]==best && indices[j]<id)) {
                best=values[j];id=indices[j];
            }
            ids[rank]=id;scores[rank]=best;selected=id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0)scores[2]=x[p[1]];
}
