// Original Metal text tower seams. Contract: our CUDA SAM 3 text path and
// Meta VETextEncoder at 2345a4ad10. 32 causal tokens, 16 heads of 64; no RoPE.
kernel void sam3_text_embed(device const float* table [[buffer(0)]],
    device const float* pos [[buffer(1)]], device const uint* ids [[buffer(2)]],
    device float* x [[buffer(3)]], constant uint* p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*1024)return;
    uint row=i/1024,c=i%1024;
    // Both terms stay F32. Rounding the gathered embedding to F16 changes
    // the residual stream before the first layer has even run.
    x[i]=table[ulong(ids[row])*1024+c]+pos[(row%32)*1024+c];
}

kernel void sam3_text_qkv(device const half* in [[buffer(0)]],
    device const float* bias [[buffer(1)]], device half* q [[buffer(2)]],
    device half* k [[buffer(3)]], device half* v [[buffer(4)]],
    constant uint* p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*1024)return;
    uint col=i%1024;ulong row=ulong(i/1024)*3072;
    // Unlike the vision fused producer, text rounds its GEMM before bias.
    // Scale Q before its final half store; K/V have exactly one bias seam.
    q[i]=half((float(in[row+col])+bias[col])*0.125f);
    k[i]=half(float(in[row+1024+col])+bias[1024+col]);
    v[i]=half(float(in[row+2048+col])+bias[2048+col]);
}

// One group per (prompt, head), eight SIMD groups sweep four query rows
// each. Reuse the tiny K/V domain from padded threadgroup storage; no NxN
// score allocation, no reduction whose partition changes with batch size.
// This is intentionally a short-domain latency kernel, not global vision
// attention. Dot products and value folds retain CUDA's component/key order.
kernel void sam3_text_attention(device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
    device half* out [[buffer(3)]], constant uint* p [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    threadgroup float sk[32*65],sv[32*65],sq[8*64];
    uint lane=tid%32,sg=tid/32;ulong base=ulong(g/16)*32*1024+(g%16)*64;
    for(uint i=tid;i<32*64;i+=256){uint row=i/64,c=i%64;
        sk[row*65+c]=float(k[base+row*1024+c]);
        sv[row*65+c]=float(v[base+row*1024+c]);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint row=sg;row<32;row+=8){
        sq[sg*64+lane]=float(q[base+row*1024+lane]);
        sq[sg*64+lane+32]=float(q[base+row*1024+lane+32]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float score=-INFINITY;
        if(lane<=row){score=0;
            for(uint c=0;c<64;++c)score=fma(sq[sg*64+c],sk[lane*65+c],score);}
        float maximum=score;
        for(uint step=16;step;step/=2)maximum=max(maximum,simd_shuffle_xor(maximum,step));
        float weight=lane<=row?exp(score-maximum):0.f,sum=weight;
        for(uint step=16;step;step/=2)sum+=simd_shuffle_xor(sum,step);
        float a=0,b=0;
        for(uint j=0;j<=row;++j){float w=simd_shuffle(weight,j);
            a=fma(w,sv[j*65+lane],a);b=fma(w,sv[j*65+lane+32],b);}
        float inv=precise::divide(1.f,sum);
        out[base+row*1024+lane]=half(a*inv);
        out[base+row*1024+lane+32]=half(b*inv);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
