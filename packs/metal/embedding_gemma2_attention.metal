// Bidirectional global attention with BF16 score/probability boundaries.
// A 128-query slab uses at most 8 MiB at 8K (four heads), not a square score
// allocation. Independent tensor tiles avoid a second QK walk for softmax.
// p: packed sequence start, sequence length, query offset, slab length.
kernel void eg2_global_qk(device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],device bfloat* scores [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint3 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat a[32*64],b[64*64];
    uint stride=(p[1]+63)/64*64;
    auto ta=tensor(a,extents<int,64,32>(),array<int,2>{1,64});
    auto tb=tensor(b,extents<int,64,64>(),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> mm;
    auto acc=mm.get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<512;base+=64){
        for(uint i=tid*4;i<32*64;i+=512){uint r=g.y*32+i/64,d=base+i%64;
            *reinterpret_cast<threadgroup bfloat4*>(a+i)=r<p[3]?*reinterpret_cast<device const bfloat4*>(q+(ulong(p[0]+p[2]+r)*4+g.z)*512+d):bfloat4(0);}
        for(uint i=tid*4;i<64*64;i+=512){uint r=g.x*64+i/64,d=base+i%64;
            *reinterpret_cast<threadgroup bfloat4*>(b+i)=r<p[1]?*reinterpret_cast<device const bfloat4*>(k+ulong(p[0]+r)*512+d):bfloat4(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm.run(ta,tb,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();
        uint r=g.y*32+ij[1],col=g.x*64+ij[0];
        if(it.is_valid_element() && r<p[3])scores[(ulong(g.z)*128+r)*stride+col]=bfloat(col<p[1]?*it:-INFINITY);
    }
}

kernel void eg2_global_softmax(device bfloat* scores [[buffer(0)]],
    constant uint* p [[buffer(1)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float parts[8];
    uint stride=(p[1]+63)/64*64;
    ulong start=(ulong(g.y)*128+g.x)*stride;
    float high=-INFINITY;
    for(uint j=tid;j<p[1];j+=256)high=max(high,float(scores[start+j]));
    high=simd_max(high);if(tid%32==0)parts[tid/32]=high;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    high=simd_max(tid%32<8?parts[tid%32]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;
    for(uint j=tid;j<p[1];j+=256)sum+=precise::exp(float(scores[start+j])-high);
    sum=simd_sum(sum);if(tid%32==0)parts[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(tid%32<8?parts[tid%32]:0.0f);
    for(uint j=tid;j<stride;j+=256)scores[start+j]=j<p[1]?bfloat(precise::exp(float(scores[start+j])-high)/sum):bfloat(0);
}

kernel void eg2_global_pv(device const bfloat* scores [[buffer(0)]],
    device const bfloat* v [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint3 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat a[32*64],b[64*64];
    uint stride=(p[1]+63)/64*64;
    auto ta=tensor(a,extents<int,64,32>(),array<int,2>{1,64});
    auto tb=tensor(b,extents<int,64,64>(),array<int,2>{1,64});
    // V is already K-by-N. Keep it in that orientation so adjacent lanes
    // read adjacent channels, rather than 512-element-strided token columns.
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> mm;
    auto acc=mm.get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint base=0;base<stride;base+=64){
        for(uint i=tid*4;i<32*64;i+=512){uint r=g.y*32+i/64,k=base+i%64;
            *reinterpret_cast<threadgroup bfloat4*>(a+i)=r<p[3]?*reinterpret_cast<device const bfloat4*>(scores+(ulong(g.z)*128+r)*stride+k):bfloat4(0);}
        for(uint i=tid*4;i<64*64;i+=512){uint col=g.x*64+i%64,k=base+i/64;
            *reinterpret_cast<threadgroup bfloat4*>(b+i)=k<p[1]?*reinterpret_cast<device const bfloat4*>(v+ulong(p[0]+k)*512+col):bfloat4(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm.run(ta,tb,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=acc.begin();it!=acc.end();++it){auto ij=it.get_multidimensional_index();
        uint r=g.y*32+ij[1],col=g.x*64+ij[0];
        if(it.is_valid_element() && r<p[3])out[(ulong(p[0]+p[2]+r)*4+g.z)*512+col]=mlx_bf(*it);
    }
}
