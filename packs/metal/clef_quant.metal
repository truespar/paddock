// Original packed Clef kernels. Only the current tile is widened: no persistent
// dequantized model, and no narrowing of Q8_0's exact half-scale * signed byte.
// Strict F32 MPP is the correctness path; a faster split/tensor route must earn
// election against this on complete decision graphs, not just isolated GEMMs.
inline float clef_weight(device const uchar* w,device const bfloat* s,device const bfloat* b,
 uint kind,uint K,uint row,uint col) {
 if(kind==0)return float(reinterpret_cast<device const bfloat*>(w)[ulong(row)*K+col]);
 if(kind==1){device const uchar* q=w+(ulong(row)*(K/32)+col/32)*34;
  return float(*reinterpret_cast<device const half*>(q))*float(as_type<char>(q[2+col%32]));}
 ulong g=ulong(row)*(K/64)+col/64;
 // The checkpoint's affine equation, separate multiply/add as MLX dequantize.
 return float(w[ulong(row)*K+col])*float(s[g])+float(b[g]);
}
// MLX normalizes Q/K with RMS epsilon (sum/128 + eps), while the official
// Transformers/GGUF contract uses L2 epsilon (sum + eps). Keep this distinction
// local to the MLX checkpoint lane; changing the shared DeltaNet kernel would
// silently alter already-qualified GGUF decisions.
kernel void clef_mlx_qk_norm(device float* qkv [[buffer(0)]],constant uint* p [[buffer(1)]],
 uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 #pragma clang fp contract(off)
 ulong base=ulong(g.y)*p[2]+g.x*128;
 float4 v=*reinterpret_cast<device float4*>(qkv+base+lane*4);
 float inv=rsqrt(simd_sum(dot(v,v))/128.f+1e-6f);
 v*=inv;v*=g.x<p[0]?1.f/128.f:rsqrt(128.f);
 *reinterpret_cast<device float4*>(qkv+base+lane*4)=v;
}

kernel void clef_quant_gather(device const uchar* w [[buffer(0)]],device const bfloat* s [[buffer(1)]],
 device const bfloat* b [[buffer(2)]],device const uint* ids [[buffer(3)]],
 device const uint2* spans [[buffer(4)]],device float* y [[buffer(5)]],constant uint* p [[buffer(6)]],
 uint i [[thread_position_in_grid]]) {
 #pragma clang fp contract(off)
 if(i>=p[0]*p[1])return;uint row=i/p[0],col=i%p[0];
 if(!p[3]){y[i]=clef_weight(w,s,b,p[2],p[0],ids[row],col);return;}
 uint2 span=spans[row];float sum=0;
 for(uint r=span.x;r<span.y;++r)sum+=clef_weight(w,s,b,p[2],p[0],ids[r],col);
 y[i]=sum/float(span.y-span.x);
}
kernel void clef_mm_quant(device const uchar* w [[buffer(0)]],device const bfloat* s [[buffer(1)]],
 device const bfloat* b [[buffer(2)]],device const float* x [[buffer(3)]],device float* y [[buffer(4)]],
 device const float* bias [[buffer(5)]],constant uint* p [[buffer(6)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 #pragma clang fp reassociate(off)
 #pragma clang fp contract(off)
 uint K=p[0],N=p[1],M=p[2],m=g.y*32,n=g.x*64;
 threadgroup float ax[32*32],bw[64*32],tile[32*64];
 auto a=tensor(ax,extents<int,32,32>(),array<int,2>{1,32});
 auto bt=tensor(bw,extents<int,32,64>(),array<int,2>{1,32});
 constexpr auto desc=matmul2d_descriptor(32,64,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(bt),float>();
 for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
 for(uint base=0;base<K;base+=32) {
  for(uint i=tid;i<32*32;i+=128){uint row=m+i/32,k=base+i%32;float z=0;
   if(row<M&&k<K){ulong at=ulong(row)*K+k;if(p[4]){device const bfloat* xp=reinterpret_cast<device const bfloat*>(x);
     z=float(xp[at])+float(xp[ulong(M)*K+at]);}else z=x[at];}ax[i]=z;}
  for(uint i=tid;i<64*32;i+=128){uint row=n+i/32,k=base+i%32;
   bw[i]=(row<N&&k<K)?clef_weight(w,s,b,p[3],K,row,k):0.f;}
  threadgroup_barrier(mem_flags::mem_threadgroup);op.run(a,bt,acc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()){
  auto ij=it.get_multidimensional_index();uint row=m+ij[1],col=n+ij[0];
  if(row<M&&col<N){float z=*it;if(p[5])z+=bias[col];if(p[6]==1)z+=y[ulong(row)*N+col];
   if(p[6]==2)z=mv_gelu_value(z);if(p[6]==3)tile[ij[1]*64+ij[0]]=z;else y[ulong(row)*N+col]=z;}}
 if(p[6]==3){threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<32*32;i+=128){uint row=m+i/32,col=n/2+i%32;
   if(row<M&&col<N/2){float a0=tile[i/32*64+i%32*2],b0=tile[i/32*64+i%32*2+1];float z=(a0/(1.f+exp(-a0)))*b0;
    ulong at=ulong(row)*(N/2)+col;if(p[4]){device bfloat* yp=reinterpret_cast<device bfloat*>(y);bfloat hi=bfloat(z);
      yp[at]=hi;yp[ulong(M)*(N/2)+at]=bfloat(z-float(hi));}else y[at]=z;}}}
}
