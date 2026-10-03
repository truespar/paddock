// Nemotron 3 Diarization. Bounded full-attention encoder; eight independent
// speaker probabilities, never a categorical softmax over speakers.
// Architecture/streaming references and attribution: diarization.NOTICE.md.
// MLX BF16 contract: 32-key online-softmax partitions, base-2 score domain,
// and relaxed tensor products. These arithmetic boundaries match MLX 0.32.1's
// NAX attention; the GGUF and Kumo paths retain their existing F32 contract.
// Fuse pointwise consumers at the tensor producer without contracting across
// the checkpoint's rounded projection, bias, residual and activation boundaries.
// Keep the contraction type/K order from gmlx_vmm64. The M5 path uses a
// 32-row tile to expose more workgroups for this encoder's narrow projections.
template<uint BM> inline void diar_project_bf16(device bfloat* w,device float* x,
 device float* out,device const float* bias,constant uint* p,uint2 g) {
 uint K=p[0],N=p[1],M=p[2],m=g.y*BM,n=g.x*64;
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(BM,64,dynamic_length_v<int>,false,true,true);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 op.run(a,b,acc);
 for(auto it=acc.begin();it!=acc.end();++it) {
  auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
  if(it.is_valid_element() && row<M && col<N) {
   float z=*it;if(p[3])z+=bias[col];z=mlx_bf(z);ulong at=ulong(row)*N+col;
   if(p[4]==2)z=mlx_bf(z+out[at]);
   if(p[4]==3)z=gmlx_erf_activation(z);
   if(p[4]==4)z=max(z,0.f);
   if(p[4]==5)z=mlx_bf(1.f/(1.f+exp(-z)));
   out[at]=z;
  }
 }
}
kernel void diar_project64(device bfloat* w [[buffer(0)]],device float* x [[buffer(1)]],
 device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]]) {diar_project_bf16<64>(w,x,out,bias,p,g);}
kernel void diar_project32(device bfloat* w [[buffer(0)]],device float* x [[buffer(1)]],
 device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]]) {diar_project_bf16<32>(w,x,out,bias,p,g);}
// Packed Q8_0 producer. The trained weights stay in their original blocks;
// only one 64x128 weight tile exists in threadgroup memory. Reuse the padded
// half input directly, preserving quant_tile<32>'s K=128 accumulation steps,
// then the materialized F32 bias/pointwise boundaries.
template<bool PrepareHalf=false> inline void diar_q8_project_impl(device const uchar* w,device half* x,
 device float* out,device const float* bias,constant uint* p,uint2 g,uint tid,threadgroup half* weights,device half* prepared=nullptr) {
 #pragma clang fp reassociate(off)
 constexpr uint BM=32,BN=64,BK=128;
 uint K=p[0],N=p[1],M=p[2],m=g.y*BM,n=g.x*BN;
 uint pitch=(K+127)/128*128,padded_rows=(M+127)/128*128;
 auto input=tensor(x,dextents<int,2>(pitch,padded_rows),array<int,2>{1,int(pitch)});
 auto b=tensor(weights,extents<int,BK,BN>());
 constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(input),decltype(b),float>();
 for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
 for(uint base=0;base<K;base+=BK) {
  for(uint i=tid*32;i<BN*BK;i+=4096) {
   uint col=n+i/BK,k=base+i%BK;
   if(col<N && k<K) {
    device const uchar* block=w+((ulong(col)*K+k)/32)*34;
    float scale=float(*reinterpret_cast<device const half*>(block));
    #pragma unroll
    for(uint j=0;j<8;++j) {
     float4 value=float4(*reinterpret_cast<device const packed_char4*>(block+2+j*4))*scale;
     *reinterpret_cast<threadgroup half4*>(weights+i+j*4)=half4(value);
    }
   } else {
    #pragma unroll
    for(uint j=0;j<8;++j)*reinterpret_cast<threadgroup half4*>(weights+i+j*4)=half4(0);
   }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto a=input.slice<BK,BM>(base,m);op.run(a,b,acc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(auto it=acc.begin();it!=acc.end();++it) {
  auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
  if(it.is_valid_element() && row<M && col<N) {
   float z=*it;if(p[3])z+=bias[col];ulong at=ulong(row)*N+col;
   if(p[4]==2)z+=out[at];
   if(p[4]==3)z=mv_gelu_value(z);
   if(p[4]==4)z=max(z,0.f);
   if(p[4]==5)z=1.f/(1.f+exp(-z));
   out[at]=z;
   if constexpr(PrepareHalf)prepared[at]=half(z);
  }
 }
 // Only selected for a width divisible by 128. Use a distinct, already
 // allocated head buffer: x is still live in other projection workgroups.
 if constexpr(PrepareHalf)if(g.y==0){
  for(uint i=tid;i<(padded_rows-M)*BN;i+=128){uint col=n+i%BN;
   if(col<N)prepared[ulong(M+i/BN)*N+col]=half(0);
  }
 }
}
kernel void diar_q8_project32(device const uchar* w [[buffer(0)]],device half* x [[buffer(1)]],
 device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup half weights[64*128];
 diar_q8_project_impl(w,x,out,bias,p,g,tid,weights);
}
kernel void diar_q8_project32_prepare(device const uchar* w [[buffer(0)]],device half* x [[buffer(1)]],
 device float* out [[buffer(2)]],device const float* bias [[buffer(3)]],device half* prepared [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup half weights[64*128];
 diar_q8_project_impl<true>(w,x,out,bias,p,g,tid,weights,prepared);
}
kernel void diar_attention_bf16(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],
 device bfloat* v [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float scores[1024],probs[1024],maximum[32],denominator[32],correction[32];
 kumo_attention_tile_impl<64,32,true,true,bfloat,true,true>(q,k,v,out,p,g,tid,scores,probs,maximum,denominator,correction);
}
kernel void diar_attention_f32(device float* q [[buffer(0)]],device float* k [[buffer(1)]],
 device float* v [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float scores[2048],probs[2048],maximum[32],denominator[32],correction[32];
 kumo_attention_tile_impl<64,64,false,false,float,false,true>(q,k,v,out,p,g,tid,scores,probs,maximum,denominator,correction);
}
kernel void diar_attention_q8_input(device float* q [[buffer(0)]],device float* k [[buffer(1)]],
 device float* v [[buffer(2)]],device float* out [[buffer(3)]],device half* prepared [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float scores[2048],probs[2048],maximum[32],denominator[32],correction[32];
 kumo_attention_tile_impl<64,64,false,false,float,false,true,true>(q,k,v,out,p,g,tid,scores,probs,maximum,denominator,correction,prepared);
}
// Unpacked arithmetic comparator, never selected by serving.
kernel void diar_attention_row_check(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],
 device bfloat* v [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float scores[1024],probs[1024],maximum[32],denominator[32],correction[32];
 kumo_attention_tile_impl<64,32,true,true,bfloat,true>(q,k,v,out,p,g,tid,scores,probs,maximum,denominator,correction);
}
// Diagnostic-only inverse packing. Serving attention consumes head-major
// Q/K/V directly; snapshots retain the reference's row-major tensor schema.
kernel void diar_heads_trace(device const uchar* x [[buffer(0)]],device float* y [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0]*512){uint src=(i%512/64*p[0]+i/512)*64+i%64;
  y[i]=p[1]?float(reinterpret_cast<device const bfloat*>(x)[src]):reinterpret_cast<device const float*>(x)[src];}
}
// One frame per group, 512-point radix-2 FFT in threadgroup memory. All
// coordinates are absolute; centered windows keep the preceding PCM sample
// for preemphasis. Sparse mel spans skip only exact zero filter coefficients.
kernel void diar_frontend(device const float* pcm [[buffer(0)]],device const float* window [[buffer(1)]],
 device const float* fb [[buffer(2)]],device const uint2* spans [[buffer(3)]],device const float2* tw [[buffer(4)]],
 device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint frame [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 #pragma clang fp contract(off)
 #pragma clang fp reassociate(off)
 uint global=p[2]+frame; // offset, total samples, first frame, BF16 output
 if(frame>=p[4] || global>=p[1]/160){if(tid<128)out[frame*128+tid]=0.f;return;}
 threadgroup float2 z[512];
 for(uint j=tid;j<512;j+=256) {
  int pos=int(global*160+j)-256;float x=0;
  if(pos>=0 && pos<int(p[1])) {x=pcm[pos-int(p[0])];if(pos>0)x-=0.97f*pcm[pos-1-int(p[0])];}
  x*=window[j];z[reverse_bits(j)>>23]=float2(x,0.f);
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint size=2;size<=512;size*=2){uint width=size/2,k=tid%width,a=tid/width*size+k,b=a+width;
  float2 e=z[a],o=z[b],t=tw[k*(512/size)];float2 r=float2(t.x*o.x-t.y*o.y,t.y*o.x+t.x*o.y);
  z[a]=e+r;z[b]=e-r;threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(tid<128){float sum=0;uint2 span=spans[tid];
  for(uint k=span.x;k<span.y;++k){float2 v=z[k];sum+=fb[tid*257+k]*(v.x*v.x+v.y*v.y);}
  float mel=log(sum+0x1p-24f);out[frame*128+tid]=p[3]?mlx_bf(mel):mel;}
}
kernel void diar_round(device float* x [[buffer(0)]],constant uint* p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0])x[i]=mlx_bf(x[i]);
}
kernel void diar_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device const float* b [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 #pragma clang fp reassociate(off)
 threadgroup float sums[8];uint D=512;ulong at=ulong(row)*D;
 float a=x[at+tid],c=x[at+tid+256];
 float mean=laya_sum(a+c,sums,lane,sg)/512.f;
 float var=laya_sum((a-mean)*(a-mean)+(c-mean)*(c-mean),sums,lane,sg)/512.f;
 float inv=rsqrt(var+1e-5f);
 for(uint j=tid;j<D;j+=256){float z=(x[at+j]-mean)*inv;
  if(p[0])z=mlx_bf(z);z=fma(z,w[j],b[j]);y[at+j]=p[0]?mlx_bf(z):z;}
}
// Residual + LayerNorm consumers. Retain the exact reduction geometry of
// gmlx_layer_norm / diar_norm respectively; never move the sum across the
// rounded residual boundary. Each lane owns every residual value it updates.
kernel void diar_residual_norm_bf16(device const float* delta [[buffer(0)]],device float* x [[buffer(1)]],
 device const float* w [[buffer(2)]],device const float* bias [[buffer(3)]],device float* out [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 #pragma clang fp contract(off)
 #pragma clang fp reassociate(off)
 threadgroup float sums[4],squares[4];ulong at=ulong(row)*512+tid*4;
 float z[4],sum=0,sq=0;
 for(uint i=0;i<4;++i){z[i]=mlx_bf(mlx_bf(delta[at+i])+x[at+i]);x[at+i]=z[i];sum+=z[i];sq+=z[i]*z[i];}
 sum=simd_sum(sum);sq=simd_sum(sq);
 if(tid%32==0){sums[tid/32]=sum;squares[tid/32]=sq;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float mean=precise::divide(simd_sum(tid%32<4?sums[tid%32]:0.f),512.f);
 float variance=precise::divide(simd_sum(tid%32<4?squares[tid%32]:0.f),512.f)-mean*mean;
 float inv=precise::rsqrt(max(variance,0.f)+1e-5f);
 for(uint i=0;i<4;++i){uint d=tid*4+i;out[at+i]=mlx_bf(fma(mlx_bf((z[i]-mean)*inv),w[d],bias[d]));}
}
kernel void diar_residual_norm_f32(device const float* delta [[buffer(0)]],device float* x [[buffer(1)]],
 device const float* w [[buffer(2)]],device const float* bias [[buffer(3)]],device float* out [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 #pragma clang fp reassociate(off)
 threadgroup float sums[8];ulong at=ulong(row)*512+tid;
 float a=delta[at]+x[at],b=delta[at+256]+x[at+256];x[at]=a;x[at+256]=b;
 float mean=laya_sum(a+b,sums,lane,sg)/512.f;
 float var=laya_sum((a-mean)*(a-mean)+(b-mean)*(b-mean),sums,lane,sg)/512.f;
 float inv=rsqrt(var+1e-5f);
 out[at]=fma((a-mean)*inv,w[tid],bias[tid]);out[at+256]=fma((b-mean)*inv,w[tid+256],bias[tid+256]);
}
// Normalization is the last F32 consumer before a Q8 projection. Materialize
// its F32 output (including trace boundaries) AND the exact half operand in
// the existing GEMM scratch. Extra groups zero padded rows, never read them.
// Each consumer must finish before the next producer reuses this scratch.
template<bool Residual> inline void diar_norm_q8_input_impl(device const float* delta,device float* x,
 device const float* w,device const float* bias,device float* out,device half* prepared,
 uint rows,uint row,uint tid,uint lane,uint sg,threadgroup float* sums) {
 #pragma clang fp reassociate(off)
 ulong at=ulong(row)*512+tid;
 if(row>=rows){prepared[at]=half(0);prepared[at+256]=half(0);return;}
 float a=x[at],b=x[at+256];
 if constexpr(Residual){a=delta[at]+a;b=delta[at+256]+b;x[at]=a;x[at+256]=b;}
 float mean=laya_sum(a+b,sums,lane,sg)/512.f;
 float var=laya_sum((a-mean)*(a-mean)+(b-mean)*(b-mean),sums,lane,sg)/512.f;
 float inv=rsqrt(var+1e-5f);
 float y0=fma((a-mean)*inv,w[tid],bias[tid]),y1=fma((b-mean)*inv,w[tid+256],bias[tid+256]);
 out[at]=y0;out[at+256]=y1;prepared[at]=half(y0);prepared[at+256]=half(y1);
}
kernel void diar_norm_q8_input(device float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device const float* bias [[buffer(2)]],device float* out [[buffer(3)]],device half* prepared [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float sums[8];
 diar_norm_q8_input_impl<false>(x,x,w,bias,out,prepared,p[0],row,tid,lane,sg,sums);
}
kernel void diar_residual_norm_q8_input(device const float* delta [[buffer(0)]],device float* x [[buffer(1)]],
 device const float* w [[buffer(2)]],device const float* bias [[buffer(3)]],device float* out [[buffer(4)]],
 device half* prepared [[buffer(5)]],constant uint* p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float sums[8];
 diar_norm_q8_input_impl<true>(delta,x,w,bias,out,prepared,p[0],row,tid,lane,sg,sums);
}
// p = elements, width, BF16 contract, operation (bias, residual, GELU,
// ReLU, sigmoid). Bias and residual round at separate graph boundaries.
kernel void diar_post(device float* x [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* residual [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0])return;float z=x[i];
 if(p[3]==1)z+=b[i%p[1]];
 if(p[2])z=mlx_bf(z);
 if(p[3]==2){z+=residual[i];if(p[2])z=mlx_bf(z);residual[i]=z;}
 if(p[3]==3)z=p[2]?gmlx_erf_activation(z):mv_gelu_value(z);
 if(p[3]==4)z=max(z,0.f);
 if(p[3]==5){z=1.f/(1.f+exp(-z));if(p[2])z=mlx_bf(z);}
 x[i]=z;
}
kernel void diar_heads(device const float* qkv [[buffer(0)]],device float* q [[buffer(1)]],
 device float* k [[buffer(2)]],device float* v [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*512)return;uint row=i/512,d=i%512,j=d%64,other=d/64*64+(j+32)%64;
 float angle=float(row)*(p[1]?exp2(-float(j%32)/32.f*13.287712379549449f):pow(10000.f,-float(j%32)/32.f)),c=fast::cos(angle),s=fast::sin(angle);
 ulong at=ulong(row)*1536;
 float a=j<32?qkv[at+d]*c-qkv[at+other]*s:qkv[at+other]*s+qkv[at+d]*c;
 float z=j<32?qkv[at+512+d]*c-qkv[at+512+other]*s:qkv[at+512+other]*s+qkv[at+512+d]*c;
 if(!p[1]){float sign=j<32?-1.f:1.f;a=qkv[at+d]*c+qkv[at+other]*(s*sign);z=qkv[at+512+d]*c+qkv[at+512+other]*(s*sign);}
 uint dst=(d/64*p[0]+row)*64+j;
 if(p[1]) {reinterpret_cast<device bfloat*>(q)[dst]=bfloat(a);reinterpret_cast<device bfloat*>(k)[dst]=bfloat(z);reinterpret_cast<device bfloat*>(v)[dst]=bfloat(qkv[at+1024+d]);}
 else {q[dst]=a;k[dst]=z;v[dst]=qkv[at+1024+d];}
}
// im2col is small (<=684 x 576); its projection reuses the tensor GEMM.
kernel void diar_conv_rows(device const float* x [[buffer(0)]],device float* rows [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*576)return;int r=int(i/576)+int(i%576/192)-1;uint c=i%192;
 rows[i]=(r>=0 && r<int(p[0]))?x[r*192+c]:0.f;
}
