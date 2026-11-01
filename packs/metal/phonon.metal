// Paddock's Phonon FlowLM and streaming Mimi operators. All activations and
// reductions are F32. Exact BF16 matrix storage is widened inside the kernels.
// Matrices are row-major [out,in]; temporal activations are [time,channels].
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// A&S 7.1.26 erf approximation, the same F32 polynomial used by Paddock's
// vision encoders. Kept self-contained so loading this tiny model does not
// compile or specialize the language/vision kernel packs.
inline float phonon_gelu(float x){
 float z=abs(x)*0.7071067811865475244f,t=1.f/(1.f+0.3275911f*z),t2=t*t;
 float even=0.254829592f-0.284496736f*t,odd=1.421413741f-1.453152027f*t;
 float tail=t*(even+t2*(odd+1.061405429f*t2));
 return 0.5f*x*(1.f+copysign(1.f-tail*exp(-z*z),x));
}
inline float phonon_epilogue(float value,device float* y,uint at,uint mode){
 if(mode==1)return value+y[at];
 if(mode==2)return phonon_gelu(value);
 if(mode==3)return value/(1.f+exp(-value));
 return value;
}
kernel void phonon_mv(device const float* w [[buffer(0)]],device const float* x [[buffer(1)]],
 device const float* b [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint K=p[0],N=p[1],n=g.x*4+sg;if(n>=N)return;
 float4 sums=0;for(uint k=lane*4;k<K;k+=128){
  float4 a=p[5]?float4(*reinterpret_cast<device const bfloat4*>(reinterpret_cast<device const bfloat*>(w)+n*K+k)):
   *reinterpret_cast<device const float4*>(w+n*K+k);
  float4 z=*reinterpret_cast<device const float4*>(x+g.y*K+k);sums+=a*z;
 }
 float v=simd_sum((sums.x+sums.y)+(sums.z+sums.w));if(lane==0)y[g.y*N+n]=phonon_epilogue(v+(p[3]?b[n]:0.f),y,g.y*N+n,p[4]);
}
// Decode-only LayerNorm/adaLN + projection. The tiny 512-element normalization
// is duplicated per output tile to avoid serial single-group launches. Its
// reduction order and intermediate F32 rounding match phonon_norm/point.
kernel void phonon_norm_mv(device const float* w [[buffer(0)]],device const float* x [[buffer(1)]],
 device const float* nw [[buffer(2)]],device const float* nb [[buffer(3)]],device const float* mod [[buffer(4)]],
 device const float* bias [[buffer(5)]],device float* y [[buffer(6)]],constant uint* p [[buffer(7)]],
 uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 threadgroup float normalized[512],reduction[4];float sum=0;
 for(uint i=tid;i<512;i+=128)sum+=x[i];sum=simd_sum(sum);
 if(lane==0)reduction[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 float mean=((reduction[0]+reduction[1])+(reduction[2]+reduction[3]))/512.f;
 threadgroup_barrier(mem_flags::mem_threadgroup);sum=0;
 for(uint i=tid;i<512;i+=128){float d=x[i]-mean;sum+=d*d;}sum=simd_sum(sum);
 if(lane==0)reduction[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 float inv=rsqrt(((reduction[0]+reduction[1])+(reduction[2]+reduction[3]))/512.f+as_type<float>(p[3]));
 for(uint i=tid;i<512;i+=128){float a=(x[i]-mean)*inv;
  if(p[6])a=a*nw[i]+nb[i];
  if(p[5])a=a*(1.f+mod[p[4]+512+i])+mod[p[4]+i];
  normalized[i]=a;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 uint n=group*4+sg;if(n>=p[0])return;float4 sums=0;
 for(uint k=lane*4;k<512;k+=128){
  float4 a=p[1]?float4(*reinterpret_cast<device const bfloat4*>(reinterpret_cast<device const bfloat*>(w)+n*512+k)):
   *reinterpret_cast<device const float4*>(w+n*512+k);
  sums+=a*(*reinterpret_cast<threadgroup const float4*>(normalized+k));
 }
 float value=simd_sum((sums.x+sums.y)+(sums.z+sums.w));
 if(lane==0)y[n]=phonon_epilogue(value+(p[2]?bias[n]:0.f),y,n,p[7]);
}
// Eight rows and 32 columns, four SIMD groups. Bounds checked staging keeps
// ragged projections and convolution tails valid without padded allocations.
kernel void phonon_mm(device const float* w [[buffer(0)]],device const float* x [[buffer(1)]],
 device const float* b [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint K=p[0],N=p[1],M=p[2],row=g.y*8,col=g.x*32;
 threadgroup float a[8*32],v[32*32],out[8*32];
 simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
 for(uint base=0;base<K;base+=32){
  for(uint i=tid;i<8*32;i+=128){uint r=row+i/32,k=base+i%32;a[i]=(r<M&&k<K)?x[r*K+k]:0.f;}
  for(uint i=tid;i<32*32;i+=128){uint n=col+i/32,k=base+i%32;v[i]=(n<N&&k<K)?(p[5]?float(reinterpret_cast<device const bfloat*>(w)[n*K+k]):w[n*K+k]):0.f;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=0;k<32;k+=8){simdgroup_float8x8 l,r;simdgroup_load(l,a+k,32);simdgroup_load(r,v+sg*8*32+k,32,ulong2(0),true);simdgroup_multiply_accumulate(acc,l,r,acc);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(acc,out+sg*8,32);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<8*32;i+=128){uint r=row+i/32,n=col+i%32;if(r<M&&n<N)y[r*N+n]=phonon_epilogue(out[i]+(p[3]?b[n]:0.f),y,r*N+n,p[4]);}
}
kernel void phonon_point(device const float* x [[buffer(0)]],device const float* z [[buffer(1)]],
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]){
 if(i>=p[0])return;float a=x[i],b=z[i%p[2]],v=a;
 switch(p[1]){
  case 1:v=a+b;break;
  case 2:v=phonon_gelu(a);break;
  case 3:v=a/(1.f+exp(-a));break;
  case 4:v=a>=0?a:(abs(a)<0.001f?a*(1.f+a*(0.5f+a/6.f)):exp(a)-1.f);break;
  case 5:v=a*b+y[i];break;
  case 6:v=a*as_type<float>(p[3])+b;break;
  case 7:v=a*(1.f+z[(i%p[2])+p[2]])+b;break;
  case 9:v=(a+b)/(1.f+exp(-(a+b)));break;
 }
 y[i]=v;
}
// Larger causal-decoder tiles reuse each staged operand across eight 8x8
// products per SIMD group. Accumulation stays F32 and follows the same K order.
kernel void phonon_mm32(device const float* w [[buffer(0)]],device const float* x [[buffer(1)]],
 device const float* b [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint K=p[0],N=p[1],M=p[2],row=g.y*32,col=g.x*64;
 threadgroup float a[32*64],v[64*64],out[32*64];
 simdgroup_float8x8 acc[2][4];
 for(uint i=0;i<2;i++)for(uint j=0;j<4;j++)acc[i][j]=make_filled_simdgroup_matrix<float,8,8>(0.f);
 for(uint base=0;base<K;base+=64){
  for(uint i=tid;i<32*64;i+=128){uint r=row+i/64,k=base+i%64;a[i]=(r<M&&k<K)?x[r*K+k]:0.f;}
  for(uint i=tid;i<64*64;i+=128){uint n=col+i/64,k=base+i%64;v[i]=(n<N&&k<K)?(p[5]?float(reinterpret_cast<device const bfloat*>(w)[n*K+k]):w[n*K+k]):0.f;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=0;k<64;k+=8){
   simdgroup_float8x8 l[2],r[4];
   for(uint i=0;i<2;i++)simdgroup_load(l[i],a+((sg/2)*16+i*8)*64+k,64);
   for(uint j=0;j<4;j++)simdgroup_load(r[j],v+((sg%2)*32+j*8)*64+k,64,ulong2(0),true);
   for(uint i=0;i<2;i++)for(uint j=0;j<4;j++)simdgroup_multiply_accumulate(acc[i][j],l[i],r[j],acc[i][j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(uint i=0;i<2;i++)for(uint j=0;j<4;j++)simdgroup_store(acc[i][j],out+((sg/2)*16+i*8)*64+(sg%2)*32+j*8,64);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<32*64;i+=128){uint r=row+i/64,n=col+i%64;if(r<M&&n<N)y[r*N+n]=phonon_epilogue(out[i]+(p[3]?b[n]:0.f),y,r*N+n,p[4]);}
}
kernel void phonon_copy(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=x[i];}
kernel void phonon_denormalize(device const float* x [[buffer(0)]],device const float* s [[buffer(1)]],
 device const float* m [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]){
 if(i<p[0])y[i]=x[i]*s[i%32]+m[i%32];
}
kernel void phonon_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device const float* b [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 threadgroup float reduction[4];uint N=p[0];float mean=0;
 for(uint i=tid;i<N;i+=128)mean+=x[row*N+i];mean=simd_sum(mean);
 if(lane==0)reduction[sg]=mean;threadgroup_barrier(mem_flags::mem_threadgroup);
 mean=p[2]==1?0.f:((reduction[0]+reduction[1])+(reduction[2]+reduction[3]))/N;
 threadgroup_barrier(mem_flags::mem_threadgroup);float var=0;
 for(uint i=tid;i<N;i+=128){float d=x[row*N+i]-mean;var+=d*d;}var=simd_sum(var);
 if(lane==0)reduction[sg]=var;threadgroup_barrier(mem_flags::mem_threadgroup);
 float inv=rsqrt(((reduction[0]+reduction[1])+(reduction[2]+reduction[3]))/N+as_type<float>(p[1]));
 for(uint i=tid;i<N;i+=128){float a=(x[row*N+i]-mean)*inv;y[row*N+i]=p[2]==2?a:a*w[i]+(p[2]==1?0.f:b[i]);}
}
kernel void phonon_rope(device const float* qkv [[buffer(0)]],device float* q [[buffer(1)]],
 device float* k [[buffer(2)]],device float* v [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]){
 if(i>=p[0]*256)return;uint row=i/256,pair=i%256,dim=pair*2,pos=p[1]+row,at=(pos%p[2])*512+dim;
 float theta=float(pos)*pow(10000.f,-float(dim%64)/64.f),c=cos(theta),s=sin(theta);
 float2 a=*reinterpret_cast<device const float2*>(qkv+row*1536+dim);
 float2 b=*reinterpret_cast<device const float2*>(qkv+row*1536+512+dim);
 *reinterpret_cast<device float2*>(q+row*512+dim)=float2(a.x*c-a.y*s,a.x*s+a.y*c);
 *reinterpret_cast<device float2*>(k+at)=float2(b.x*c-b.y*s,b.x*s+b.y*c);
 *reinterpret_cast<device float2*>(v+at)=*reinterpret_cast<device const float2*>(qkv+row*1536+1024+dim);
}
// Eight lanes cooperate on each key: contiguous loads replace the scalar
// dot product's 512-float lane stride. Four SIMD groups then split the value
// reduction; the partition is fixed across request and decoder batch sizes.
kernel void phonon_attention(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
 device const float* v [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 threadgroup float scores[1536],reduce[4],partial[4*64];uint pos=p[0]+g.y,head=g.x*64;
 uint start=pos+1>p[2]?pos+1-p[2]:0,count=pos+1-start;
 for(uint j=tid/8;j<((count+15)/16)*16;j+=16){float sum=0;
  if(j<count){uint at=((start+j)%p[1])*512+head;for(uint d=tid%8;d<64;d+=8)sum+=q[g.y*512+head+d]*k[at+d];}
  for(uint delta=4;delta;delta/=2)sum+=simd_shuffle_xor(sum,delta);
  if(tid%8==0&&j<count)scores[j]=sum*0.125f;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float hi=-INFINITY;for(uint j=tid;j<count;j+=128)hi=max(hi,scores[j]);hi=simd_max(hi);
 if(lane==0)reduce[sg]=hi;threadgroup_barrier(mem_flags::mem_threadgroup);
 hi=max(max(reduce[0],reduce[1]),max(reduce[2],reduce[3]));threadgroup_barrier(mem_flags::mem_threadgroup);
 float sum=0;for(uint j=tid;j<count;j+=128){float a=exp(scores[j]-hi);scores[j]=a;sum+=a;}sum=simd_sum(sum);
 if(lane==0)reduce[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 float inv=1.f/((reduce[0]+reduce[1])+(reduce[2]+reduce[3]));
 float a=0,b=0;for(uint j=sg;j<count;j+=4){uint at=((start+j)%p[1])*512+head+lane;float score=scores[j];a+=score*v[at];b+=score*v[at+32];}
 partial[sg*64+lane]=a;partial[sg*64+lane+32]=b;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<64)y[g.y*512+head+tid]=((partial[tid]+partial[64+tid])+(partial[128+tid]+partial[192+tid]))*inv;
}
// Causal convolution staging, using immutable history until all output rows
// Fixed 64-key partitions give single-query FlowLM enough independent groups
// to occupy the GPU. Each writes an unnormalized value vector and softmax stats.
kernel void phonon_attention_split(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
 device const float* v [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 threadgroup float scores[64],reduce[4],partial[256];
 uint head=g.x*64,first=g.y*64,count=min(64u,p[0]+1-first),base=(g.x*p[1]+g.y)*66;
 for(uint j=tid/8;j<64;j+=16){float s=0;
  if(j<count)for(uint d=tid%8;d<64;d+=8)s+=q[head+d]*k[(first+j)*512+head+d];
  for(uint delta=4;delta;delta/=2)s+=simd_shuffle_xor(s,delta);
  if(tid%8==0)scores[j]=j<count?s*0.125f:-INFINITY;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float hi=simd_max(tid<64?scores[tid]:-INFINITY);
 if(lane==0)reduce[sg]=hi;threadgroup_barrier(mem_flags::mem_threadgroup);
 hi=max(reduce[0],reduce[1]);threadgroup_barrier(mem_flags::mem_threadgroup);
 float z=tid<count?exp(scores[tid]-hi):0;
 if(tid<64)scores[tid]=z;float sum=simd_sum(z);
 if(lane==0)reduce[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0){out[base+64]=hi;out[base+65]=reduce[0]+reduce[1];}
 float a=0,b=0;
 for(uint j=sg;j<count;j+=4){uint at=(first+j)*512+head+lane;a+=scores[j]*v[at];b+=scores[j]*v[at+32];}
 partial[sg*64+lane]=a;partial[sg*64+lane+32]=b;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<64)out[base+tid]=(partial[tid]+partial[64+tid])+(partial[128+tid]+partial[192+tid]);
}
kernel void phonon_attention_join(device const float* partial [[buffer(0)]],device float* y [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint head [[threadgroup_position_in_grid]],uint d [[thread_index_in_threadgroup]]){
 threadgroup float scales[24],denom;
 float hi=-INFINITY;for(uint i=0;i<p[0];i++)hi=max(hi,partial[(head*p[0]+i)*66+64]);
 if(d<p[0])scales[d]=exp(partial[(head*p[0]+d)*66+64]-hi);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(d==0){float sum=0;for(uint i=0;i<p[0];i++)sum+=scales[i]*partial[(head*p[0]+i)*66+65];denom=sum;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float value=0;for(uint i=0;i<p[0];i++)value+=scales[i]*partial[(head*p[0]+i)*66+d];
 y[head*64+d]=value/denom;
}

// Causal convolution staging, using immutable history until all output rows
// have consumed it. The subsequent history dispatch is a separate dependency.
kernel void phonon_conv_input(device const float* x [[buffer(0)]],device const float* history [[buffer(1)]],
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]){
 uint C=p[0],W=p[1],T=p[2];if(i>=T*C*W)return;
 uint row=i/(C*W),d=i%(C*W),channel=d/W,tap=d%W;int t=int(row+tap)-int(W-1);
 y[i]=t<0?history[(int(W-1)+t)*C+channel]:x[uint(t)*C+channel];
}
kernel void phonon_history(device const float* x [[buffer(0)]],device float* history [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]){
 uint C=p[0],H=p[1],T=p[2];if(i>=H*C)return;
 // All current decoder blocks have T >= H. No cross-thread in-place shift.
 history[i]=x[(T-H)*C+i];
}
// A transposed stride-S, width-2S convolution is a projection of the previous
// and current frames to S output phases. Weights are reordered once at load.
kernel void phonon_transpose_input(device const float* x [[buffer(0)]],device const float* history [[buffer(1)]],
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]){
 uint C=p[0];if(i>=p[1]*C*2)return;uint row=i/(C*2),d=i%(C*2);int t=int(row)-(d<C?1:0);uint c=d%C;
 y[i]=t<0?history[c]:x[uint(t)*C+c];
}
kernel void phonon_upsample(device const float* x [[buffer(0)]],device const float* history [[buffer(1)]],
 device const float* w [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]){
 if(i>=p[0]*8192)return;uint c=i%512,phase=(i/512)%16,frame=i/8192;
 float previous=frame?x[(frame-1)*512+c]:history[c];
 y[i]=x[frame*512+c]*w[c*32+phase]+previous*w[c*32+phase+16];
}

// Apple TensorOps keeps strict float operands and F32 accumulation here.
// No F16/BF16 downcast or reduced-precision multiplication is requested.
template<typename T> inline void phonon_mm_tensor_impl(device T* w,device float* x,
 device const float* b,device float* y,constant uint* p,uint2 g) {
 #pragma clang fp reassociate(off)
 uint K=p[0],N=p[1],M=p[2],row=g.y*32,col=g.x*64;
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,row);
 auto v=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,col);
 constexpr auto desc=matmul2d_descriptor(32,64,dynamic_length_v<int>,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(v),float>();
 op.run(a,v,acc);
 for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()){
  auto ij=it.get_multidimensional_index();uint r=row+ij[1],n=col+ij[0];
  if(r<M&&n<N)y[r*N+n]=phonon_epilogue(*it+(p[3]?b[n]:0.f),y,r*N+n,p[4]);
 }
}
kernel void phonon_mm_tensor(device float* w [[buffer(0)]],device float* x [[buffer(1)]],
 device const float* b [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint2 g [[threadgroup_position_in_grid]]) {
 if(p[5])phonon_mm_tensor_impl(reinterpret_cast<device bfloat*>(w),x,b,y,p,g);
 else phonon_mm_tensor_impl(w,x,b,y,p,g);
}
