// Bounded F32 online attention for both Clef's causal backbone (D=256)
// and joint head (D=64). Request-local tiles fix the arithmetic under packing.
// No quadratic score tensor, F16 KV cache, or activation narrowing.
template<uint D,bool Causal> inline void clef_attention_impl(device float* q,device float* k,
 device float* v,device float* out,device const uint4* tiles,constant uint* p,uint3 g,uint tid,
 threadgroup float* scores,threadgroup float* probs,threadgroup float* maxs,
 threadgroup float* den,threadgroup float* correction) {
 // One score/softmax walk feeds the entire output head. Splitting D=256
 // into four output groups repeats QK and softmax four times at long context.
 // N=32 uses every SIMD lane, while fixed request-local tiles keep batching
 // from changing reductions. The score workspace stays 4 KiB per group.
 constexpr uint M=16,N=32,O=D,KD=32;
 uint4 tile=tiles[g.y];uint first=tile.x,count=tile.y,kfirst=tile.z,keys=tile.w;
 uint H=p[0],KH=p[1],head=g.x,kh=head/(H/KH),dim=g.z*O;
 uint qs=p[2],ks=p[3],vs=p[4];
 q+=p[5]+ulong(first)*qs+head*D;
 k+=p[6]+ulong(kfirst)*ks+kh*D;v+=p[7]+ulong(kfirst)*vs+kh*D;
 auto tp=tensor(probs,extents<int,N,M>(),array<int,2>{1,N});
 auto ts=tensor(scores,extents<int,N,M>(),array<int,2>{1,N});
 constexpr auto qkd=matmul2d_descriptor(M,N,KD,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(M,O,N,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroups<4>> qk;
 matmul2d<pvd,execution_simdgroups<4>> pv;
 auto q0=tensor(q,dextents<int,2>{KD,int(count)},array<int,2>{1,int(qs)});
 auto k0=tensor(k,dextents<int,2>{KD,N},array<int,2>{1,int(ks)});
 auto v0=tensor(v+dim,dextents<int,2>{O,N},array<int,2>{1,int(vs)});
 auto result=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(v0),float>();
 for(uint i=0;i<result.get_capacity();++i)result[i]=0;
 if(tid<M){maxs[tid]=-INFINITY;den[tid]=0;}
 uint limit=Causal?min(keys,first-kfirst+count):keys;
 for(uint base=0;base<limit;base+=N) {
  uint valid=min(N,limit-base);
  auto score=qk.template get_destination_cooperative_tensor<decltype(q0),decltype(k0),float>();
  for(uint i=0;i<score.get_capacity();++i)score[i]=0;
  for(uint d=0;d<D;d+=KD) {
   auto a=tensor(q+d,dextents<int,2>{KD,int(count)},array<int,2>{1,int(qs)});
   auto b=tensor(k+ulong(base)*ks+d,dextents<int,2>{KD,int(valid)},array<int,2>{1,int(ks)});
   qk.run(a,b,score);
  }
  score.store(ts);threadgroup_barrier(mem_flags::mem_threadgroup);
  // One SIMD group per row; masked tail lanes never read invalid scores.
  for(uint row=tid/32;row<M;row+=4) {
   uint lane=tid%32;bool ok=row<count && lane<valid;
   if constexpr(Causal)ok=ok && kfirst+base+lane<=first+row;
   float z=ok?scores[row*N+lane]*rsqrt(float(D==96?72:D)):-INFINITY;
   float hi=max(maxs[row],simd_max(z));
   float a=isfinite(maxs[row])?exp(maxs[row]-hi):0;
   float b=ok?exp(z-hi):0;float sum=simd_sum(b);
   if(lane<N)probs[row*N+lane]=b;
   if(lane==0){maxs[row]=hi;den[row]=den[row]*a+sum;correction[row]=a;}
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(auto it=result.begin();it!=result.end();++it)if(it.is_valid_element())*it*=correction[it.get_multidimensional_index()[1]];
  auto tv=tensor(v+ulong(base)*vs+dim,dextents<int,2>{O,int(valid)},array<int,2>{1,int(vs)});
  pv.run(tp,tv,result);threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(auto it=result.begin();it!=result.end();++it)if(it.is_valid_element()) {
  auto ij=it.get_multidimensional_index();if(ij[1]<count)
   out[(ulong(first+ij[1])*H+head)*D+dim+ij[0]]=*it/den[ij[1]];
 }
}
#define CLEF_ATTENTION(NAME,D,CAUSAL) \
kernel void NAME(device float* q [[buffer(0)]],device float* k [[buffer(1)]],device float* v [[buffer(2)]], \
 device float* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]], \
 uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float s[512],b[512],m[16],d[16],c[16]; \
 clef_attention_impl<D,CAUSAL>(q,k,v,out,tiles,p,g,tid,s,b,m,d,c); }
CLEF_ATTENTION(clef_causal,256,true)
CLEF_ATTENTION(clef_attention,64,false)
CLEF_ATTENTION(clef_vision_attention,96,false)
#undef CLEF_ATTENTION
