// Native Clef decision graph. BF16 storage, F32 activation/attention/head.
// Original kernels; architecture: Cloudflare/clef-flash joint_schema_model.py.
// Fixed reduction geometry: packing neighbors must not elect new arithmetic.
// Split once per producer, reuse across Q/K/V or QKV/Z/AB consumers. Unlike
// narrowing to BF16, the second part retains ~16 significant input bits.
kernel void clef_prepare(device const float* x [[buffer(0)]],device bfloat* parts [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0]){float z=x[i];bfloat hi=bfloat(z);parts[i]=hi;parts[p[0]+i]=bfloat(z-float(hi));}
}

template<uint BM,bool Grouped,uint BK=0> inline void clef_mm_parts_impl(device bfloat* w,device bfloat* x,
 device float* y,constant uint* p,uint2 g,uint tid,threadgroup float* tile) {
 #pragma clang fp reassociate(off)
 uint K=p[0],N=p[1],M=p[2],m=g.y*BM,n=g.x*64;
 if constexpr(Grouped)if(ulong(K)*N*2>8*1024*1024) {
  uint nt=(N+63)/64,mt=(M+BM-1)/BM,index=g.y*nt+g.x;
  uint group=index/(nt*8),first=group*8,count=min(8u,mt-first),local=index-group*nt*8;
  m=(first+local%count)*BM;n=(local/count)*64;
 }
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto low=tensor(x+ulong(M)*K,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
 constexpr auto mode=BK?matmul2d_descriptor::mode::multiply_accumulate:matmul2d_descriptor::mode::multiply;
 constexpr auto desc=matmul2d_descriptor(BM,64,BK?int(BK):dynamic_length_v<int>,false,true,true,mode);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto hi=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 auto lo=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 if constexpr(BK==0){op.run(a,b,hi);op.run(low,b,lo);}
 else {
  for(uint i=0;i<hi.get_capacity();++i){hi[i]=0;lo[i]=0;}
  // Keep the two activation parts on the same weight stripe. A complete K
  // walk for high followed by another for low can evict the shared weights.
  // Separate accumulators retain the split-product arithmetic contract.
  for(uint base=0;base<K;base+=BK) {
   auto ah=tensor(x+ulong(m)*K+base,dextents<int,2>(min(BK,K-base),min(BM,M-m)),array<int,2>{1,int(K)});
   auto al=tensor(x+ulong(M+m)*K+base,dextents<int,2>(min(BK,K-base),min(BM,M-m)),array<int,2>{1,int(K)});
   auto bw=tensor(w+ulong(n)*K+base,dextents<int,2>(min(BK,K-base),min(64u,N-n)),array<int,2>{1,int(K)});
   op.run(ah,bw,hi);op.run(al,bw,lo);
   threadgroup_barrier(mem_flags::mem_none);
  }
 }
 for(uint i=0;i<hi.get_capacity();++i)hi[i]+=lo[i];
 for(auto it=hi.begin();it!=hi.end();++it)if(it.is_valid_element()) {
  auto ij=it.get_multidimensional_index();uint row=m+ij[1],col=n+ij[0];
  if(row<M && col<N){float z=*it;if(p[3]==1)z+=y[ulong(row)*N+col];
   if(p[3]==3)tile[ij[1]*64+ij[0]]=z;else y[ulong(row)*N+col]=z;}
 }
 if(p[3]==3) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The MLP intermediate is consumed only by another projection. Emit both
  // parts directly, in the same bytes as one F32 plane, without a second
  // conversion dispatch or a larger workspace.
  device bfloat* packed=reinterpret_cast<device bfloat*>(y);
  for(uint i=tid;i<BM*32;i+=128){uint row=m+i/32,col=n/2+i%32;
   if(row<M && col<N/2){float a=tile[(i/32)*64+(i%32)*2],b=tile[(i/32)*64+(i%32)*2+1];
    float z=(a/(1.f+exp(-a)))*b;bfloat h=bfloat(z);ulong at=ulong(row)*(N/2)+col;
    packed[at]=h;packed[ulong(M)*(N/2)+at]=bfloat(z-float(h));}
  }
 }
}
#define CLEF_MM_PARTS(NAME,BM,GROUPED) \
kernel void NAME(device bfloat* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device float* y [[buffer(2)]], \
 constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float tile[BM*64];clef_mm_parts_impl<BM,GROUPED>(w,x,y,p,g,tid,tile); }
CLEF_MM_PARTS(clef_mm_parts,64,true)
CLEF_MM_PARTS(clef_mm_parts_linear,64,false)
#undef CLEF_MM_PARTS
#define CLEF_MM_BLOCKED(NAME,BK) \
kernel void NAME(device bfloat* w [[buffer(0)]],device bfloat* x [[buffer(1)]],device float* y [[buffer(2)]], \
 constant uint* p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float tile[64*64];clef_mm_parts_impl<64,true,BK>(w,x,y,p,g,tid,tile); }
CLEF_MM_BLOCKED(clef_mm_parts_k256,256)
CLEF_MM_BLOCKED(clef_mm_parts_k1024,1024)
#undef CLEF_MM_BLOCKED

kernel void clef_mm(device bfloat* w [[buffer(0)]], device float* x [[buffer(1)]],
 device float* y [[buffer(2)]], device const float* bias [[buffer(3)]],
 constant uint* p [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
 #pragma clang fp reassociate(off)
 uint K=p[0],N=p[1],M=p[2],m=g.y*32,n=g.x*64;
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto b=tensor(w,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(32,64,dynamic_length_v<int>,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 op.run(a,b,acc);
 threadgroup float tile[32*64];
 for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
  auto ij=it.get_multidimensional_index();uint row=m+ij[1],col=n+ij[0];
  if(row<M && col<N) {
   float z=*it;if(p[3])z+=bias[col];
   if(p[4]==1)z+=y[ulong(row)*N+col];
   if(p[4]==2)z=mv_gelu_value(z);
   if(p[4]==4)z=0.5f*z*(1.f+precise::tanh(0.7978845608028654f*(z+0.044715f*z*z*z)));
   if(p[4]==3)tile[ij[1]*64+ij[0]]=z;
   else y[ulong(row)*N+col]=z;
  }
 }
 if(p[4]==3) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<32*32;i+=128) {
   uint row=m+i/32,col=n/2+i%32;
   if(row<M && col<N/2) {
    float a=tile[(i/32)*64+(i%32)*2],b=tile[(i/32)*64+(i%32)*2+1];
    y[ulong(row)*(N/2)+col]=(a/(1.f+exp(-a)))*b;
   }
  }
 }
}

kernel void clef_embed(device const bfloat* w [[buffer(0)]],device const uint* ids [[buffer(1)]],
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0]*p[1])y[i]=float(w[ulong(ids[i/p[0]])*p[0]+i%p[0]]);
}

kernel void clef_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device const float* bias [[buffer(2)]],device float* y [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 #pragma clang fp contract(off)
 threadgroup float red[8];uint D=p[0];ulong base=ulong(row)*D;
 float s=0;for(uint j=tid;j<D;j+=256)s+=x[base+j];
 float mean=p[2]?laya_sum(s,red,lane,sg)/float(D):0.f;
 float ss=0;for(uint j=tid;j<D;j+=256){float z=x[base+j]-mean;ss+=z*z;}
 float inv=rsqrt(laya_sum(ss,red,lane,sg)/float(D)+as_type<float>(p[1]));
 for(uint j=tid;j<D;j+=256){float z=(x[base+j]-mean)*inv*w[j];y[base+j]=p[2]?z+bias[j]:z;}
}

// Text rotary positions start at zero independently for every packed request.
// The small table is constructed once using the reference's F32 frequency and
// angle boundaries; no growing iterative-angle error on long requests.
kernel void clef_rope(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device const float2* rope [[buffer(2)]],device const uint* positions [[buffer(3)]],
 device float* y [[buffer(4)]],constant uint* p [[buffer(5)]],
 uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 #pragma clang fp contract(off)
 uint H=p[0],D=p[1],R=p[2],stride=p[3],head=g.x,row=g.y;
 ulong at=ulong(row)*stride+head*D*(p[4]?2:1);float values[8],s=0;
 for(uint j=0;j<8;++j){float a=x[at+lane+32*j];values[j]=a;s+=a*a;}
 float inv=rsqrt(simd_sum(s)/float(D)+as_type<float>(p[5]));
 for(uint j=0;j<8;++j) {
  uint d=lane+32*j;float z=values[j]*inv*w[d];
  if(d<R) {
   uint other=(d+R/2)%R;float v=x[at+other]*inv*w[other];
   uint pair=d%(R/2),axis=(pair%3==1&&pair<3*p[6])?1:((pair%3==2&&pair<3*p[7])?2:0);
   float2 cs=rope[positions[row*3+axis]*(R/2)+pair];
   z=z*cs.x+(d<R/2?-v:v)*cs.y;
  }
  y[(ulong(row)*H+head)*D+d]=z;
 }
}
kernel void clef_attn_gate(device float* y [[buffer(0)]],device const float* q [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0]*p[1]){uint row=i/p[0],col=i%p[0],D=p[2];
  y[i]*=1.f/(1.f+exp(-q[ulong(row)*p[0]*2+(col/D)*D*2+D+col%D]));}
}

kernel void clef_conv(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device const uint2* bounds [[buffer(2)]],device float* y [[buffer(3)]],
 constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*p[1])return;uint row=i/p[0],d=i%p[0];float a=0;
 for(uint j=0;j<4;++j){int r=int(row)+int(j)-3;if(r>=int(bounds[row].x))a+=x[ulong(r)*p[0]+d]*w[d*4+j];}
 y[i]=a/(1.f+exp(-a));
}
kernel void clef_gates(device const float* ab [[buffer(0)]],device const float* a [[buffer(1)]],
 device const float* dt [[buffer(2)]],device float* gates [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint i [[thread_position_in_grid]]) {
 uint H=p[0];if(i>=H*p[1])return;uint row=i/H,h=i%H;float z=ab[row*2*H+h]+dt[h];
 gates[2*i]=a[h]*(max(z,0.f)+log(1.f+exp(-abs(z))));
 gates[2*i+1]=1.f/(1.f+exp(-ab[row*2*H+H+h]));
}
// HF grouped value-head order; fresh register state per request, no persistent
// 256-request state reservation and no recurrent data crosses request bounds.
kernel void clef_recurrent(device const float* qkv [[buffer(0)]],device const float* gates [[buffer(1)]],
 device const uint2* runs [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint first=runs[g.z].x,count=runs[g.z].y,H=p[0],KH=p[1],C=(2*KH+H)*128;
 uint h=g.y,kh=p[2]?h%KH:h/(H/KH),v=g.x*16+tid/8,lane=tid%8;float state[16];
 for(uint j=0;j<16;++j)state[j]=0;
 for(uint r=first;r<first+count;++r) {
  float decay=exp(gates[(r*H+h)*2]),beta=gates[(r*H+h)*2+1],pred=0;
  for(uint j=0;j<16;++j){state[j]*=decay;pred+=state[j]*qkv[ulong(r)*C+KH*128+kh*128+lane+8*j];}
  pred+=simd_shuffle_xor(pred,1);pred+=simd_shuffle_xor(pred,2);pred+=simd_shuffle_xor(pred,4);
  float delta=(qkv[ulong(r)*C+2*KH*128+h*128+v]-pred)*beta,y=0;
  for(uint j=0;j<16;++j){uint d=lane+j*8;state[j]+=qkv[ulong(r)*C+KH*128+kh*128+d]*delta;y+=state[j]*qkv[ulong(r)*C+kh*128+d];}
  y+=simd_shuffle_xor(y,1);y+=simd_shuffle_xor(y,2);y+=simd_shuffle_xor(y,4);
  if(lane==0)out[(ulong(r)*H+h)*128+v]=y;
 }
}
