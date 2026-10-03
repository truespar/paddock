// Kumo-Tabular prepared-table graph. Original F32 weights and activations;
// no low-precision substitution. See kumo.NOTICE.md for architecture sources.
// Attention is online: never allocate a rows x rows score matrix.
// K and V share one input and reduction contract. Issue the same 32x32
// projection tiles together, writing separate planes without a packing pass.
kernel void kumo_kv(device float* w [[buffer(0)]],device float* x [[buffer(1)]],
 device float* key [[buffer(2)]],device float* value [[buffer(3)]],device const float* bias [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]]) {
 uint D=p[0],M=p[1],n=g.x*32,m=g.y*32;
 auto a=tensor(x,dextents<int,2>(D,M),array<int,2>{1,int(D)}).slice(0,m);
 auto b=tensor(w+ulong(D)*D,dextents<int,2>(D,2*D),array<int,2>{1,int(D)}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(32,32,dynamic_length_v<int>,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();op.run(a,b,acc);
 for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
  auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
  if(col<2*D && row<M){float z=*it+bias[D+col];if(col<D)key[ulong(row)*D+col]=z;else value[ulong(row)*D+col-D]=z;}
 }
}
kernel void kumo_mm(device float* w [[buffer(0)]], device float* x [[buffer(1)]],
 device float* y [[buffer(2)]], device const float* bias [[buffer(3)]],
 constant uint* p [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]]) {
 uint K=p[0],N=p[1],M=p[2],n=g.x*32,m=g.y*32,off=p[3];
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto b=tensor(w+ulong(off)*K,dextents<int,2>(K,N),array<int,2>{1,int(K)}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(32,32,dynamic_length_v<int>,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 op.run(a,b,acc);
 for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
  auto ij=it.get_multidimensional_index(); uint col=n+ij[0],row=m+ij[1];
  if(col<N && row<M) {float z=*it+bias[off+col];if(p[4])z=mv_gelu_value(z);y[ulong(row)*N+col]=z;}
 }
}
// MLP output projection followed by residual addition. Preserve the bias
// rounding boundary of kumo_mm + kumo_add; no intermediate output plane.
kernel void kumo_mlp_out(device float* w [[buffer(0)]],device float* x [[buffer(1)]],
 device float* y [[buffer(2)]],device const float* bias [[buffer(3)]],
 constant uint* p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]]) {
 #pragma clang fp reassociate(off)
 uint D=p[0],M=p[1],K=2*D,n=g.x*32,m=g.y*32;
 auto a=tensor(x,dextents<int,2>(K,M),array<int,2>{1,int(K)}).slice(0,m);
 auto b=tensor(w,dextents<int,2>(K,D),array<int,2>{1,int(K)}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(32,32,dynamic_length_v<int>,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();op.run(a,b,acc);
 for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
  auto ij=it.get_multidimensional_index();uint col=n+ij[0],row=m+ij[1];
  if(col<D && row<M){float z=*it+bias[col];y[ulong(row)*D+col]+=z;}
 }
}
kernel void kumo_norm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float sums[8];uint D=p[0],len=p[1],stride=p[2];
 ulong src=ulong(row/len*stride+row%len)*D;float ss=0;
 for(uint j=tid;j<D;j+=256)ss+=x[src+j]*x[src+j];
 float inv=rsqrt(laya_sum(ss,sums,lane,sg)/D+1.1920928955078125e-7f);
 for(uint j=tid;j<D;j+=256)y[ulong(row)*D+j]=x[src+j]*inv*w[j];
}
// Retain the residual needed by the MLP while normalizing its register
// values. The reduction layout is identical to kumo_norm; no extra plane.
kernel void kumo_add_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device const float* w [[buffer(2)]],device float* residual [[buffer(3)]],device float* norm [[buffer(4)]],
 constant uint* p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 // Keep the rounded residual before squaring/scaling. Reassociation across
 // the fused add otherwise changes the original two-dispatch arithmetic.
 #pragma clang fp reassociate(off)
 threadgroup float sums[8];uint D=p[0];float values[4],ss=0;
 for(uint j=tid,n=0;j<D;j+=256,++n){ulong at=ulong(row)*D+j;float z=a[at]+b[at];residual[at]=z;values[n]=z;ss+=z*z;}
 float inv=rsqrt(laya_sum(ss,sums,lane,sg)/D+1.1920928955078125e-7f);
 for(uint j=tid,n=0;j<D;j+=256,++n)norm[ulong(row)*D+j]=values[n]*inv*w[j];
}
kernel void kumo_heads(device const float* x [[buffer(0)]],device const float* freq [[buffer(1)]],
 device float* y [[buffer(2)]],constant uint* p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint H=p[0],D=p[1],seq=p[2],pos=(row/H)%seq;float v[2];float ss=0;
 for(uint i=0;i<D/32;++i){uint j=lane+32*i;ulong at=ulong(row)*D;float z=x[at+j];
  if(p[3]){float a=float(pos)*freq[j%(D/2)];z=z*cos(a)+(j<D/2?-1.f:1.f)*x[at+(j+D/2)%D]*sin(a);}
  v[i]=z;ss+=z*z;
 }
 float inv=rsqrt(simd_sum(ss)/D+1e-6f);
 for(uint i=0;i<D/32;++i)y[ulong(row)*D+lane+32*i]=v[i]*inv;
}
kernel void kumo_scale(device float* q [[buffer(0)]],device const float* scale [[buffer(1)]],
 device const float* gate [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0])return;float z=q[i]*log(float(max(p[3],1u)))*scale[(i/p[1])%p[2]];
 if(p[4])z*=1.f+tanh(gate[i]);q[i]=z;
}
kernel void kumo_attention(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
 device const float* v [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint H=p[0],D=p[1],Q=p[2],K=p[3],KH=p[6],head=row%H,pos=(row/H)%Q,batch=row/(H*Q);
 uint kh=(p[4] && pos>=p[5]) ? head/(H/p[4]) : head;
 float qs[2],acc[2]={0.f,0.f};for(uint j=0;j<D/32;++j)qs[j]=q[ulong(row)*D+lane+32*j];
 float m=-INFINITY,s=0.f;
 for(uint r=0;r<K;++r){ulong at=(ulong(batch)*K*KH+r*KH+kh)*D;float dot=0;
  for(uint j=0;j<D/32;++j)dot+=qs[j]*k[at+lane+32*j];
  float score=simd_sum(dot)*rsqrt(float(D)),nm=max(m,score),a=exp(m-nm),b=exp(score-nm);
  s=s*a+b;for(uint j=0;j<D/32;++j)acc[j]=acc[j]*a+b*v[at+lane+32*j];m=nm;
 }
 for(uint j=0;j<D/32;++j)out[ulong(row)*D+lane+32*j]=acc[j]/s;
}
// Only the first Test-GQA heads are used by future query rows. Store them
// compactly; context encoding still computes with every original head.
kernel void kumo_cache_heads(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 uint H=p[0],KH=p[1],D=p[2],row=i/(KH*D),col=i%(KH*D);if(row<p[3])out[i]=x[row*H*D+col];
}

// F32 FlashAttention: queries share tensor-matrix key/value tiles. No score
// plane proportional to sequence length, no F16/BF16 conversion. Dispatches
// split at the context/query boundary so Test-GQA never changes mid-tile.
template<ushort D,uint N=64,bool Base2=false,bool Relaxed=false,typename T=float,bool BF16Output=false,bool HeadMajor=false,bool PrepareHalf=false> inline void kumo_attention_tile_impl(device T* q,device T* k,device T* v,
 device float* out,constant uint* p,uint3 g,uint tid,threadgroup float* scores,
 threadgroup float* probs,threadgroup float* maximum,threadgroup float* denominator,threadgroup float* correction,device half* prepared=nullptr) {
 constexpr uint M=32;
 const float scale=rsqrt(float(D))*(Base2?1.44269504089f:1.f);
 uint H=p[0],Q=p[2],K=p[3],KH=p[6],head=g.x,first=p[7]+g.y*M;
 uint count=min(M,p[8]-g.y*M),kh=(p[4] && first>=p[5])?head/(H/p[4]):head;
 // Head-major storage keeps padding between heads; the visible key count K
 // may be shorter. p[9] is that physical row pitch, never a mask length.
 uint qs=HeadMajor?D:H*D,ks=HeadMajor?D:KH*D,stored_k=HeadMajor?p[9]:K;
 q+=(ulong(g.z)*Q*H+(HeadMajor?head*Q:head))*D;
 k+=(ulong(g.z)*stored_k*KH+(HeadMajor?kh*stored_k:kh))*D;
 v+=(ulong(g.z)*stored_k*KH+(HeadMajor?kh*stored_k:kh))*D;
 auto tq=tensor(q+first*qs,dextents<int,2>{D,int(count)},array<int,2>{1,int(qs)});
 auto ts=tensor(scores,extents<int,N,M>(),array<int,2>{1,N});
 auto tp=tensor(probs,extents<int,N,M>(),array<int,2>{1,N});
 constexpr uint KD=Relaxed?16:D,PVK=Relaxed?16:N;
 constexpr auto qkd=Relaxed?matmul2d_descriptor(M,N,KD,false,true,true,matmul2d_descriptor::mode::multiply_accumulate):matmul2d_descriptor(M,N,D,false,true,false);
 constexpr auto pvd=matmul2d_descriptor(M,D,PVK,false,false,Relaxed,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroups<4>> qk;
 matmul2d<pvd,execution_simdgroups<4>> pv;
 auto tv0=tensor(v,dextents<int,2>{D,N},array<int,2>{1,int(ks)});
 auto acc=pv.template get_destination_cooperative_tensor<decltype(tp),decltype(tv0),float>();
 for(ushort i=0;i<acc.get_capacity();++i)acc[i]=0;
 if(tid<M){maximum[tid]=-INFINITY;denominator[tid]=0;}
 for(uint base=0;base<K;base+=N) {
  uint valid=min(N,K-base);
  auto tk=tensor(k+base*ks,dextents<int,2>{D,int(valid)},array<int,2>{1,int(ks)});
  auto score=qk.template get_destination_cooperative_tensor<decltype(tq),decltype(tk),float>();
  if constexpr (Relaxed) {
   for(ushort i=0;i<score.get_capacity();++i)score[i]=0;
   for(uint d=0;d<D;d+=KD) {
    auto qpart=tensor(q+first*qs+d,dextents<int,2>{KD,int(count)},array<int,2>{1,int(qs)});
    auto kpart=tensor(k+base*ks+d,dextents<int,2>{KD,int(valid)},array<int,2>{1,int(ks)});
    qk.run(qpart,kpart,score);
   }
  } else {qk.run(tq,tk,score);}
  score.store(ts);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint row=tid/4,lane=tid%4;float hi=maximum[row];
  for(uint j=lane;j<N;j+=4)if(j<valid)hi=max(hi,scores[row*N+j]*scale);
  hi=max(hi,simd_shuffle_xor(hi,1));hi=max(hi,simd_shuffle_xor(hi,2));
  float old=isfinite(maximum[row])?(Base2?fast::exp2(maximum[row]-hi):exp(maximum[row]-hi)):0,sum=0;
  for(uint j=lane;j<N;j+=4) {float z=j<valid?(Base2?fast::exp2(scores[row*N+j]*scale-hi):exp(scores[row*N+j]*scale-hi)):0;probs[row*N+j]=z;sum+=z;}
  sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,2);
  if(lane==0){maximum[row]=hi;denominator[row]=denominator[row]*old+sum;correction[row]=old;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(ushort i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]*=correction[acc.get_multidimensional_index(i)[1]];
  auto tv=tensor(v+base*ks,dextents<int,2>{D,int(valid)},array<int,2>{1,int(ks)});
  if constexpr (Relaxed) {
   for(uint j=0;j<N;j+=PVK) {
    auto ppart=tensor(probs+j,dextents<int,2>{PVK,M},array<int,2>{1,N});
    auto vpart=tensor(v+(base+j)*ks,dextents<int,2>{D,int(j<valid?min(PVK,valid-j):0)},array<int,2>{1,int(ks)});
    pv.run(ppart,vpart,acc);
   }
  } else {pv.run(tp,tv,acc);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(ushort i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
  auto ij=acc.get_multidimensional_index(i);if(ij[1]<count){float z=acc[i]/denominator[ij[1]];z=BF16Output?mlx_bf(z):z;out[(ulong(g.z)*Q*H+(first+ij[1])*H+head)*D+ij[0]]=z;
   if constexpr(PrepareHalf)prepared[(ulong(g.z)*((Q+127)/128*128)*H+(first+ij[1])*H+head)*D+ij[0]]=half(z);
  }
 }
 // Diarization's Q8 consumer needs zero-padded F16 rows. The first query
 // tile clears only its head's disjoint tail; no extra scratch/dispatch and
 // no read of stale values from a larger previous window. Other callers
 // compile out this epilogue entirely.
 if constexpr(PrepareHalf)if(g.y==0){uint padded=(Q+127)/128*128;
  for(uint i=tid;i<(padded-Q)*D;i+=128)prepared[(ulong(g.z)*padded*H+(Q+i/D)*H+head)*D+i%D]=half(0);
 }
}
#define KUMO_ATTENTION_TILE(NAME,D) \
kernel void NAME(device float* q [[buffer(0)]],device float* k [[buffer(1)]],device float* v [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup float scores[2048],probs[2048],maximum[32],denominator[32],correction[32]; \
 kumo_attention_tile_impl<D>(q,k,v,out,p,g,tid,scores,probs,maximum,denominator,correction); }
KUMO_ATTENTION_TILE(kumo_attention_tile32,32)
KUMO_ATTENTION_TILE(kumo_attention_tile64,64)
#undef KUMO_ATTENTION_TILE

// A one-key softmax is exactly one. Query projection, normalization, scaling
// and QK products cannot affect the result; preserve the Test-GQA head map.
kernel void kumo_single_value(device const float* v [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 uint H=p[0],D=p[1],Q=p[2],KH=p[6],row=i/(H*D),head=i/D%H,j=i%D;
 if(row>=p[9])return;uint pos=row%Q,batch=row/Q;
 uint kh=(p[4] && pos>=p[5])?head/(H/p[4]):head;
 out[i]=v[(batch*KH+kh)*D+j];
}
kernel void kumo_add(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0])out[i]=a[i]+b[i];
}
// Cell Fourier features are prepared once; projection shares them across all
// output channels. Categorical mask/means are context-fitted host inputs.
kernel void kumo_fourier(device const float* x [[buffer(0)]],device const float* means [[buffer(1)]],
 device const uint* cat [[buffer(2)]],device const float* num [[buffer(3)]],device const float* categorical [[buffer(4)]],
 device float* out [[buffer(5)]],constant uint* p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
 uint R=p[0],C=p[1],f=i%192,row=(i/192)%R,col=i/(192*R);if(col>=C)return;
 uint group=f/64,j=f%64,src=(col+((1u<<group)-1))%C;
 float z=x[row*C+src];if(isnan(z))z=means[src];
 float a=z*(cat[src]?categorical[group*32+j%32]:num[group*32+j%32]);out[i]=j<32?sin(a):cos(a);
}
kernel void kumo_cell_weights(device const uint* cat [[buffer(0)]],device const float* nw [[buffer(1)]],
 device const float* cw [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 uint C=p[0],D=p[1],f=i%192,j=(i/192)%D,col=i/(192*D);if(col>=C)return;
 uint src=(col+((1u<<(f/64))-1))%C;out[i]=(cat[src]?cw:nw)[j*64+f%64];
}
kernel void kumo_cell_mm(device float* w [[buffer(0)]],device float* x [[buffer(1)]],device float* out [[buffer(2)]],
 constant uint* p [[buffer(3)]],uint3 g [[threadgroup_position_in_grid]]) {
 uint R=p[0],D=p[1],n=g.x*32,m=g.y*32,col=g.z;
 auto a=tensor(x+ulong(col)*R*192,dextents<int,2>(192,R),array<int,2>{1,192}).slice(0,m);
 auto b=tensor(w+ulong(col)*D*192,dextents<int,2>(192,D),array<int,2>{1,192}).slice(0,n);
 constexpr auto desc=matmul2d_descriptor(32,32,192,false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();op.run(a,b,acc);
 for(auto it=acc.begin();it!=acc.end();++it)if(it.is_valid_element()) {
  auto ij=it.get_multidimensional_index();uint j=n+ij[0],r=m+ij[1];
  if(j<D && r<R)out[(ulong(col)*R+r)*D+j]=*it;
 }
}
kernel void kumo_cell_bias(device float* out [[buffer(0)]],device const float* x [[buffer(1)]],
 device const uint* cat [[buffer(2)]],device const float* nb [[buffer(3)]],device const float* cb [[buffer(4)]],
 device const float* missing [[buffer(5)]],device const float* target [[buffer(6)]],device const float* y [[buffer(7)]],
 constant uint* p [[buffer(8)]],uint i [[thread_position_in_grid]]) {
 uint R=p[0],C=p[1],D=p[2],j=i%D,r=(i/D)%R,col=i/(D*R);if(col>=C)return;
 float b=0,n=0;for(uint g=0;g<3;++g){uint src=(col+((1u<<g)-1))%C;b+=(cat[src]?cb:nb)[j];if(isnan(x[r*C+src]))n+=missing[j*3+g];}
 float z=(out[i]+b)+n;if(r<p[3])z+=p[4]?target[uint(y[r])*D+j]:target[j]*y[r];out[i]=z;
}
kernel void kumo_copy_repeat(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
 constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[1])out[i]=x[i%p[0]];}
// Pack/unpack the row axis; readout tokens persist between encoder stages.
kernel void kumo_rows(device const float* cells [[buffer(0)]],device const float* cls [[buffer(1)]],device float* out [[buffer(2)]],
 constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 uint R=p[0],C=p[1],D=p[2],j=i%D,c=(i/D)%(C+4),r=i/(D*(C+4));if(r>=R)return;
 out[i]=c<4?cls[(r*4+c)*D+j]:cells[((c-4)*R+r)*D+j];
}
kernel void kumo_unrows(device const float* x [[buffer(0)]],device float* cells [[buffer(1)]],device float* cls [[buffer(2)]],
 constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 uint R=p[0],C=p[1],D=p[2],j=i%D,c=(i/D)%(C+4),r=i/(D*(C+4));if(r>=R)return;
 if(c<4)cls[(r*4+c)*D+j]=x[i];else cells[((c-4)*R+r)*D+j]=x[i];
}
kernel void kumo_labels(device float* x [[buffer(0)]],device const float* y [[buffer(1)]],device const float* target [[buffer(2)]],
 constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 uint D=p[0],r=i/D,j=i%D;if(r<p[1])x[i]+=p[2]?target[uint(y[r])*D+j]:target[j]*y[r];
}
