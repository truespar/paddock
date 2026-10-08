// Original E2B-geometry image seams; the bounded online tensor attention
// shares Paddock's split-query implementation, never a quadratic score buffer.
kernel void eg2v_bmm128(device bfloat* w [[buffer(0)]],device float* x [[buffer(1)]],
    device uint* out [[buffer(2)]],device const float* bias [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]]) {vis_project<128,bfloat,float,true>(w,x,out,bias,p,g);}
// MLX convolution tensors are NHWC; the shared subsampler uses GGUF's OIHW.
kernel void eg2_media_mlx_cast(device const bfloat* x [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0])return;uint ci=p[1];
    float v=float(x[(i/(9*ci))*9*ci+(i%9)*ci+(i/9)%ci]);
    y[i]=p[2]?mlx_bf(log(1.0f+exp(v))):v;
}
kernel void eg2v_mlx_patches(device const uchar* rgb [[buffer(0)]],device float* x [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/768,d=i%768,pw=p[0]/16;if(row>=pw*(p[1]/16))return;
    uint xx=(row%pw)*16+(d/3)%16,yy=(row/pw)*16+d/48,c=d%3;
    x[i]=mlx_bf(2.0f*(float(rgb[(yy*p[0]+xx)*3+c])*(1.0f/255.0f)-0.5f));
}
kernel void eg2v_mlx_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/768,d=i%768;if(row>=p[0]*p[1])return;
    x[i]=mlx_bf(x[i]+mlx_bf(pos[(row%p[0])*768+d]+pos[(10240+row/p[0])*768+d]));
}
kernel void eg2v_mlx_qkv(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],device const float* qw [[buffer(3)]],device const float* kw [[buffer(4)]],
    device bfloat* qo [[buffer(5)]],device bfloat* ko [[buffer(6)]],device bfloat* vo [[buffer(7)]],
    constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    ulong b=ulong(g.y)*p[1]+g.x*64,o=(ulong(g.y)*12+g.x)*64;float qs=0,ks=0,vs=0;
    for(uint j=lane;j<64;j+=32){qs+=q[b+j]*q[b+j];ks+=k[b+j]*k[b+j];vs+=v[b+j]*v[b+j];}
    float qi=precise::rsqrt(simd_sum(qs)/64.0f+1e-6f),ki=precise::rsqrt(simd_sum(ks)/64.0f+1e-6f),vi=precise::rsqrt(simd_sum(vs)/64.0f+1e-6f);
    for(uint j=lane;j<64;j+=32){uint local=j%32,other=(j/32)*32+(local+16)%32;
        float angle=precise::divide(float(j<32?g.y%p[0]:g.y/p[0]),pow(100.0f,float(local%16)/16.0f));
        float cs=mlx_bf(cos(angle)),sn=mlx_bf(sin(angle))*(local<16?-1.0f:1.0f);
        qo[o+j]=bfloat(mlx_bf(mlx_bf(mlx_bf(q[b+j]*qi*qw[j])*cs)+mlx_bf(mlx_bf(q[b+other]*qi*qw[other])*sn)));
        ko[o+j]=bfloat(mlx_bf(mlx_bf(mlx_bf(k[b+j]*ki*kw[j])*cs)+mlx_bf(mlx_bf(k[b+other]*ki*kw[other])*sn)));
        vo[o+j]=bfloat(v[b+j]*vi);}
}
kernel void eg2v_mlx_attention(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],device bfloat* v [[buffer(2)]],
    device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<true,64,64,12,false,false,bfloat>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
kernel void eg2v_mlx_pool(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/768,d=i%768,nx=p[0]/3,ny=p[1]/3;if(row>=nx*ny)return;float sum=0;
    for(uint y=0;y<3;++y)for(uint xx=0;xx<3;++xx)sum+=x[(((row/nx)*3+y)*p[0]+(row%nx)*3+xx)*768+d]*(1.0f/9.0f);
    out[i]=mlx_bf(mlx_bf(sum)*mlx_bf(sqrt(768.0f)));
}
kernel void eg2v_patches(device const uchar* rgb [[buffer(0)]],device float* x [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/768,d=i%768,pw=p[0]/16;if(row>=pw*(p[1]/16))return;
    uint xx=(row%pw)*16+d%16,yy=(row/pw)*16+(d%256)/16,c=d/256;
    x[i]=float(half(float(rgb[(yy*p[0]+xx)*3+c])/255.0f*2.0f-1.0f));
}
kernel void eg2v_position(device float* x [[buffer(0)]],device const float* pos [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/768,d=i%768;if(row>=p[0]*p[1])return;
    x[i]=(x[i]+pos[(row%p[0])*768+d])+pos[(10240+row/p[0])*768+d];
}
kernel void eg2v_qkv(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],device const float* qw [[buffer(3)]],device const float* kw [[buffer(4)]],
    device half* qo [[buffer(5)]],device half* ko [[buffer(6)]],device half* vo [[buffer(7)]],
    constant uint* p [[buffer(8)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    ulong b=ulong(g.y)*p[1]+g.x*64,o=(ulong(g.y)*12+g.x)*64;float qs=0,ks=0,vs=0;
    for(uint j=lane;j<64;j+=32){qs+=q[b+j]*q[b+j];ks+=k[b+j]*k[b+j];vs+=v[b+j]*v[b+j];}
    float qi=rsqrt(simd_sum(qs)/64.0f+1e-6f),ki=rsqrt(simd_sum(ks)/64.0f+1e-6f),vi=rsqrt(simd_sum(vs)/64.0f+1e-6f);
    for(uint j=lane;j<64;j+=32){uint local=j%32,other=(j/32)*32+(local+16)%32;
        float angle=float(j<32?g.y%p[0]:g.y/p[0])*pow(100.0f,-float(local%16)/16.0f);
        float cs=cos(angle),sn=sin(angle)*(local<16?-1.0f:1.0f);
        qo[o+j]=half(q[b+j]*qi*qw[j]*cs+q[b+other]*qi*qw[other]*sn);
        ko[o+j]=half(k[b+j]*ki*kw[j]*cs+k[b+other]*ki*kw[other]*sn);
        vo[o+j]=half(v[b+j]*vi);}
}
kernel void eg2v_gate_up(device const float* gu [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    if(i>=p[0])return;uint at=(i/3072)*6144+i%3072;float x=gu[at],up=gu[at+3072];
    if(!p[1]){out[i]=vis_gelu(x)*up;return;}
    float cube=mlx_bf(pow(x,3.0f)),z=mlx_bf(x+mlx_bf(mlx_bf(0.044715f)*cube));
    z=mlx_bf(mlx_bf(0.7978845608028654f)*z);z=mlx_bf(1.0f+mlx_bf(precise::tanh(z)));
    out[i]=mlx_bf(mlx_bf(mlx_bf(0.5f*x)*z)*up);
}
kernel void eg2v_attention(device half* q [[buffer(0)]],device half* k [[buffer(1)]],device half* v [[buffer(2)]],
    device ushort* out [[buffer(3)]],device const uint4* tiles [[buffer(4)]],constant uint* p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float correction[32],normalizer[32],remap[32*64];
    vis_attention_impl<true,64,64,12>(q,k,v,out,tiles,p,g,tid,correction,normalizer,remap);
}
kernel void eg2v_pool(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/768,d=i%768,nx=p[0]/3,ny=p[1]/3;if(row>=nx*ny)return;float sum=0;
    for(uint y=0;y<3;++y)for(uint xx=0;xx<3;++xx)sum+=x[(((row/nx)*3+y)*p[0]+(row%nx)*3+xx)*768+d];
    out[i]=(sum/9.0f)*sqrt(768.0f);
}
