// Original SAM 3 input seams, matching Paddock CUDA's image.cuh/video.cuh.
// Pictures use torchvision F32 AA bilinear with ONE final uint8 rounding;
// video frames use Pillow's integer coefficients and uint8 BETWEEN axes.
// The two graphs must not be folded into a generic chat-image processor.
kernel void sam3_resize_coeff(device uint2* ranges [[buffer(0)]],device float* weights [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint o [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(o>=p[1])return;
    float scale=as_type<float>(p[3]),inv=as_type<float>(p[4]),support=max(scale,1.f),ih=float(o)+0.5f;
    int lo=max(int(fma(scale,ih,-support)+0.5f),0);
    int hi=min(int(fma(scale,ih,support)+0.5f),int(p[0]));
    ranges[o]=uint2(lo,hi-lo);
    float offset=fma(-scale,ih,float(lo)),total=0;
    for(int j=0;j<hi-lo;++j) {
        float x=abs((float(j)+offset+0.5f)*inv),v=x<1.f?1.f-x:0.f;
        weights[o*p[2]+uint(j)]=v;total+=v;
    }
    for(int j=0;j<hi-lo;++j)weights[o*p[2]+uint(j)]=precise::divide(weights[o*p[2]+uint(j)],total);
}

// p: input extent, output extent, other extent, horizontal, taps.
// Horizontal output is F32; vertical output is uint8, ties-to-even.
kernel void sam3_resize_image(device const uchar* src [[buffer(0)]],device uchar* dst [[buffer(1)]],
    device const uint2* ranges [[buffer(2)]],device const float* weights [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(i>=p[1]*p[2]*3)return;
    uint o=p[3]?(i/3)%p[1]:i/(p[2]*3);
    uint base=p[3]?((i/3/p[1])*p[0]+ranges[o].x)*3+i%3:ranges[o].x*p[2]*3+i%(p[2]*3);
    uint stride=p[3]?3:p[2]*3;device const float* w=weights+o*p[4];
    float sum=(p[3]?float(src[base]):reinterpret_cast<device const float*>(src)[base])*w[0];
    for(uint j=1;j<ranges[o].y;++j) {
        float value=p[3]?float(src[base+j*stride]):reinterpret_cast<device const float*>(src)[base+j*stride];
        sum=fma(value,w[j],sum);
    }
    if(p[3])reinterpret_cast<device float*>(dst)[i]=sum;
    else dst[i]=uchar(clamp(rint(sum),0.f,255.f));
}

kernel void sam3_resize_video(device const uchar* src [[buffer(0)]],device uchar* dst [[buffer(1)]],
    device const uint2* ranges [[buffer(2)]],device const int* weights [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[1]*p[2]*3)return;
    uint o=p[3]?(i/3)%p[1]:i/(p[2]*3);
    uint base=p[3]?((i/3/p[1])*p[0]+ranges[o].x)*3+i%3:ranges[o].x*p[2]*3+i%(p[2]*3);
    uint stride=p[3]?3:p[2]*3;int sum=1<<21;
    for(uint j=0;j<ranges[o].y;++j)sum+=int(src[base+j*stride])*weights[o*p[4]+j];
    dst[i]=uchar(clamp(sum>>22,0,255));
}
