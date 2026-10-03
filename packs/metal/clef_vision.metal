// Original Clef Metal image kernels. Preserve the model's byte-valued AA
// resize and unrounded activations; never use the chat mtmd image contract.
kernel void clef_vis_resize(device const uchar* src [[buffer(0)]],device uchar* dst [[buffer(1)]],
 device const uint2* ranges [[buffer(2)]],device const int* weights [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[1]*p[2]*3)return;uint o=p[3]?(i/3)%p[1]:i/(p[2]*3);
 uint base=p[3]?((i/3/p[1])*p[0]+ranges[o].x)*3+i%3:ranges[o].x*p[2]*3+i%(p[2]*3);
 uint stride=p[3]?3:p[2]*3;int sum=1<<(p[4]-1);
 for(uint j=0;j<ranges[o].y;++j)sum+=int(src[base+j*stride])*int(weights[o*p[5]+j]);
 dst[i]=uchar(clamp(sum>>p[4],0,255));
}
kernel void clef_vis_patch_weight(device const bfloat* src [[buffer(0)]],device bfloat* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0])return;uint c=i%1536/512,t=i%512/256,pixel=i%256;
 out[i]=src[(i/1536)*1536+t*768+pixel*3+c];
}
kernel void clef_vis_patches(device const uchar* rgb [[buffer(0)]],device float* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 #pragma clang fp contract(off)
 uint row=i/1536,d=i%1536;if(row>=(p[0]/16)*(p[1]/16))return;
 uint y=(row/4/(p[0]/32)*2+row%4/2)*16+(d%256)/16,x=(row/4%(p[0]/32)*2+row%2)*16+d%16;
 out[i]=(float(rgb[(y*p[0]+x)*3+d/512])-127.5f)/127.5f;
}
inline float clef_vis_linspace(uint n,uint i) { float step=47.f/float(n-1);return i<n/2?step*float(i):fma(-step,float(n-1-i),47.f); }
kernel void clef_vis_position(device float* x [[buffer(0)]],device const bfloat* pos [[buffer(1)]],device const uint4* info [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 #pragma clang fp contract(off)
 if(i>=p[0]*1152)return;uint4 a=info[i/1152];uint d=i%1152;float y=clef_vis_linspace(a.z,a.x),xx=clef_vis_linspace(a.w,a.y);
 uint y0=uint(y),x0=uint(xx),y1=min(y0+1,47u),x1=min(x0+1,47u);float fy=y-float(y0),fx=xx-float(x0);
 float z=float(pos[(y0*48+x0)*1152+d])*((1.f-fy)*(1.f-fx));
 z+=float(pos[(y0*48+x1)*1152+d])*((1.f-fy)*fx);z+=float(pos[(y1*48+x0)*1152+d])*(fy*(1.f-fx));z+=float(pos[(y1*48+x1)*1152+d])*(fy*fx);x[i]+=z;
}
// Separate destination avoids partner races. Zero padding permits exact F32
// online attention using the existing bounded Clef reduction geometry.
kernel void clef_vis_qkv(device const float* x [[buffer(0)]],device const uint4* info [[buffer(1)]],device const float2* rope [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 #pragma clang fp contract(off)
 uint row=i/1536,h=i%1536/96,d=i%96;if(row>=p[0])return;float q=0,k=0,v=0;
 if(d<72){uint pair=d%36,other=(d+36)%72;uint pos=pair<18?info[row].x:info[row].y;float2 a=rope[pos*18+pair%18];float sn=d<36?-a.y:a.y;ulong b=ulong(row)*3456+h*72;
 q=x[b+d]*a.x+x[b+other]*sn;k=x[b+1152+d]*a.x+x[b+1152+other]*sn;v=x[b+2304+d];}
 out[i]=q;out[p[0]*1536+i]=k;out[2*p[0]*1536+i]=v;
}
kernel void clef_vis_unpad(device const float* src [[buffer(0)]],device float* out [[buffer(1)]],constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<p[0]*1152)out[i]=src[(i/1152)*1536+(i%1152/72)*96+i%72];
}
