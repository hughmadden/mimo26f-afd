// Independent synthetic tile fixtures. Not a preparation/staging implementation.
#include "grouped.cu"
namespace {
constexpr uint8_t norm_e4[]={0,0x60,0x68,0x6c,0x70,0x74,0x78,0x7c,0x80,0xe0,0xe8,0xec,0xf0,0xf4,0xf8,0xfc};
int wc(int n,int k,int p){return (n*7+k*3+(k/32)*5+p*3)%16;}
int ws(int n,int kb,int p){return 124+(n+2*kb+p)%7;}
double xv(int row,int k){return std::ldexp(value(acode(row,k)),ascale(row,k/32)-127);}
double wv(int n,int k,int p){return std::ldexp(value(wc(n,k,p)),ws(n,k/32,p)-127);}
struct ComputeFixture {
    int width,slices;
    std::vector<uint32_t> a,b,sb,down,sd;
    std::vector<uint8_t> sa;
    explicit ComputeFixture(int w,bool full_rank=false):width(w),slices(full_rank?512/w:1),a(64*1024),b(slices*32*w*32),sb(slices*32*w*2),down(32*slices*w*16),sd(32*slices*128),sa(64*128){
        for(int row=0;row<64;++row){
            for(int k=0;k<4096;++k)a[row*1024+k/4]|=uint32_t(norm_e4[acode(row,k)])<<(8*(k%4));
            for(int kb=0;kb<128;++kb)sa[row*128+kb]=uint8_t(ascale(row,kb)-6);
        }
        for(int p=0;p<2;++p)for(int n=0;n<w*slices;++n){
            for(int k=0;k<4096;++k){const int kt=k/128,kb=k%128/32,nf=(n%w)/32,warp=n%32/8,lane=(n%8)*4+k%32/8;
                const size_t i=((n/w)*32+kt)*w*32+p*w*16+((kb*(w/32)+nf)*32+lane)*4+warp;
                b[i]|=uint32_t(wc(n,k,p))<<(4*(k%8));}
            for(int kb=0;kb<128;++kb)sb[((n/w)*32+kb/4)*w*2+p*w+n%w]|=uint32_t(ws(n,kb,p))<<(8*(kb%4));
        }
        for(int n=0;n<4096;++n){
            for(int k=0;k<w*slices;++k){const int ot=n/128,nf=n%128/32,warp=n%32/8,lane=(n%8)*4+k%32/8;
                const size_t i=(ot*slices+k/w)*w*16+(((k%w)/32*4+nf)*32+lane)*4+warp;
                down[i]|=uint32_t(wc(n,k,2))<<(4*(k%8));}
            for(int kb=0;kb<w*slices/32;++kb)sd[((n/128)*slices+kb/(w/32))*128+n%128]|=uint32_t(ws(n,kb,2))<<(8*(kb%(w/32)));
        }
    }
};
void compute_host_test(){
    for(int w:{64,128}){const ComputeFixture f(w);
        for(int p=0;p<2;++p)for(int n=0;n<w;++n)for(int k=0;k<4096;++k){
            const int lane=(n%8)*4+k%32/8;
            const auto word=f.b[(k/128)*w*32+p*w*16+((k%128/32*(w/32)+n/32)*32+lane)*4+n%32/8];
            need(((word>>(4*(k%8)))&15)==unsigned(wc(n,k,p)),"FC1 fixture mapping");
        }
    }
    std::puts("HOST PASS compute tile fixtures: widths 64/128, up/gate halves, full K4096");
}
template<int Width>
__global__ void fc1_probe(const uint32_t* a,const uint8_t* sa,const uint32_t* b,const uint32_t* sf,
        float* gate_out,float* up_out,float* mid,int rows,unsigned naive){
    const int group=blockIdx.x,base=group*16,lane=threadIdx.x%32,g=lane/4;
    const int lo=base+g,hi=lo+8;
    m26b1::ActivationView x{a,sa,1024,128,lo<rows?base+(g*5+3)%16:-1,hi<rows?base+((g+8)*5+3)%16:-1};
    float gate[Width/32][4]={},up[Width/32][4]={};
    for(int kt=0;kt<32;++kt)m26b1::fc1_k128<Width>(gate,up,x,kt,b+kt*Width*32,sf+kt*Width*2);
    for(int nf=0;nf<Width/32;++nf)for(int e=0;e<4;++e){
        if(naive==1){const float old=gate[nf][e];gate[nf][e]=up[nf][e];up[nf][e]=old;}
        if(naive==2){gate[nf][e]=fminf(gate[nf][e],10);up[nf][e]=fmaxf(-10,fminf(up[nf][e],10));}
        if(naive==4){gate[nf][e]=m26x::round_acc(gate[nf][e],M26X_NAIVE_BF16_ACCUM);up[nf][e]=m26x::round_acc(up[nf][e],M26X_NAIVE_BF16_ACCUM);}
    }
    m26b1::fc1_finish<Width>(gate,up,min(16,rows-base),mid+base*Width,gate_out+base*Width,up_out+base*Width);
}
template<int Width>
__global__ void fc2_probe(const uint32_t* a,const uint8_t* sa,const uint32_t* b,const uint32_t* sf,float* out,int rows){
    const int ot=blockIdx.x,base=blockIdx.y*16,lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,c=lane%4;
    m26b1::ActivationView x{a,sa,1024,128,base+g<rows?base+(g*5+3)%16:-1,base+g+8<rows?base+((g+8)*5+3)%16:-1};
    float acc[4][4]={};m26b1::fc2_slice<Width>(acc,x,b+ot*Width*16,sf+ot*128);
    for(int nf=0;nf<4;++nf)for(int e=0;e<4;++e){const int row=base+g+(e/2)*8,col=ot*128+nf*32+warp*8+2*c+e%2;
        out[row*4096+col]=row<rows?acc[nf][e]:0;}
}
float host_silu(float g,float u){const float e=std::exp(g>=0?-g:g);return (g>=0?g/(1.0f+e):(g*e)/(1.0f+e))*u;}
bool close_compute(float got,float want,bool padded){
    uint32_t bits;std::memcpy(&bits,&got,4);
    return bits!=0xa5a5a5a5u && std::isfinite(got) && (padded?bits==0:std::abs(double(got)-want)<=1e-5+1e-5*std::abs(want));
}
template<int Width>
size_t compute_width(unsigned naive){
    const ComputeFixture f(Width);
    Buffer da(f.a.size()*4),ds(f.sa.size()),db(f.b.size()*4),dsb(f.sb.size()*4),dd(f.down.size()*4),dsd(f.sd.size()*4);
    da.upload(f.a.data());ds.upload(f.sa.data());db.upload(f.b.data());dsb.upload(f.sb.data());dd.upload(f.down.data());dsd.upload(f.sd.data());
    // Scalar reference tables are independent of the tile-buffer indexing.
    std::vector<float> gate(64*Width),up(64*Width),mid(64*Width),fc2(64*4096);
    for(int row=0;row<64;++row){const int physical=(row/16)*16+((row%16)*5+3)%16;
        for(int n=0;n<Width;++n){double sg=0,su=0;for(int k=0;k<4096;++k){const double x=xv(physical,k);sg+=x*wv(n,k,1);su+=x*wv(n,k,0);}
            gate[row*Width+n]=float(sg);up[row*Width+n]=float(su);mid[row*Width+n]=host_silu(float(sg),float(su));}
        for(int n=0;n<4096;++n){double sum=0;for(int k=0;k<Width;++k)sum+=xv(physical,k)*wv(n,k,2);fc2[row*4096+n]=float(sum);}
    }
    size_t total_bad=0;
    for(int m:{1,2,4,8,16,64}){
        if(naive && m!=4)continue;
        const int padded=(m+15)/16*16;Buffer dg(padded*Width*4),du(padded*Width*4),dh(padded*Width*4),dy(padded*4096*4);
        fc1_probe<Width><<<padded/16,128>>>(static_cast<uint32_t*>(da.data()),static_cast<uint8_t*>(ds.data()),static_cast<uint32_t*>(db.data()),static_cast<uint32_t*>(dsb.data()),static_cast<float*>(dg.data()),static_cast<float*>(du.data()),static_cast<float*>(dh.data()),m,naive);
        ck(cudaGetLastError(),"FC1 probe launch");
        fc2_probe<Width><<<dim3(32,padded/16),128>>>(static_cast<uint32_t*>(da.data()),static_cast<uint8_t*>(ds.data()),static_cast<uint32_t*>(dd.data()),static_cast<uint32_t*>(dsd.data()),static_cast<float*>(dy.data()),m);
        ck(cudaGetLastError(),"FC2 probe launch");ck(cudaDeviceSynchronize(),"compute probes sync");
        std::vector<float> g(padded*Width),u(g.size()),h(g.size()),y(padded*4096);dg.download(g.data());du.download(u.data());dh.download(h.data());dy.download(y.data());
        dg.guard();du.guard();dh.guard();dy.guard();size_t bad=0;
        for(int row=0;row<padded;++row){for(int n=0;n<Width;++n){const auto i=row*Width+n;
            bad+=!close_compute(g[i],row<m?gate[i]:0,row>=m);bad+=!close_compute(u[i],row<m?up[i]:0,row>=m);bad+=!close_compute(h[i],row<m?mid[i]:0,row>=m);}
            for(int n=0;n<4096;++n){const auto i=row*4096+n;bad+=!close_compute(y[i],row<m?fc2[i]:0,row>=m);}}
        total_bad+=bad;
        std::printf("%s B1 compute width=%d M=%d flag=%u bad=%zu active_coordinates=%d padded_rows=%d FC1_K4096 FC2_N4096_K%d; independent projections, not fused FFN\n",bad?"ORACLE_MISMATCH":"PASS",Width,m,naive,bad,m*(Width*3+4096),padded-m,Width);
    }
    da.guard();ds.guard();db.guard();dsb.guard();dd.guard();dsd.guard();return total_bad;
}
__global__ void exceptional_probe(const uint32_t* a,const uint8_t* sa,float* output,bool bypass){
    const int test=blockIdx.x,lane=threadIdx.x,g=lane/4,c=lane%4;
    const unsigned scales[]={0,1,253,254,255};
    m26b1::ActivationView x{a+test*16*8,sa+test*16,8,1,g,g+8};
    const auto af=m26b1::load_a(x,0,c);uint32_t packed=0;
    for(int j=0;j<8;++j)packed|=uint32_t(((g+c*8+j)%8)|(g%2?8:0))<<(4*j);
    const uint32_t sb=127u|(127u<<8)|(scales[test]<<16)|(127u<<24);float d[4]={};
    if(bypass){const auto b=m26b1::e2m1_containers(packed);m26b1::mma_e4m3_e2m1<0,2>(d,af.a0,af.a1,af.a2,af.a3,b.x,b.y,af.scale,sb);}
    else m26b1::dot32<2>(d,x,0,af,packed,sb);
    for(int e=0;e<4;++e)output[test*128+(g+e/2*8)*8+c*2+e%2]=d[e];
}
size_t exceptional_test(bool bypass){
    std::vector<uint32_t> a(5*16*8);std::vector<uint8_t> sa(5*16);
    const int scales[]={0,1,253,254,255};
    for(int t=0;t<5;++t)for(int row=0;row<16;++row){sa[t*16+row]=uint8_t((t<2?239:105)+row%2);
        for(int k=0;k<32;++k){const unsigned code=k%8==0?0x80:t<2?0x78:0x40;a[(t*16+row)*8+k/4]|=code<<(8*(k%4));}}
    Buffer da(a.size()*4),ds(sa.size()),out(5*128*4);da.upload(a.data());ds.upload(sa.data());
    exceptional_probe<<<5,32>>>(static_cast<uint32_t*>(da.data()),static_cast<uint8_t*>(ds.data()),static_cast<float*>(out.data()),bypass);
    ck(cudaGetLastError(),"exception launch");ck(cudaDeviceSynchronize(),"exception sync");
    std::vector<float> got(5*128);out.download(got.data());da.guard();ds.guard();out.guard();size_t total=0;
    const double limit=std::numeric_limits<float>::max();
    for(int t=0;t<5;++t){size_t bad=0;for(int row=0;row<16;++row)for(int n=0;n<8;++n){double sum=0;
        for(int k=0;k<32;++k){const double av=k%8==0?-0.0:std::ldexp(t<2?256.:2.,int(sa[t*16+row])-127);
            const double w=std::ldexp(value(((n+k)%8)|(n%2?8:0)),std::min(scales[t],254)-127);
            const float decoded=float(std::max(-limit,std::min(limit,w)));sum+=av*double(decoded);}
        bad+=!close_compute(got[t*128+row*8+n],float(sum),false);}
        total+=bad;std::printf("%s B1 exceptional weight_scale=%d bypass=%d bad=%zu/128; conforming decoded-weight FMA\n",bad?"ORACLE_MISMATCH":"PASS",scales[t],bypass,bad);
    }return total;
}
int compute_test(unsigned naive){gpu_ready();size_t bad=0;
    if(naive!=8){bad+=compute_width<64>(naive);bad+=compute_width<128>(naive);}
    if(naive==0 || naive==8)bad+=exceptional_test(naive==8);
    return bad?3:0;
}
} // namespace
