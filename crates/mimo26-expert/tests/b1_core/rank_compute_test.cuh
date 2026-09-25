// Complete rank-local projection tests; still no quantizer or fused FFN.
namespace {
void rank_host_test(){
    for(int w:{64,128}){const ComputeFixture f(w,true);
        for(int p=0;p<2;++p)for(int n=0;n<512;++n)for(int k=0;k<4096;++k){
            const int lane=(n%8)*4+k%32/8;
            const auto word=f.b[((n/w)*32+k/128)*w*32+p*w*16+((k%128/32*(w/32)+(n%w)/32)*32+lane)*4+n%32/8];
            need(((word>>(4*(k%8)))&15)==unsigned(wc(n,k,p)),"rank FC1 fixture mapping");
        }
        for(int n=0;n<4096;++n)for(int k=0;k<512;++k){
            const int lane=(n%8)*4+k%32/8;
            const auto word=f.down[((n/128)*f.slices+k/w)*w*16+(((k%w)/32*4+n%128/32)*32+lane)*4+n%32/8];
            need(((word>>(4*(k%8)))&15)==unsigned(wc(n,k,2)),"rank FC2 fixture mapping");
            const auto scale=f.sd[((n/128)*f.slices+k/w)*128+n%128];
            need(((scale>>(8*((k%w)/32)))&255)==unsigned(ws(n,k/32,2)),"rank FC2 scale mapping");
        }
    }
    std::puts("HOST PASS rank fixtures: N512/K4096 FC1 and N4096/K512 FC2, widths 64/128");
}
template<int Width>
__global__ void rank_fc1_probe(const uint32_t* a,const uint8_t* sa,const uint32_t* b,const uint32_t* sf,
        float* gout,float* uout,float* mid,int rows,unsigned naive){
    const int slice=blockIdx.x,base=blockIdx.y*16,g=(threadIdx.x%32)/4;
    m26b1::ActivationView x{a,sa,1024,128,base+g<rows?base+(g*5+3)%16:-1,base+g+8<rows?base+((g+8)*5+3)%16:-1};
    float gate[Width/32][4]={},up[Width/32][4]={};
    for(int kt=0;kt<32;++kt)m26b1::fc1_k128<Width>(gate,up,x,kt,b+(slice*32+kt)*Width*32,sf+(slice*32+kt)*Width*2);
    // Negative permutes disjoint column slices; it does not introduce a race.
    const int col=(naive==1?(slice+1)%(512/Width):slice)*Width,off=base*512+col;
    m26b1::fc1_finish<Width>(gate,up,min(16,rows-base),mid+off,gout+off,uout+off,512);
}
template<int Width>
__global__ void rank_fc2_probe(const uint32_t* a,const uint8_t* sa,const uint32_t* b,const uint32_t* sf,float* out,int rows,unsigned naive){
    const int ot=blockIdx.x,base=blockIdx.y*16,lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,c=lane%4;
    const m26b1::ActivationView whole{a,sa,1024,128,base+g<rows?base+(g*5+3)%16:-1,base+g+8<rows?base+((g+8)*5+3)%16:-1};
    float acc[4][4]={};
    for(int slice=0;slice<512/Width;++slice){
        if(naive==2)for(int nf=0;nf<4;++nf)for(int e=0;e<4;++e)acc[nf][e]=0;
        auto x=whole;
        if(naive!=4){x.payload+=slice*(Width/4);x.scales+=slice*(Width/32);}
        m26b1::fc2_slice<Width>(acc,x,b+(ot*(512/Width)+slice)*Width*16,sf+(ot*(512/Width)+slice)*128);
    }
    for(int nf=0;nf<4;++nf)for(int e=0;e<4;++e){const int row=base+g+(e/2)*8,col=ot*128+nf*32+warp*8+2*c+e%2;
        out[row*4096+col]=row<rows?acc[nf][e]:0;}
}
struct RankReference {
    std::vector<float> gate,up,mid,down;
    explicit RankReference(int rows):gate(rows*512),up(rows*512),mid(rows*512),down(rows*4096){
        std::vector<double> x(64*4096),wg(512*4096),wu(wg.size()),wd(4096*512);
        for(int row=0;row<64;++row)for(int k=0;k<4096;++k)x[row*4096+k]=xv(row,k);
        for(int n=0;n<512;++n)for(int k=0;k<4096;++k){wg[n*4096+k]=wv(n,k,1);wu[n*4096+k]=wv(n,k,0);}
        for(int n=0;n<4096;++n)for(int k=0;k<512;++k)wd[n*512+k]=wv(n,k,2);
        for(int row=0;row<rows;++row){const int physical=(row/16)*16+((row%16)*5+3)%16;
            for(int n=0;n<512;++n){double g=0,u=0;for(int k=0;k<4096;++k){const auto av=x[physical*4096+k];g+=av*wg[n*4096+k];u+=av*wu[n*4096+k];}
                gate[row*512+n]=float(g);up[row*512+n]=float(u);mid[row*512+n]=host_silu(float(g),float(u));}
            for(int n=0;n<4096;++n){double sum=0;for(int k=0;k<512;++k)sum+=x[physical*4096+k]*wd[n*512+k];down[row*4096+n]=float(sum);}
        }
    }
};
template<int Width>
size_t rank_width(const RankReference& ref,unsigned naive){
    const ComputeFixture f(Width,true);
    Buffer da(f.a.size()*4),ds(f.sa.size()),db(f.b.size()*4),dsb(f.sb.size()*4),dd(f.down.size()*4),dsd(f.sd.size()*4);
    da.upload(f.a.data());ds.upload(f.sa.data());db.upload(f.b.data());dsb.upload(f.sb.data());dd.upload(f.down.data());dsd.upload(f.sd.data());
    size_t total=0;
    for(int m:{1,2,4,8,16,64}){
        if(naive && m!=4)continue;
        const int padded=(m+15)/16*16;Buffer dg(padded*512*4),du(padded*512*4),dh(padded*512*4),dy(padded*4096*4);
        rank_fc1_probe<Width><<<dim3(512/Width,padded/16),128>>>(static_cast<uint32_t*>(da.data()),static_cast<uint8_t*>(ds.data()),static_cast<uint32_t*>(db.data()),static_cast<uint32_t*>(dsb.data()),static_cast<float*>(dg.data()),static_cast<float*>(du.data()),static_cast<float*>(dh.data()),m,naive);
        ck(cudaGetLastError(),"rank FC1 launch");
        rank_fc2_probe<Width><<<dim3(32,padded/16),128>>>(static_cast<uint32_t*>(da.data()),static_cast<uint8_t*>(ds.data()),static_cast<uint32_t*>(dd.data()),static_cast<uint32_t*>(dsd.data()),static_cast<float*>(dy.data()),m,naive);
        ck(cudaGetLastError(),"rank FC2 launch");ck(cudaDeviceSynchronize(),"rank projection sync");
        std::vector<float> g(padded*512),u(g.size()),h(g.size()),y(padded*4096);dg.download(g.data());du.download(u.data());dh.download(h.data());dy.download(y.data());
        dg.guard();du.guard();dh.guard();dy.guard();size_t bad=0;double max_abs=0;
        for(int row=0;row<padded;++row){for(int n=0;n<512;++n){const int i=row*512+n;
            bad+=!close_compute(g[i],row<m?ref.gate[i]:0,row>=m);bad+=!close_compute(u[i],row<m?ref.up[i]:0,row>=m);bad+=!close_compute(h[i],row<m?ref.mid[i]:0,row>=m);}
            for(int n=0;n<4096;++n){const int i=row*4096+n;const float want=row<m?ref.down[i]:0;
                bad+=!close_compute(y[i],want,row>=m);max_abs=std::max(max_abs,std::abs(double(y[i])-want));}}
        total+=bad;
        std::printf("%s B1 rank width=%d M=%d flag=%u bad=%zu active_coordinates=%d padded_rows=%d FC2_maxabs=%.12g FC1_N512_K4096 FC2_N4096_K512; no fused FFN\n",bad?"ORACLE_MISMATCH":"PASS",Width,m,naive,bad,m*(512*3+4096),padded-m,max_abs);
    }
    da.guard();ds.guard();db.guard();dsb.guard();dd.guard();dsd.guard();return total;
}
int rank_test(unsigned naive){gpu_ready();const RankReference ref(naive?4:64);
    const auto bad=rank_width<64>(ref,naive)+rank_width<128>(ref,naive);return bad?3:0;
}
} // namespace
