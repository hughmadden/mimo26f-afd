// P1 M64/N16: four GA or eight SWA query tokens per CTA/KV head.
// Included inside attn_decode_tc.cu's anonymous namespace. Reuses only the
// existing original MMA/layout/codec helpers; all decode bodies stay separate.
// Synchronous staging: no outstanding async work, role handoff or early exit.
// Split=true is the P1 trial-1 occupancy variant: P1M 64->32, threads 256->128,
// grid.x doubles (qt=P1M/rep halves). The S-axis online-softmax chain stays
// inside one CTA, so the split is bitwise-identical to the baseline (Split=false).
template<bool QResidual,bool Split=false>
__global__ void prefill_tc(m26_geom g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int page_tokens,const int64_t* qpos,const int64_t* kpos,
    int T,int S,uint32_t naive,const float* sink,float* out) {
  using Storage=m26tc::PrefillStorage<BF,QResidual,Split>;
  static_assert(sizeof(Storage)==(Split?48192:(QResidual?88128:38976)),"P1 CUDA storage matches CPU model");
  extern __shared__ __align__(16) unsigned char p1_shared[];
  Storage& sh=*reinterpret_cast<Storage*>(p1_shared);
  constexpr int PM=Split?m26tc::P1M_SPLIT:m26tc::P1M,PN=m26tc::P1N,NT=Split?m26tc::P1Threads_SPLIT:m26tc::P1Threads;
  constexpr int QT=QResidual?3:1;
  int tid=threadIdx.x,lane=tid&31,warp=tid/32,rep=64/g.n_kv;
  int base=blockIdx.x*(PM/rep),kvh=blockIdx.y,mrow=(warp/2)*16;
  int64_t window=g.window;
  if((naive&M26_NAIVE_GA_WINDOWED)&&window<=0)window=128;
  float scale=float(m26::attn_scale(192,128,naive));
  if(tid<PM) {
    int t=base+tid/rep;
    sh.qp[tid]=t<T?qpos[t]:0;
    sh.q_bad[tid]=sh.row_bad[tid]=0;
    sh.maximum[tid]=-INFINITY;sh.sum[tid]=0;
  }
  __syncthreads(); // Initialize per-row poison state before cooperative Q loads.
  for(int i=tid;i<PM*192;i+=NT) {
    int r=i/192,d=i%192,t=base+r/rep,h=kvh*rep+r%rep;
    float x=t<T?q[((int64_t)t*64+h)*192+d]:0;
    BF high=__float2bfloat16_rn(x);
    if(!isfinite(x)||!isfinite(__bfloat162float(high))) {
      atomicOr(sh.q_bad+r,1);x=0;high=__float2bfloat16_rn(0.f);
    }
    int at=m26tc::tile_index(PM,r,d);sh.q[0][at]=high;
    if constexpr(QResidual) {
      float residual=x-__bfloat162float(high);BF low=__float2bfloat16_rn(residual);
      sh.q[1][at]=low;sh.q[2][at]=__float2bfloat16_rn(residual-__bfloat162float(low));
    }
  }
  // Eight output column fragments per warp, separate P-high/P-low chains.
  float output[8][2][4]={};
  __syncthreads(); // Publish immutable Q and its row-local nonfinite flags.
  for(int64_t first=0;first<S;first+=PN) {
    bool live=false;
    if(tid<PN) {
      int64_t j=first+tid;
      sh.key_bad[tid]=0;sh.kp[tid]=j<S?kpos[j]:0;
      if(j<S) {
        for(int r=0;r<PM;r+=rep)
          live|=base+r/rep<T&&m26::is_visible(sh.qp[r],sh.kp[tid],window,naive);
        sh.physical[tid]=(pages&&!(naive&M26_NAIVE_TC_IGNORE_PAGES))?
          pages[j/page_tokens]*page_tokens+j%page_tokens:int(j);
      } else sh.physical[tid]=0;
      sh.visible[tid]=live;
    }
    // Uniform skip: no row can observe this tile. Does not inspect hidden KV.
    if(!__syncthreads_or(live))continue;
    for(int i=tid;i<PN*192/2;i+=NT) {
      int r=m26tc::producer_row(i,192),d=m26tc::producer_col(i,192);
      uint16_t codes=sh.visible[r]?*reinterpret_cast<const uint16_t*>(kc+((int64_t)sh.physical[r]*g.n_kv+kvh)*192+d):0;
      if((codes&127)==127){atomicOr(sh.key_bad+r,1);codes&=0xff00;}
      if(((codes>>8)&127)==127){atomicOr(sh.key_bad+r,1);codes&=0x00ff;}
      *reinterpret_cast<uint32_t*>(sh.phase.k+m26tc::tile_index(PN,r,d))=m26tc::e4m3x2_bf16(codes);
    }
    for(int i=tid;i<PN*128/2;i+=NT) {
      int r=(i/128)*2,d=i%128;
      uint16_t a=sh.visible[r]?vc[((int64_t)sh.physical[r]*g.n_kv+kvh)*128+d]:0;
      uint16_t b=sh.visible[r+1]?vc[((int64_t)sh.physical[r+1]*g.n_kv+kvh)*128+d]:0;
      if((a&127)==127){atomicOr(sh.key_bad+r,1);a=0;}
      if((b&127)==127){atomicOr(sh.key_bad+r+1,1);b=0;}
      *reinterpret_cast<uint32_t*>(sh.v+m26tc::tile_index(128,d,r))=m26tc::e4m3x2_bf16(a|(b<<8));
    }
    // Scrub NaN operands while preserving token flags: a key visible to one
    // query must not poison a different, masked query through 0 * NaN in PV.
    __syncthreads(); // Publish K/V and token poison flags.
    float score[QT][2][4]={};
#pragma unroll
    for(int d=0;d<192;d+=16) {
      uint32_t a[4],b[2];int group=m26tc::q_acc_group(d/16);
      load_b(sh.phase.k,PN,(warp&1)*8,d,b);
      load_a(sh.q[0]+mrow*16,PM,d,a);mma(score[0][group],a,b);
      if constexpr(QResidual) {
        if(!(naive&M26_NAIVE_TC_DROP_Q_LOW)) {
          load_a(sh.q[1]+mrow*16,PM,d,a);mma(score[1][group],a,b);
          if(!(naive&M26_NAIVE_TC_DROP_Q_TAIL)){load_a(sh.q[2]+mrow*16,PM,d,a);mma(score[2][group],a,b);}
        }
      }
    }
    float final_score[4];
#pragma unroll
    for(int e=0;e<4;++e) {
      int r=mrow+m26tc::c_row(lane,e),c=(warp&1)*8+m26tc::c_col(lane,e);
      float total=score[0][0][e]+score[0][1][e];
      if constexpr(QResidual){total+=score[1][0][e]+score[1][1][e];total+=score[2][0][e]+score[2][1][e];}
      float s=total*scale;
      bool visible=base+r/rep<T&&first+c<S&&m26::is_visible(sh.qp[r],sh.kp[c],window,naive);
      bool bad=visible&&(sh.q_bad[r]||sh.key_bad[c]||!isfinite(s));
      if(bad)atomicOr(sh.row_bad+r,1);
      final_score[e]=visible&&!bad?s:-INFINITY;
    }
    __syncthreads(); // Retire ALL QK reads before K -> score/P alias transition.
#pragma unroll
    for(int e=0;e<4;++e)
      sh.phase.p.score[(mrow+m26tc::c_row(lane,e))*PN+(warp&1)*8+m26tc::c_col(lane,e)]=final_score[e];
    __syncthreads(); // Publish all score rows before four-lane row reductions.
    int r=tid/4,sub=tid%4;
    float ss[4],row_max=-INFINITY;
#pragma unroll
    for(int k=0;k<4;++k){ss[k]=sh.phase.p.score[r*PN+sub+k*4];row_max=fmaxf(row_max,ss[k]);}
#pragma unroll
    for(int d=2;d;d/=2)row_max=fmaxf(row_max,__shfl_xor_sync(0xffffffff,row_max,d,4));
    float old=sh.maximum[r],next=fmaxf(old,row_max);
    float alpha=(naive&M26_NAIVE_NO_RUNNING_RESCALE)?1.f:(isfinite(old)?expf(old-next):0.f),l=0;
#pragma unroll
    for(int k=0;k<4;++k) {
      float p=isfinite(ss[k])?expf(ss[k]-next):0;BF high=__float2bfloat16_rn(p);
      int at=m26tc::tile_index(PM,r,sub+k*4);sh.phase.p.ph[at]=high;
      sh.phase.p.pl[at]=__float2bfloat16_rn(p-__bfloat162float(high));l+=p;
    }
#pragma unroll
    for(int d=2;d;d/=2)l+=__shfl_xor_sync(0xffffffff,l,d,4);
    if(!sub){sh.maximum[r]=next;sh.sum[r]=sh.sum[r]*alpha+l;sh.phase.p.alpha[r]=alpha;}
    __syncthreads(); // Publish P and alpha to all PV consumers.
#pragma unroll
    for(int c=0;c<8;++c) {
#pragma unroll
      for(int e=0;e<4;++e) {
        float alpha=sh.phase.p.alpha[mrow+m26tc::c_row(lane,e)];
        output[c][0][e]*=alpha;output[c][1][e]*=alpha;
      }
      uint32_t a[4],b[2];load_b(sh.v,128,(warp&1)*64+c*8,0,b);
      load_a(sh.phase.p.ph+mrow*16,PM,0,a);mma(output[c][0],a,b);
      if(!(naive&M26_NAIVE_TC_DROP_P_LOW)){load_a(sh.phase.p.pl+mrow*16,PM,0,a);mma(output[c][1],a,b);}
    }
    __syncthreads(); // Retire P/alpha AND V before next tile overwrites them.
  }
  if(tid<PM) {
    int h=kvh*rep+tid%rep;
    bool on=sink&&(g.window>0||(naive&M26_NAIVE_SINK_ON_GA));
    float alpha=1,l=sh.sum[tid];
    if(on&&base+tid/rep<T) {
      float bias=sink[(naive&M26_NAIVE_SINK_PER_KV)?kvh:h];
      if(!isfinite(bias))sh.row_bad[tid]=1;
      float next=fmaxf(sh.maximum[tid],bias);
      alpha=isfinite(sh.maximum[tid])?expf(sh.maximum[tid]-next):0;
      l=l*alpha+expf(bias-next)*((naive&M26_NAIVE_SINK_PER_SPLIT)?(int64_t(S)+PN-1)/PN:1);
    }
    sh.sum[tid]=l;sh.phase.p.alpha[tid]=alpha;
  }
  __syncthreads(); // Publish one sink fold / final denominator per Q row.
#pragma unroll
  for(int c=0;c<8;++c) {
#pragma unroll
    for(int e=0;e<4;++e) {
      int r=mrow+m26tc::c_row(lane,e),t=base+r/rep,h=kvh*rep+r%rep;
      int col=(warp&1)*64+c*8+m26tc::c_col(lane,e);
      if(t<T) {
        float o=(output[c][0][e]+output[c][1][e])*sh.phase.p.alpha[r];
        if(naive&M26_NAIVE_VSCALE_ON_READ)o*=0.707f;
        out[((int64_t)t*64+h)*128+col]=sh.row_bad[r]?NAN:sh.sum[r]>0?o/sh.sum[r]:0;
      }
    }
  }
}
