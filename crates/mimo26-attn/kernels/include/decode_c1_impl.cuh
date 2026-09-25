// Experimental C1 implementation; included inside attn_decode_tc.cu's namespace.
// Original code. C3 remains a separate entry. No performance/promotion claim.
using C1Shared=m26tc::WarpPipeModel<BF>;
static_assert(sizeof(C1Shared)==44416,"C1 typed storage matches CPU model");
__device__ __forceinline__ uint32_t c1_addr(const void* p) {
  return uint32_t(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void c1_init(uint64_t* p,int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(c1_addr(p)),"r"(n) : "memory");
}
__device__ __forceinline__ void c1_arrive(uint64_t* p) {
#if __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(c1_addr(p)) : "memory");
#else
  // Legacy spelling has release/CTA semantics by default.
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(c1_addr(p)) : "memory");
#endif
}
__device__ __forceinline__ void c1_wait(uint64_t* p,int phase) {
  uint32_t done;
  do {
#if __CUDA_ARCH__ >= 900
    asm volatile("{ .reg .pred ready; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%1], %2; selp.u32 %0, 1, 0, ready; }"
                 : "=r"(done) : "r"(c1_addr(p)),"r"(phase) : "memory");
#else
    asm volatile("{ .reg .pred ready; mbarrier.test_wait.parity.shared::cta.b64 ready, [%1], %2; selp.u32 %0, 1, 0, ready; }"
                 : "=r"(done) : "r"(c1_addr(p)),"r"(phase) : "memory");
#endif
  } while(!done);
}
__device__ __forceinline__ void c1_inval(uint64_t* p) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(c1_addr(p)) : "memory");
}
__device__ __forceinline__ void c1_producer_sync() { asm volatile("bar.sync 1, 64;" ::: "memory"); }
__device__ __forceinline__ void c1_qk_sync() { asm volatile("bar.sync 2, 64;" ::: "memory"); }
__device__ void c1_prefetch(C1Shared& s,int64_t first,int64_t hi,m26_geom g,int kvh,
                          const uint8_t* kc,const uint8_t* vc,const int32_t* pages,
                          int page_tokens,uint32_t naive) {
  int tid=threadIdx.x; // ONLY producer threads 0..63 enter this function.
  if(tid<16) {
    int j=int(first+tid);
    s.physical[tid]=j<hi?((pages&&!(naive&M26_NAIVE_TC_IGNORE_PAGES))?
        pages[j/page_tokens]*page_tokens+j%page_tokens:j):0;
  }
  c1_producer_sync();
  for(int vector=tid;vector<16*320/16;vector+=64) {
    bool key=vector<16*192/16;
    int local=key?vector:vector-16*192/16;
    int width=key?192:128,row=local/(width/16),d=(local%(width/16))*16;
    bool valid=first+row<hi;
    const uint8_t* base=key?kc:vc;
    const uint8_t* src=valid?base+((int64_t)s.physical[row]*g.n_kv+kvh)*width+d:base;
    int dst=key?m26tc::raw_k(row,d):16*192+row*128+d;
    copy16(s.raw+dst,src,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<bool QResidual>
__global__ void decode_c1(m26_geom g,const float* q,const uint8_t* kc,const uint8_t* vc,
                         const int32_t* pages,int page_tokens,const int64_t* qpos,
                         const int64_t* kpos,int T,int S,int splits,uint32_t naive,float* partials) {
  extern __shared__ __align__(16) unsigned char c1_memory[];
  auto& s=*reinterpret_cast<C1Shared*>(c1_memory);
  constexpr int CN=16;
  int tid=threadIdx.x,lane=tid&31,t=blockIdx.z,kvh=blockIdx.y,sp=blockIdx.x,rep=64/g.n_kv;
  int64_t lo=(int64_t)S*sp/splits,hi=(int64_t)S*(sp+1)/splits;
  int tiles=int((hi-lo+CN-1)/CN);
  int64_t window=g.window;
  if((naive&M26_NAIVE_GA_WINDOWED)&&window<=0)window=128;
  float scale=float(m26::attn_scale(192,128,naive));
  for(int i=tid;i<M*192;i+=256) {
    int row=i/192,col=i%192,dst=m26tc::tile_index(M,row,col);
    float x=row<rep?q[((int64_t)t*64+kvh*rep+row)*192+col]:0;
    BF h=__float2bfloat16_rn(x);s.qh[dst]=h;
    if constexpr(QResidual) {
      float residual=x-__bfloat162float(h);BF l=__float2bfloat16_rn(residual);
      s.ql[dst]=l;s.qt[dst]=__float2bfloat16_rn(residual-__bfloat162float(l));
    }
  }
  if(tid<M){s.maximum[tid]=-INFINITY;s.sum[tid]=0;}
  if(tid==0)s.bad=0;
  if(tid<2){c1_init(&s.k_ready[tid],64);c1_init(&s.p_ready[tid],64);c1_init(&s.free_slot[tid],128);}
  __syncthreads(); // Publish barrier initialization and immutable Q to all roles.
  if(tid<64) {
    for(int tile=0;tile<tiles;++tile) {
      int slot=tile&1;auto& a=s.slot[slot];int64_t first=lo+int64_t(tile)*CN;
      // Raw DMA may run ahead of free-slot availability, but cannot overwrite
      // raw until all expansion readers of the previous iteration have retired.
      c1_prefetch(s,first,hi,g,kvh,kc,vc,pages,page_tokens,naive);
      if(tile>=2)c1_wait(&s.free_slot[slot],((tile/2)-1)&1);
      wait_current(false);
      if(tid<CN)a.visible[tid]=first+tid<hi&&m26::is_visible(qpos[t],kpos[first+tid],window,naive);
      c1_producer_sync(); // All raw groups complete; visibility is published.
      for(int i=tid;i<CN*192/2;i+=64) {
        int row=m26tc::producer_row(i,192),d=m26tc::producer_col(i,192);
        uint16_t codes=a.visible[row]?*reinterpret_cast<const uint16_t*>(s.raw+m26tc::raw_k(row,d)):0;
        *reinterpret_cast<uint32_t*>(a.phase.k+m26tc::tile_index(CN,row,d))=m26tc::e4m3x2_bf16(codes);
      }
      for(int i=tid;i<CN*128/2;i+=64) {
        int row=(i/128)*2,d=i%128;
        uint16_t x=a.visible[row]?s.raw[CN*192+row*128+d]:0;
        uint16_t y=a.visible[row+1]?s.raw[CN*192+(row+1)*128+d]:0;
        *reinterpret_cast<uint32_t*>(a.v+m26tc::tile_index(128,d,row))=m26tc::e4m3x2_bf16(x|(y<<8));
      }
      c1_producer_sync(); // Retire EVERY raw reader before either warp reuses it.
      c1_arrive(&s.k_ready[slot]); // One release arrival per producer thread.
    }
    wait_current(false); // Also well-defined for the empty split; no pending DMA.
  } else if(tid<128) {
    int qtid=tid-64,qwarp=qtid/32;
    for(int tile=0;tile<tiles;++tile) {
      int slot=tile&1;auto& a=s.slot[slot];c1_wait(&s.k_ready[slot],(tile/2)&1);
      constexpr int Terms=QResidual?3:1;
      float score[Terms][2][4]={};
#pragma unroll
      for(int d=0;d<192;d+=16) {
        int group=m26tc::q_acc_group(d/16);uint32_t av[4],bv[2];
        load_b(a.phase.k,CN,qwarp*8,d,bv);load_a(s.qh,M,d,av);mma(score[0][group],av,bv);
        if constexpr(QResidual) {
          if(!(naive&M26_NAIVE_TC_DROP_Q_LOW)) {
            load_a(s.ql,M,d,av);mma(score[1][group],av,bv);
            if(!(naive&M26_NAIVE_TC_DROP_Q_TAIL)){load_a(s.qt,M,d,av);mma(score[2][group],av,bv);}
          }
        }
      }
      float final_score[4];bool bad=false;
#pragma unroll
      for(int e=0;e<4;++e) {
        float total=score[0][0][e]+score[0][1][e];
        if constexpr(QResidual){total+=score[1][0][e]+score[1][1][e];total+=score[2][0][e]+score[2][1][e];}
        int r=m26tc::c_row(lane,e),col=qwarp*8+m26tc::c_col(lane,e);
        float value=total*scale;bool live=r<rep&&a.visible[col];
        bad|=live&&!isfinite(value);
        final_score[e]=live&&isfinite(value)?value:-INFINITY;
      }
      if(bad)atomicOr(&s.bad,1); // Never return a role early: publish and drain.
      c1_qk_sync(); // K has retired in BOTH QK warps before any alias write.
#pragma unroll
      for(int e=0;e<4;++e)a.phase.p.score[m26tc::c_row(lane,e)*CN+qwarp*8+m26tc::c_col(lane,e)]=final_score[e];
      c1_qk_sync();
      // Four lanes per row cover all 16 rows with the two QK warps.
      int row=qtid/4,sub=qtid%4;float values[4],maximum=-INFINITY;
#pragma unroll
      for(int k=0;k<4;++k){values[k]=a.phase.p.score[row*CN+sub+k*4];maximum=fmaxf(maximum,values[k]);}
#pragma unroll
      for(int d=2;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,4));
      float old=s.maximum[row],next=fmaxf(old,maximum);
      float alpha=(maximum==-INFINITY||(naive&M26_NAIVE_NO_RUNNING_RESCALE))?1.f:(isfinite(old)?expf(old-next):0.f);
      float sum=0;
#pragma unroll
      for(int k=0;k<4;++k) {
        float p=isfinite(values[k])?expf(values[k]-next):0;BF high=__float2bfloat16_rn(p);
        int dst=m26tc::tile_index(M,row,sub+k*4);
        a.phase.p.ph[dst]=high;a.phase.p.pl[dst]=__float2bfloat16_rn(p-__bfloat162float(high));sum+=p;
      }
#pragma unroll
      for(int d=2;d;d/=2)sum+=__shfl_xor_sync(0xffffffff,sum,d,4);
      if(!sub){s.maximum[row]=next;s.sum[row]=s.sum[row]*alpha+sum;a.phase.p.alpha[row]=alpha;}
      // Running m/l may advance; this slot's alpha remains live until PV release.
      c1_arrive(&s.p_ready[slot]);
    }
  } else {
    int warp=(tid-128)/32;
    float output[4][2][4]={}; // N16 has one k16 group, separate P-high/P-low.
    for(int tile=0;tile<tiles;++tile) {
      int slot=tile&1;auto& a=s.slot[slot];c1_wait(&s.p_ready[slot],(tile/2)&1);
#pragma unroll
      for(int c=0;c<4;++c) {
#pragma unroll
        for(int e=0;e<4;++e) {
          float alpha=a.phase.p.alpha[m26tc::c_row(lane,e)];
          output[c][0][e]*=alpha;output[c][1][e]*=alpha;
        }
        uint32_t av[4],bv[2];load_b(a.v,128,warp*32+c*8,0,bv);
        load_a(a.phase.p.ph,M,0,av);mma(output[c][0],av,bv);
        if(!(naive&M26_NAIVE_TC_DROP_P_LOW)){load_a(a.phase.p.pl,M,0,av);mma(output[c][1],av,bv);}
      }
      c1_arrive(&s.free_slot[slot]); // Every PV reader retires, not just a leader.
    }
    int64_t count=(int64_t)T*64*splits;
#pragma unroll
    for(int c=0;c<4;++c) {
#pragma unroll
      for(int e=0;e<4;++e) {
        int row=m26tc::c_row(lane,e),col=warp*32+c*8+m26tc::c_col(lane,e);
        if(row<rep){auto idx=m26tc::partial_index(t,kvh*rep+row,sp,splits);partials[2*count+idx*128+col]=output[c][0][e]+output[c][1][e];}
      }
    }
  }
  __syncthreads(); // All roles and raw DMA drained; PV global stores also retired.
  if(s.bad) { if(tid<128)poison(partials,T,t,kvh,rep,sp,splits); }
  else if(tid<rep) {
    int64_t count=(int64_t)T*64*splits,idx=m26tc::partial_index(t,kvh*rep+tid,sp,splits);
    partials[idx]=s.maximum[tid];partials[count+idx]=s.sum[tid];
  }
  if(tid<2){c1_inval(&s.k_ready[tid]);c1_inval(&s.p_ready[tid]);c1_inval(&s.free_slot[tid]);}
}
