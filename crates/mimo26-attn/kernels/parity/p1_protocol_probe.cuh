// Host-driven analytic P1 probes, independent of external-corpus row counts.
#include "../../tests/gpu/p1_probe_check.h"
void p1_protocol_probe() {
  auto launch=g_bf16q?m26_attn_prefill_fp8_tc_bf16q:m26_attn_prefill_fp8_tc;
  size_t outputs=0,guards=0;int cases=0;
  auto run=[&](int T,int nkv,int mode) {
    int S=mode==1?0:mode==2?33:mode?2:273;
    m26_geom g{64,nkv,192,128,nkv==8?128:0,1.0};
    int rep=64/nkv,np=std::max(1,(S+255)/256),physical=np*256;
    std::vector<float> q(size_t(T)*64*192,.25f),sink(64);
    std::vector<uint8_t> k(size_t(physical)*nkv*192,0),v(size_t(physical)*nkv*128,0x30);
    std::vector<int64_t> qp(T),kp(std::max(S,1));std::vector<int32_t> pages(np);
    for(int p=0;p<np;++p)pages[p]=np-1-p;
    for(int h=0;h<64;++h)sink[h]=float(std::log(double(1+h%7)));
    for(int t=0;t<T;++t)qp[t]=mode?1000+t:999+(t*37+13)%301;
    for(int j=0;j<S;++j)kp[j]=1000+j;
    if(mode==1)std::fill(q.begin(),q.end(),NAN);
    if(mode==2){std::fill(qp.begin(),qp.end(),999);std::fill(k.begin(),k.end(),0x7f);std::fill(v.begin(),v.end(),0xff);}
    if(mode==4){kp[0]=1000;kp[1]=1127;qp[0]=1128;qp[1]=1127;}
    if(mode==3||mode==4||mode==5) {
      int j=mode==4?0:1,p=pages[j/256]*256+j%256;
      for(int h=0;h<nkv;++h) {
        if(mode==5)k[(size_t(p)*nkv+h)*192]=0x7f;
        else v[(size_t(p)*nkv+h)*128]=mode==4?0xff:0x7f;
      }
    }
    if(mode==6)q[(64+7)*192]=NAN; // only query 1, head 7
    size_t count=size_t(T)*64*128,padded=size_t((T+nkv-1)/nkv*nkv)*64*128;
    std::vector<float> expected(padded+128,12345.f),got(expected.size(),12345.f);
    for(int t=0;t<T;++t)for(int h=0;h<64;++h) {
      int n=0;bool bad=false,qbad=false;
      for(int d=0;d<192;++d)qbad|=!std::isfinite(q[(size_t(t)*64+h)*192+d]);
      for(int j=0;j<S;++j)if(kp[j]<=qp[t]&&(g.window<=0||kp[j]>=qp[t]-g.window+1)) {
        ++n;int p=pages[j/256]*256+j%256,kh=h/rep;bad|=qbad;
        for(int d=0;d<192;++d)bad|=(k[(size_t(p)*nkv+kh)*192+d]&127)==127;
        for(int d=0;d<128;++d)bad|=(v[(size_t(p)*nkv+kh)*128+d]&127)==127;
      }
      double den=n+(g.window>0?std::exp(double(sink[h])):0);
      float value=bad?NAN:den>0?float(.5*n/den):0;
      for(int d=0;d<128;++d)expected[64+(size_t(t)*64+h)*128+d]=value;
    }
    auto* dq=(float*)dev_upload(q.data(),q.size()*4);
    auto* dk=(uint8_t*)dev_upload(k.data(),k.size());auto* dv=(uint8_t*)dev_upload(v.data(),v.size());
    auto* dqp=(int64_t*)dev_upload(qp.data(),qp.size()*8);auto* dkp=(int64_t*)dev_upload(kp.data(),kp.size()*8);
    auto* dp=(int32_t*)dev_upload(pages.data(),pages.size()*4);auto* ds=(float*)dev_upload(sink.data(),sink.size()*4);
    auto* dout=(float*)dev_upload(got.data(),got.size()*4);
    M26_CALL(launch,&g,dq,dk,dv,dp,256,dqp,dkp,T,S,0,ds,dout+64,0);
    checked_copy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost);
    if(!p1_probe_matches(got,expected,64,count)) {
      fprintf(stderr,"RESULT: FAIL P1_PROTOCOL T=%d nkv=%d mode=%d\n",T,nkv,mode);std::exit(2);
    }
    outputs+=count;guards+=got.size()-count;++cases;
    checked_free(dout);checked_free(ds);checked_free(dp);checked_free(dkp);checked_free(dqp);checked_free(dv);checked_free(dk);checked_free(dq);
  };
  for(int nkv:{4,8})for(int T:{1,3,4,5,8,9,17})run(T,nkv,0);
  for(int mode=1;mode<=6;++mode)run(2,(mode==1||mode==4)?8:4,mode);
  printf("P1_PROTOCOL PASS cases=%d outputs=%zu guards=%zu query=%s scans=full\n",cases,outputs,guards,g_bf16q?"bf16q":"f32q");
}
