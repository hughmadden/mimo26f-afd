// Included inside gemm_parity.cu's private namespace; shares its CUDA RAII/error
// helpers. References come only from the external full-source NumPy oracle.
struct NumericalFailure:std::runtime_error {using std::runtime_error::runtime_error;};
struct PaddingRead:std::runtime_error {using std::runtime_error::runtime_error;};
struct Guarded {
    Device storage;size_t bytes;
    explicit Guarded(size_t n):storage(n+256),bytes(n) {
        ck(cudaMemset(storage.ptr,0xa5,storage.bytes),"redzones");
        if(bytes)ck(cudaMemset(data(),0xff,bytes),"NaN poison");
    }
    void* data() {return static_cast<uint8_t*>(storage.ptr)+128;} // preserve aligned128B row loads
    void upload(const void* src) {if(bytes)ck(cudaMemcpy(data(),src,bytes,cudaMemcpyHostToDevice),"guarded upload");}
    void check(const char* label) {
        std::array<uint32_t,64> red{};
        ck(cudaMemcpy(red.data(),storage.ptr,128,cudaMemcpyDeviceToHost),"prefix guard");
        ck(cudaMemcpy(red.data()+32,static_cast<uint8_t*>(data())+bytes,128,cudaMemcpyDeviceToHost),"suffix guard");
        for(auto v:red)need(v==0xa5a5a5a5u,std::string("redzone corrupted: ")+label);
    }
};
std::vector<float> columns(const std::vector<float>& values,int rows,int stride,int first,int width) {
    need(rows>=0 && size_t(rows)*stride<=values.size() && first>=0 && width>=0 && first+width<=stride,"oracle view OOB");
    std::vector<float> result(size_t(rows)*width);
    for(int t=0;t<rows;++t)std::copy_n(values.data()+size_t(t)*stride+first,width,result.data()+size_t(t)*width);
    return result;
}
double compare_values(const std::vector<float>& got,const std::vector<float>& want,const std::string& label) {
    need(got.size()==want.size(),"comparison shape mismatch "+label);double maximum=0;
    for(size_t i=0;i<got.size();++i) {
        const double error=std::abs(double(got[i])-want[i]);
        if(!std::isfinite(got[i]) || !std::isfinite(want[i]) || error>1e-5+1e-5*std::abs(double(want[i]))) {
            std::ostringstream msg;msg<<label<<" element="<<i<<" got="<<got[i]<<" want="<<want[i]<<" abs="<<error;
            throw NumericalFailure(msg.str());
        }
        maximum=std::max(maximum,error);
    }
    return maximum;
}
void tp4_comparator_selftest() {
    need(columns({0,1,2,3,4,5,6,7},2,4,1,2)==std::vector<float>({1,2,5,6}),"oracle column view");
    compare_values({1.000001f},{1.0f},"inside tolerance");
    for(float bad:{1.0f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        bool rejected=false;try{compare_values({bad},{0.0f},"deliberate wrong answer");}
        catch(const NumericalFailure&){rejected=true;}need(rejected,"numerical detector powerless");
    }
    bool bounds=false;try{columns({1,2},2,2,0,2);}catch(const std::runtime_error&){bounds=true;}
    need(bounds,"oracle view accepted short input");
    std::puts("HOST PASS TP4 comparator: column mapping, tolerance, finite/nonfinite wrong-answer detection");
}
// projection=-1 means actual three-GEMM+SiLU FFN. Metadata arrays contain only
// real groups; padded groups have no metadata entries. Pool has four DISTINCT
// rank images, addressed by explicit IDs, never by group ordinal.
std::vector<float> run_projection(const proof::Tp4Case& c,const std::vector<int32_t>& ids,
    const std::vector<int32_t>& offsets,const std::vector<float>& input,int projection,uint32_t naive,
    uint32_t inject_fault=0) {
    need(offsets.size()==ids.size()+1 && !ids.empty(),"host group shape");
    const int tokens=offsets.back(),cols=projection==2?512:4096,rows=(projection==0||projection==1)?512:4096;
    need(tokens>=0 && input.size()==size_t(tokens)*cols,"host input shape");
    const size_t weight_bytes=size_t(4)*M26X_QUARTER_SLICE_BYTES;
    Guarded weights(weight_bytes),x(input.size()*4),out(size_t(tokens)*rows*4),scratch(size_t(tokens)*512*2*4);
    Device dids(ids.size()*4),doffsets(offsets.size()*4),fault(4);
    for(int rank=0;rank<4;++rank)ck(cudaMemcpy(static_cast<uint8_t*>(weights.data())+size_t(rank)*M26X_QUARTER_SLICE_BYTES,
        c.image[rank].data(),M26X_QUARTER_SLICE_BYTES,cudaMemcpyHostToDevice),"Rust image upload");
    x.upload(input.data());ck(cudaMemset(fault.ptr,0,4),"clear fault");
    m26x_plan p{};p.layout_version=2;p.manifest_arch=M26X_BAKED_ARCH;p.manifest_sms=M26X_BAKED_SMS;
    p.capacity_class=M26X_CAPACITY_CLASS;p.resident_experts=4;p.n_groups=int(ids.size());p.padded_groups=3;
    p.total_tokens=tokens;for(size_t i=0;i<ids.size();++i)p.max_m=std::max(p.max_m,offsets[i+1]-offsets[i]);
    p.grouped_bytes=weight_bytes;p.x_bytes=x.bytes;p.out_bytes=out.bytes;p.scratch_bytes=scratch.bytes;
    p.expert_ids=static_cast<int32_t*>(dids.ptr);p.group_offsets=static_cast<int32_t*>(doffsets.ptr);p.fault=static_cast<uint32_t*>(fault.ptr);
    need(m26x_validate_host_plan(&p,ids.data(),offsets.data())==0,"host plan rejected before upload");
    auto device_ids=ids,device_offsets=offsets;
    if(inject_fault==2)device_ids[0]=4; // validated host metadata, deliberately corrupted DEVICE copy
    if(inject_fault==1)device_offsets[1]=tokens+1;
    dids.upload(device_ids.data());doffsets.upload(device_offsets.data());
    size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"TP4 allocation reserve");
    need(free>=size_t(4096)*1024*1024,"TP4 allocation breached4GiB reserve");
    if(projection==-1)ck(m26x_expert_ffn_v2(&p,static_cast<uint8_t*>(weights.data()),static_cast<float*>(x.data()),
        0,naive,static_cast<float*>(scratch.data()),out.data(),nullptr),"v2 FFN launch");
    else ck(m26x_grouped_gemm_v2(&p,static_cast<uint8_t*>(weights.data()),static_cast<float*>(x.data()),
        projection,0,naive,out.data(),nullptr),"v2 GEMM launch");
    ck(cudaDeviceSynchronize(),"TP4 synchronize");
    uint32_t fault_bits=0;ck(cudaMemcpy(&fault_bits,fault.ptr,4,cudaMemcpyDeviceToHost),"fault readback");
    weights.check("weights");x.check("input");out.check("output");scratch.check("scratch");
    if(inject_fault) {need(fault_bits==inject_fault,"wrong device metadata fault bits");return {};}
    if(naive==M26X_NAIVE_PAD_ROW_READ && (fault_bits&8))throw PaddingRead("nonresident poison byte was actually read; fault="+std::to_string(fault_bits));
    need(fault_bits==0,"unexpected device metadata fault="+std::to_string(fault_bits));
    std::vector<float> result(size_t(tokens)*rows);
    if(!result.empty())ck(cudaMemcpy(result.data(),out.data(),out.bytes,cudaMemcpyDeviceToHost),"TP4 output");
    return result;
}
size_t slice_coordinate_proof(const proof::Tp4Case& c,const proof::Json& fixture,int layer,int expert) {
    size_t count=0;
    for(int p=0;p<3;++p) {
        const std::string name=proof::tensor_prefix(layer,expert)+proof::projection(p);const proof::Json* b=nullptr;
        for(const auto& block:fixture.at("blocks").list())if(block.at("name").str()==name)b=&block;
        need(b!=nullptr,"missing full-coordinate block");
        const int rows=p==2?4096:512,cols=p==2?512:4096;
        for(int rank=0;rank<4;++rank) {
            const auto& image=c.image[rank];const auto wo=proof::payload_offset(p),so=proof::scale_offset(p);
            const std::vector<uint8_t> w(image.begin()+wo,image.begin()+wo+size_t(rows)*cols/2);
            const std::vector<uint8_t> s(image.begin()+so,image.begin()+so+size_t(rows)*cols/32);
            const auto got=gpu_unpack(w,s,rows,cols,0);
            const auto& pos=b->at("positions").list();const auto& expected=b->at("expected_f32_bits").list();
            for(size_t i=0;i<pos.size();++i) {
                const int full_row=int(pos[i].list()[0].num()),full_col=int(pos[i].list()[1].num());
                const int owner=p==2?full_col/512:full_row/512;
                if(owner!=rank)continue; // every position belongs to EXACTLY one rank, checked by total
                const int local_row=p==2?full_row:full_row-rank*512,local_col=p==2?full_col-rank*512:full_col;
                if(proof::hex32(got[size_t(local_row)*cols+local_col])!=expected[i].str())
                    throw NumericalFailure("Rust-slice GPU coordinate mismatch "+name+" rank="+std::to_string(rank));
                ++count;
            }
        }
    }
    need(count==3*2048,"rank partition skipped/doubled a fixture position");return count;
}
void accumulator_probe(uint32_t naive) {
    // Supplemental adversarial accumulator test, clearly synthetic. W=1 and
    // x=f32(0.001); the independent exact-real sum is4096*x. Same bytes/input
    // are used for positive FP32 and negative BF16 accumulation.
    proof::Tp4Case c;
    for(auto& image:c.image) {
        image.resize(M26X_QUARTER_SLICE_BYTES,0);
        std::fill(image.begin(),image.begin()+1048576,uint8_t(0x22));
        std::fill(image.begin()+1048576,image.begin()+1114112,uint8_t(127));
    }
    const std::vector<float> input(4096,0.001f),want(512,float(double(0.001f)*4096));
    compare_values(run_projection(c,{0},{0,1},input,0,0),want,"synthetic FP32 accumulator positive");
    std::puts("ACCUMULATOR GPU PASS: FP32 positive, independent W=1 constant-input oracle");
    if(naive==M26X_NAIVE_BF16_ACCUM)
        compare_values(run_projection(c,{0},{0,1},input,0,naive),want,"synthetic BF16 accumulator negative");
}
void aot_refusals() {
    // Real-device rejection, not a synthetic identity object.
    need(m26x_check_aot(M26X_BAKED_ARCH,M26X_BAKED_SMS+1,M26X_CAPACITY_CLASS,0)==cudaErrorInvalidDevice,"wrong SM accepted");
    need(m26x_check_aot(M26X_BAKED_ARCH+1,M26X_BAKED_SMS,M26X_CAPACITY_CLASS,0)==cudaErrorInvalidDevice,"wrong arch accepted");
    need(m26x_check_aot(M26X_BAKED_ARCH,M26X_BAKED_SMS,M26X_CAPACITY_CLASS==256?2048:256,0)==cudaErrorInvalidValue,"wrong capacity accepted");
    std::printf("AOT GPU PASS class=%d live identity; wrong SM, arch and capacity refused\n",M26X_CAPACITY_CLASS);
}
int tp4_gpu(const std::string& root,const std::string& path,uint32_t naive) {
    proof::layout_marker(root);const auto fixture=proof::read_json(path);proof::validate_fixture(fixture);
    const auto x=proof::floats(root+"/x.f32",64*4096);ready();aot_refusals();
    if(!naive || naive==M26X_NAIVE_BF16_ACCUM)accumulator_probe(naive);
    if(naive==M26X_NAIVE_BF16_ACCUM)return 0; // outer cell FAILS if mutation was not detected
    int cases=0,gemms=0;size_t bits=0;double maximum=0;
    for(int layer:{1,24,46})for(int expert:{0,7,255}) {
        const auto c=proof::load_case(root,fixture,layer,expert);const std::string name=proof::tag(layer,expert);
        if(!naive)bits+=slice_coordinate_proof(c,fixture,layer,expert);
        for(int m:{1,2,4,8,16,64}) {
            const auto input=columns(x,m,4096,0,4096);std::vector<float> sum(size_t(m)*4096,0);
            for(int rank=0;rank<4;++rank) {
                const std::string label=name+" M="+std::to_string(m)+" rank="+std::to_string(rank);
                if(!naive)for(int p=0;p<3;++p) {
                    const auto in=p==2?columns(c.h,m,2048,rank*512,512):input;
                    const auto want=p==2?columns(c.down[rank],m,4096,0,4096):columns(p==0?c.gate:c.up,m,2048,rank*512,512);
                    const auto got=run_projection(c,{rank},{0,m},in,p,0);
                    compare_values(got,want,label+" "+proof::projection(p));++gemms;
                }
                const auto result=run_projection(c,{rank},{0,m},input,-1,naive);
                compare_values(result,columns(c.partial[rank],m,4096,0,4096),label+" partial");
                for(size_t i=0;i<sum.size();++i)sum[i]+=result[i];
            }
            const double error=compare_values(sum,columns(c.y,m,4096,0,4096),name+" TP4 sum M="+std::to_string(m));
            maximum=std::max(maximum,error);++cases;
            std::printf("TP4 GPU PASS %s M=%d outputs=%zu max_abs=%.9e\n",name.c_str(),m,sum.size(),error);
        }
        if(!naive) {
            // Ragged sparse IDs plus invalid ID in empty group, without padding metadata.
            const std::array<std::pair<int,int>,3> ragged={{{1,2},{2,3},{5,3}}};
            for(const auto counts:ragged) {
                const int a=counts.first,total=a+counts.second;
                const auto result=run_projection(c,{3,-99,1},{0,a,a,total},columns(x,total,4096,0,4096),-1,0);
                auto want=columns(c.partial[3],total,4096,0,4096);
                std::copy(c.partial[1].begin()+a*4096,c.partial[1].begin()+total*4096,want.begin()+a*4096);
                compare_values(result,want,name+" sparse/empty/ragged");
            }
            need(run_projection(c,{-99},{0,0},{},-1,0).empty(),"zero-token FFN not empty");
            for(uint32_t fault:{1u,2u})run_projection(c,{0},{0,1},columns(x,1,4096,0,4096),-1,0,fault);
        }
    }
    need(cases==54,"TP4 case count");
    if(!naive)need(gemms==648 && bits==55296,"layerwise or coordinate proof count");
    std::printf("TP4 GPU PASS:54/54 sums,216 partials,%d isolated GEMMs,%zu coordinate bits,%d sparse/empty/ragged cases,%d zero-token cases,%d device-metadata faults; class=%d max_abs=%.9e flag=%u\n",
        gemms,bits,naive?0:27,naive?0:9,naive?0:18,M26X_CAPACITY_CLASS,maximum,naive);return 0;
}
