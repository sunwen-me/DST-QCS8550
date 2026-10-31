#include "qnn_runtime/session.hpp"
#include <QnnInterface.h>
#include <QnnOpDef.h>
#include <System/QnnSystemInterface.h>
#include <HTP/QnnHtpGraph.h>
#include <opencv2/core.hpp>
#include <dlfcn.h>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace qnn_runtime {
namespace {
void check(Qnn_ErrorHandle_t code, const std::string& action) {
    if(code != QNN_SUCCESS) {
        std::ostringstream message;message<<"QNN "<<action<<" failed: 0x"<<std::hex<<code;
        throw std::runtime_error(message.str());
    }
}
void log(const char* format,QnnLog_Level_t,std::uint64_t,va_list args) {
    std::vfprintf(stderr,format,args);std::fputc('\n',stderr);
}
void* library(const std::filesystem::path& path) {
    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!handle)throw std::runtime_error(std::string("QNN dlopen: ")+dlerror());
    return handle;
}
template<class F> F symbol(void* handle,const char* name) {
    auto function=reinterpret_cast<F>(dlsym(handle,name));
    if(!function)throw std::runtime_error(std::string("missing QNN symbol: ")+name);
    return function;
}
Qnn_DataType_t dtype(Type type) {
    switch(type){
    case Type::Float32:return QNN_DATATYPE_FLOAT_32;
    case Type::Float16:return QNN_DATATYPE_FLOAT_16;
    case Type::Int32:return QNN_DATATYPE_INT_32;
    case Type::UInt32:return QNN_DATATYPE_UINT_32;
    case Type::Bool8:return QNN_DATATYPE_BOOL_8;
    case Type::UInt8:return QNN_DATATYPE_UINT_8;
    }
    throw std::runtime_error("unknown tensor type");
}
Type dtype(Qnn_DataType_t type) {
    switch(type){
    case QNN_DATATYPE_FLOAT_32:return Type::Float32;
    case QNN_DATATYPE_FLOAT_16:return Type::Float16;
    case QNN_DATATYPE_INT_32:return Type::Int32;
    case QNN_DATATYPE_UINT_32:return Type::UInt32;
    case QNN_DATATYPE_BOOL_8:return Type::Bool8;
    case QNN_DATATYPE_UINT_8:return Type::UInt8;
    default:throw std::runtime_error("unsupported QNN I/O data type: "+std::to_string(type));
    }
}
template<class T> std::vector<std::uint8_t> encode_bytes(const T* data,std::size_t count) {
    std::vector<std::uint8_t> output(count*sizeof(T));
    if(!output.empty())std::memcpy(output.data(),data,output.size());
    return output;
}
template<class T> T at(const std::vector<std::uint8_t>& data,std::size_t i) {
    T value;std::memcpy(&value,data.data()+i*sizeof(T),sizeof(T));return value;
}
}
std::size_t element_size(Type type) {
    switch(type){case Type::Float32:case Type::Int32:case Type::UInt32:return 4;
    case Type::Float16:return 2;case Type::Bool8:case Type::UInt8:return 1;}
    throw std::runtime_error("unknown tensor type");
}
std::size_t Tensor::elements() const {
    std::size_t result=1;
    if(shape.empty() || shape.size()>8)throw std::runtime_error("invalid tensor rank");
    for(auto dimension:shape){
        if(dimension==0 || result>64U*1024U*1024U/dimension)
            throw std::runtime_error("invalid/oversized tensor shape");
        result*=dimension;
    }
    return result;
}
std::string shape_string(const Shape& shape) {
    std::string result="[";for(std::size_t i=0;i<shape.size();++i){if(i)result+=',';result+=std::to_string(shape[i]);}
    return result+"]";
}
Param Param::f32(std::string name,float value){return {std::move(name),Type::Float32,{},encode_bytes(&value,1)};}
Param Param::u32(std::string name,std::uint32_t value){return {std::move(name),Type::UInt32,{},encode_bytes(&value,1)};}
Param Param::boolean(std::string name,bool value){const std::uint8_t v=value?1:0;return {std::move(name),Type::Bool8,{},encode_bytes(&v,1)};}
Param Param::u32s(std::string name,std::vector<std::uint32_t> values,Shape shape){
    if(shape.empty())shape={static_cast<std::uint32_t>(values.size())};
    return {std::move(name),Type::UInt32,std::move(shape),encode_bytes(values.data(),values.size())};
}
Param Param::i32s(std::string name,std::vector<std::int32_t> values,Shape shape){
    if(shape.empty())shape={static_cast<std::uint32_t>(values.size())};
    return {std::move(name),Type::Int32,std::move(shape),encode_bytes(values.data(),values.size())};
}

struct Session::Impl {
    struct Entry { Tensor data;Role role;Qnn_Tensor_t value=QNN_TENSOR_INIT; };
    struct Node {std::string name,type;std::vector<Param> parameters;
        std::vector<std::string> tensor_names;
        std::vector<Qnn_Param_t> qparams;std::vector<Qnn_Tensor_t> inputs,outputs;};
    void* backend_library=nullptr;void* system_library=nullptr;
    const QNN_INTERFACE_VER_TYPE* api=nullptr;
    const QNN_SYSTEM_INTERFACE_VER_TYPE* system=nullptr;
    Qnn_LogHandle_t logger=nullptr;Qnn_BackendHandle_t backend=nullptr;
    Qnn_DeviceHandle_t device=nullptr;Qnn_ContextHandle_t context=nullptr;
    Qnn_GraphHandle_t graph=nullptr;QnnSystemContext_Handle_t metadata=nullptr;
    Qnn_ProfileHandle_t profile=nullptr;
    // Keep the backing file mapped through contextFree. Large CLIP contexts
    // must not retain a second, anonymous heap copy of all model weights.
    struct Binary {
        void* address=MAP_FAILED;std::size_t length=0;
        ~Binary(){if(address!=MAP_FAILED)munmap(address,length);}
        void* data()const{return address;}
        std::size_t size()const{return length;}
        void open(const std::string& path){
            const int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
            if(fd<0)throw std::runtime_error("cannot open QNN context: "+path);
            struct stat st{};
            if(fstat(fd,&st)!=0 || st.st_size<=0 || st.st_size>4LL*1024*1024*1024){
                close(fd);throw std::runtime_error("invalid QNN model size");
            }
            length=static_cast<std::size_t>(st.st_size);
            address=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
            if(address==MAP_FAILED)throw std::runtime_error("cannot map QNN context");
        }
    } binary;
    std::deque<Entry> entries;std::map<std::string,std::size_t> index;
    std::vector<std::string> input_names,output_names;
    std::deque<Node> nodes;
    bool ready=false;

    explicit Impl(const std::string& runtime,bool profiling) {
        try{
            const auto libs=std::filesystem::path(runtime)/"lib/aarch64-oe-linux-gcc11.2";
            backend_library=library(libs/"libQnnHtp.so");system_library=library(libs/"libQnnSystem.so");
            const QnnInterface_t** providers=nullptr;std::uint32_t count=0;
            check(symbol<decltype(&QnnInterface_getProviders)>(backend_library,"QnnInterface_getProviders")(&providers,&count),"providers");
            for(std::uint32_t i=0;i<count;++i){
                const auto v=providers[i]->apiVersion.coreApiVersion;
                if(v.major==QNN_API_VERSION_MAJOR && v.minor>=QNN_API_VERSION_MINOR){api=&providers[i]->QNN_INTERFACE_VER_NAME;break;}
            }
            if(!api)throw std::runtime_error("compatible QNN API not found");
            const char* build=nullptr;check(api->backendGetBuildId(&build),"build ID");
            if(!build || std::string(build).find("2.50.")==std::string::npos)
                throw std::runtime_error("QNN 2.50 required; refusing "+std::string(build?build:"unknown"));
            std::cout<<"QNN_RUNTIME HTP build="<<build<<std::endl;
            const QnnSystemInterface_t** systems=nullptr;count=0;
            check(symbol<decltype(&QnnSystemInterface_getProviders)>(system_library,"QnnSystemInterface_getProviders")(&systems,&count),"system providers");
            for(std::uint32_t i=0;i<count;++i){
                const auto v=systems[i]->systemApiVersion;
                if(v.major==QNN_SYSTEM_API_VERSION_MAJOR && v.minor>=QNN_SYSTEM_API_VERSION_MINOR){system=&systems[i]->QNN_SYSTEM_INTERFACE_VER_NAME;break;}
            }
            if(!system)throw std::runtime_error("compatible QNN system API not found");
            check(api->logCreate(log,QNN_LOG_LEVEL_WARN,&logger),"logger");
            check(api->deviceCreate(logger,nullptr,&device),"HTP device");
            check(api->backendCreate(logger,nullptr,&backend),"HTP backend");
            if(profiling)check(api->profileCreate(backend,QNN_PROFILE_LEVEL_DETAILED,&profile),"profiling");
        }catch(...){release();throw;}
    }
    ~Impl(){release();}
    void release(){
        if(context)api->contextFree(context,nullptr);
        if(profile)api->profileFree(profile);
        if(backend)api->backendFree(backend);
        if(device)api->deviceFree(device);
        if(metadata)system->systemContextFree(metadata);
        if(logger)api->logFree(logger);
        if(backend_library)dlclose(backend_library);
        if(system_library)dlclose(system_library);
        context=nullptr;profile=nullptr;backend=nullptr;device=nullptr;metadata=nullptr;logger=nullptr;
        backend_library=nullptr;system_library=nullptr;
    }
    Entry& entry(const std::string& name){return entries.at(index.at(name));}
    static void bind(Entry& e){
        auto apply=[&](auto& t){t.name=e.data.name.c_str();t.rank=static_cast<std::uint32_t>(e.data.shape.size());
            t.dimensions=e.data.shape.data();t.memType=QNN_TENSORMEMTYPE_RAW;
            t.clientBuf.data=e.data.bytes.empty()?nullptr:e.data.bytes.data();
            t.clientBuf.dataSize=static_cast<std::uint32_t>(e.data.bytes.size());};
        if(e.value.version==QNN_TENSOR_VERSION_1)apply(e.value.v1);else apply(e.value.v2);
    }
    Entry& add(std::string name,Type type,Shape shape,Role role,std::span<const std::uint8_t> data={}){
        if(name.empty() || index.count(name))throw std::invalid_argument("duplicate/empty tensor name: "+name);
        const auto pos=entries.size();entries.emplace_back();auto& e=entries.back();
        e.data.name=std::move(name);e.data.type=type;e.data.shape=std::move(shape);e.role=role;
        const auto size=e.data.elements()*element_size(type);
        if(role!=Role::Native)e.data.bytes.resize(size);
        if(!data.empty()){
            if(data.size()!=size || role!=Role::Static)throw std::invalid_argument("invalid constant tensor data");
            std::copy(data.begin(),data.end(),e.data.bytes.begin());
        }
        e.value.v1.type=role==Role::Input?QNN_TENSOR_TYPE_APP_WRITE:role==Role::Output?
            QNN_TENSOR_TYPE_APP_READ:role==Role::Static?QNN_TENSOR_TYPE_STATIC:QNN_TENSOR_TYPE_NATIVE;
        e.value.v1.dataType=dtype(type);bind(e);
        index.emplace(e.data.name,pos);
        if(role==Role::Input)input_names.push_back(e.data.name);
        if(role==Role::Output)output_names.push_back(e.data.name);
        return e;
    }
    void event(std::ostream& out,QnnProfile_EventId_t id,int depth)const{
        if(depth>16)return;
        QnnProfile_EventData_t data=QNN_PROFILE_EVENT_DATA_INIT;
        check(api->profileGetEventData(id,&data),"profile event");
        out<<"{\"depth\":"<<depth<<",\"type\":"<<data.type<<",\"unit\":"<<data.unit
            <<",\"value\":"<<data.value<<",\"identifier\":"<<std::quoted(data.identifier?data.identifier:"")<<"}\n";
        const QnnProfile_EventId_t* children=nullptr;std::uint32_t count=0;
        check(api->profileGetSubEvents(id,&children,&count),"profile children");
        for(std::uint32_t i=0;i<count;++i)event(out,children[i],depth+1);
    }
};
Session::Session(const std::string& runtime,bool profile):impl_(std::make_unique<Impl>(runtime,profile)){}
Session::~Session()=default;
Tensor& Session::tensor(const std::string& name){return impl_->entry(name).data;}
const Tensor& Session::tensor(const std::string& name)const{return impl_->entry(name).data;}
std::vector<std::string> Session::inputs()const{return impl_->input_names;}
std::vector<std::string> Session::outputs()const{return impl_->output_names;}

void Session::load(const std::string& model){
    auto& s=*impl_;if(s.context)throw std::runtime_error("session already has a graph");
    if(s.binary.data()!=MAP_FAILED)throw std::runtime_error("session already loaded a binary");
    s.binary.open(model);
    check(s.system->systemContextCreate(&s.metadata),"metadata context");
    const QnnSystemContext_BinaryInfo_t* info=nullptr;
    check(s.system->systemContextGetMetaData(s.metadata,s.binary.data(),s.binary.size(),&info),"model metadata");
    QnnSystemContext_GraphInfo_t* graphs=nullptr;std::uint32_t count=0;
    switch(info->version){
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:graphs=info->contextBinaryInfoV1.graphs;count=info->contextBinaryInfoV1.numGraphs;break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:graphs=info->contextBinaryInfoV2.graphs;count=info->contextBinaryInfoV2.numGraphs;break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:graphs=info->contextBinaryInfoV3.graphs;count=info->contextBinaryInfoV3.numGraphs;break;
    default:throw std::runtime_error("unsupported binary metadata version");}
    if(count!=1 || !graphs)throw std::runtime_error("expected one QNN graph");
    std::string name;
    const auto load_io=[&](const Qnn_Tensor_t& value,Role role){
        if(value.version!=QNN_TENSOR_VERSION_1 && value.version!=QNN_TENSOR_VERSION_2)
            throw std::runtime_error("unsupported tensor version");
        const auto load=[&](const auto& t){
            Shape shape(t.dimensions,t.dimensions+t.rank);
            auto& e=s.add(t.name,dtype(t.dataType),std::move(shape),role);
            e.value=value;Impl::bind(e);
            std::cout<<"QNN tensor "<<e.data.name<<' '<<shape_string(e.data.shape)<<" type="<<static_cast<int>(e.data.type)<<std::endl;
        };
        if(value.version==QNN_TENSOR_VERSION_1)load(value.v1);else load(value.v2);
    };
    const auto load_graph=[&](const auto& g){
        if(!g.graphName || g.numGraphInputs>16 || g.numGraphOutputs>16)throw std::runtime_error("invalid graph metadata");
        name=g.graphName;
        for(std::uint32_t i=0;i<g.numGraphInputs;++i)load_io(g.graphInputs[i],Role::Input);
        for(std::uint32_t i=0;i<g.numGraphOutputs;++i)load_io(g.graphOutputs[i],Role::Output);
    };
    switch(graphs[0].version){
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:load_graph(graphs[0].graphInfoV1);break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:load_graph(graphs[0].graphInfoV2);break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:load_graph(graphs[0].graphInfoV3);break;
    default:throw std::runtime_error("unsupported graph metadata version");}
    check(s.api->contextCreateFromBinary(s.backend,s.device,nullptr,s.binary.data(),s.binary.size(),&s.context,nullptr),"load context");
    check(s.api->graphRetrieve(s.context,name.c_str(),&s.graph),"retrieve graph");s.ready=true;
}
void Session::create(const std::string& name){
    auto& s=*impl_;if(s.context)throw std::runtime_error("session already has a graph");
    check(s.api->contextCreate(s.backend,s.device,nullptr,&s.context),"create context");
    QnnHtpGraph_CustomConfig_t precision{};precision.option=QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    precision.precision=QNN_PRECISION_FLOAT16;
    QnnGraph_Config_t config=QNN_GRAPH_CONFIG_INIT;config.option=QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    config.customConfig=&precision;const QnnGraph_Config_t* configs[]={&config,nullptr};
    check(s.api->graphCreate(s.context,name.c_str(),configs,&s.graph),"create graph");
}
void Session::add_tensor(std::string name,Type type,Shape shape,Role role,std::span<const std::uint8_t> data){
    auto& s=*impl_;if(!s.graph || s.ready)throw std::runtime_error("graph is not editable");
    auto& e=s.add(std::move(name),type,std::move(shape),role,data);
    if(role!=Role::Static)e.value.v1.clientBuf={nullptr,0};
    check(s.api->tensorCreateGraphTensor(s.graph,&e.value),"register tensor "+e.data.name);
}
void Session::op(std::string name,std::string type,std::vector<std::string> input_names,
                 std::vector<std::string> output_names,std::vector<Param> params){
    auto& s=*impl_;if(!s.graph || s.ready)throw std::runtime_error("graph is not editable");
    s.nodes.emplace_back();auto& node=s.nodes.back();node.name=std::move(name);node.type=std::move(type);
    node.parameters=std::move(params);node.qparams.resize(node.parameters.size(),Qnn_Param_t QNN_PARAM_INIT);
    node.tensor_names.resize(node.parameters.size());
    for(std::size_t i=0;i<node.parameters.size();++i){
        auto& p=node.parameters[i];auto& q=node.qparams[i];q.name=p.name.c_str();
        if(p.shape.empty()){
            q.paramType=QNN_PARAMTYPE_SCALAR;q.scalarParam.dataType=dtype(p.type);
            if(p.bytes.size()!=element_size(p.type))throw std::runtime_error("invalid scalar parameter");
            if(p.type==Type::Float32)q.scalarParam.floatValue=at<float>(p.bytes,0);
            else if(p.type==Type::UInt32)q.scalarParam.uint32Value=at<std::uint32_t>(p.bytes,0);
            else if(p.type==Type::Bool8)q.scalarParam.bool8Value=p.bytes[0];
            else throw std::runtime_error("unsupported scalar parameter type");
        }else{
            q.paramType=QNN_PARAMTYPE_TENSOR;q.tensorParam=Qnn_Tensor_t QNN_TENSOR_INIT;
            node.tensor_names[i]=node.name+"__"+p.name;
            auto& t=q.tensorParam.v1;t.name=node.tensor_names[i].c_str();t.type=QNN_TENSOR_TYPE_STATIC;
            t.dataType=dtype(p.type);t.rank=static_cast<std::uint32_t>(p.shape.size());t.dimensions=p.shape.data();
            t.memType=QNN_TENSORMEMTYPE_RAW;t.clientBuf.data=p.bytes.data();t.clientBuf.dataSize=p.bytes.size();
            Tensor check_data{p.name,p.type,p.shape,{}};
            if(p.bytes.size()!=check_data.elements()*element_size(p.type))throw std::runtime_error("invalid tensor parameter length");
            check(s.api->tensorCreateGraphTensor(s.graph,&q.tensorParam),"register parameter "+node.tensor_names[i]);
        }
    }
    for(const auto& n:input_names)node.inputs.push_back(s.entry(n).value);
    for(const auto& n:output_names)node.outputs.push_back(s.entry(n).value);
    Qnn_OpConfig_t config=QNN_OPCONFIG_INIT;
    config.v1.name=node.name.c_str();config.v1.packageName=QNN_OP_PACKAGE_NAME_QTI_AISW;config.v1.typeName=node.type.c_str();
    config.v1.numOfParams=node.qparams.size();config.v1.params=node.qparams.data();
    config.v1.numOfInputs=node.inputs.size();config.v1.inputTensors=node.inputs.data();
    config.v1.numOfOutputs=node.outputs.size();config.v1.outputTensors=node.outputs.data();
    check(s.api->graphAddNode(s.graph,config),"add node "+node.name+" ("+node.type+")");
}
void Session::finalize(){auto& s=*impl_;check(s.api->graphFinalize(s.graph,nullptr,nullptr),"finalize graph");s.ready=true;}
void Session::run(){
    auto& s=*impl_;if(!s.ready)throw std::runtime_error("graph is not finalized");
    std::vector<Qnn_Tensor_t> inputs,outputs;
    for(const auto& name:s.input_names){auto& e=s.entry(name);Impl::bind(e);inputs.push_back(e.value);}
    for(const auto& name:s.output_names){auto& e=s.entry(name);Impl::bind(e);outputs.push_back(e.value);}
    check(s.api->graphExecute(s.graph,inputs.data(),inputs.size(),outputs.data(),outputs.size(),s.profile,nullptr),"HTP execute");
}
void Session::save(const std::string& path){
    auto& s=*impl_;Qnn_ContextBinarySize_t size=0,written=0;
    check(s.api->contextGetBinarySize(s.context,&size),"binary size");
    std::vector<std::uint8_t> data(size);
    check(s.api->contextGetBinary(s.context,data.data(),data.size(),&written),"save context");
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),written);
    if(!out)throw std::runtime_error("cannot save context binary");
}
void Session::write_profile(const std::string& path)const{
    const auto& s=*impl_;if(!s.profile)throw std::runtime_error("profiling was not enabled");
    std::ofstream out(path);const QnnProfile_EventId_t* events=nullptr;std::uint32_t count=0;
    check(s.api->profileGetEvents(s.profile,&events,&count),"profile events");
    for(std::uint32_t i=0;i<count;++i)s.event(out,events[i],0);
    if(!out)throw std::runtime_error("cannot write profile");
}
void Session::write_f32(const std::string& name,std::span<const float> values){
    auto& t=tensor(name);if(t.elements()!=values.size())throw std::runtime_error("input length mismatch: "+name);
    if(t.type==Type::Float32)std::memcpy(t.bytes.data(),values.data(),values.size_bytes());
    else if(t.type==Type::Float16){for(std::size_t i=0;i<values.size();++i){const cv::float16_t h(values[i]);std::memcpy(t.bytes.data()+2*i,&h,2);}}
    else throw std::runtime_error("input is not floating point: "+name);
}
std::vector<float> Session::read_f32(const std::string& name)const{
    const auto& t=tensor(name);std::vector<float> values(t.elements());
    if(t.type==Type::Float32)std::memcpy(values.data(),t.bytes.data(),t.bytes.size());
    else for(std::size_t i=0;i<values.size();++i){
        if(t.type==Type::Float16){cv::float16_t h;std::memcpy(&h,t.bytes.data()+2*i,2);values[i]=static_cast<float>(h);}
        else if(t.type==Type::Int32)values[i]=static_cast<float>(at<std::int32_t>(t.bytes,i));
        else if(t.type==Type::UInt32)values[i]=static_cast<float>(at<std::uint32_t>(t.bytes,i));
        else values[i]=t.bytes[i];
    }
    return values;
}
} // namespace qnn_runtime
