#ifndef __SD_MODEL_DIFFUSION_LTXV_QNN_HPP__
#define __SD_MODEL_DIFFUSION_LTXV_QNN_HPP__

// LTX-Video 0.9 transformer on the Qualcomm Hexagon NPU.
//
// The transformer is precompiled offline (android/tools/ltxv_qnn) into four QNN context binaries
// ltxv_chunk{0..3}.bin with static shapes: [tokens, dim] hidden states, 128 T5 tokens. This runner
// replaces LTXVRunner inside the normal stable-diffusion.cpp pipeline, so T5, image conditioning,
// sampling and the VAE keep running on ggml; only the 28 transformer blocks run on the HTP.

#ifdef SD_LTXV_QNN

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpSystemContext.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#include "model/diffusion/model.hpp"
#include "model/vae/ltx_vae.hpp"

namespace LTXV {

    class QnnRuntime {
    public:
        struct Graph {
            std::string name;
            Qnn_ContextHandle_t context = nullptr;
            Qnn_GraphHandle_t graph     = nullptr;
            std::vector<Qnn_Tensor_t> inputs;
            std::vector<Qnn_Tensor_t> outputs;
            std::vector<std::vector<uint32_t>> dims;  // keeps tensor dimension arrays alive
            std::vector<std::vector<uint8_t>> buffers;
        };

        // One QNN backend/device per process, shared by the transformer and the VAE runners.
        static std::shared_ptr<QnnRuntime> acquire(const std::string& lib_dir) {
            static std::mutex mutex;
            static std::weak_ptr<QnnRuntime> shared;
            std::lock_guard<std::mutex> lock(mutex);
            if (auto existing = shared.lock()) {
                return existing;
            }
            auto runtime = std::make_shared<QnnRuntime>();
            if (!runtime->init(lib_dir)) {
                return nullptr;
            }
            shared = runtime;
            return runtime;
        }

        ~QnnRuntime() {
            for (auto& [name, g] : graphs_) {
                if (g.context != nullptr) {
                    qnn_.contextFree(g.context, nullptr);
                }
            }
            if (device_ != nullptr) {
                qnn_.deviceFree(device_);
            }
            if (backend_ != nullptr) {
                qnn_.backendFree(backend_);
            }
            if (sys_context_ != nullptr) {
                sys_.systemContextFree(sys_context_);
            }
            if (htp_lib_ != nullptr) {
                dlclose(htp_lib_);
            }
            if (sys_lib_ != nullptr) {
                dlclose(sys_lib_);
            }
        }

        bool init(const std::string& lib_dir) {
            if (!lib_dir.empty()) {
                // FastRPC looks up libQnnHtpV*Skel.so through ADSP_LIBRARY_PATH.
                std::string adsp = lib_dir + ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp";
                setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
            }
            std::string prefix = lib_dir.empty() ? "" : lib_dir + "/";
            htp_lib_           = dlopen((prefix + "libQnnHtp.so").c_str(), RTLD_NOW | RTLD_LOCAL);
            sys_lib_           = dlopen((prefix + "libQnnSystem.so").c_str(), RTLD_NOW | RTLD_LOCAL);
            if (htp_lib_ == nullptr || sys_lib_ == nullptr) {
                LOG_ERROR("ltxv qnn: failed to load QNN libraries from '%s': %s", lib_dir.c_str(), dlerror());
                return false;
            }
            using GetProviders    = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
            using GetSysProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
            auto get_providers     = reinterpret_cast<GetProviders>(dlsym(htp_lib_, "QnnInterface_getProviders"));
            auto get_sys_providers = reinterpret_cast<GetSysProviders>(dlsym(sys_lib_, "QnnSystemInterface_getProviders"));
            if (get_providers == nullptr || get_sys_providers == nullptr) {
                LOG_ERROR("ltxv qnn: QNN provider symbols not found");
                return false;
            }
            const QnnInterface_t** providers = nullptr;
            uint32_t count                   = 0;
            if (get_providers(&providers, &count) != QNN_SUCCESS || count == 0) {
                LOG_ERROR("ltxv qnn: no QNN interface providers");
                return false;
            }
            qnn_ = providers[0]->QNN_INTERFACE_VER_NAME;
            const QnnSystemInterface_t** sys_providers = nullptr;
            if (get_sys_providers(&sys_providers, &count) != QNN_SUCCESS || count == 0) {
                LOG_ERROR("ltxv qnn: no QNN system interface providers");
                return false;
            }
            sys_ = sys_providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

            if (qnn_.backendCreate(nullptr, nullptr, &backend_) != QNN_SUCCESS) {
                LOG_ERROR("ltxv qnn: backendCreate failed");
                return false;
            }
            if (qnn_.deviceCreate(nullptr, nullptr, &device_) != QNN_SUCCESS) {
                LOG_ERROR("ltxv qnn: deviceCreate failed");
                return false;
            }
            if (sys_.systemContextCreate(&sys_context_) != QNN_SUCCESS) {
                LOG_ERROR("ltxv qnn: systemContextCreate failed");
                return false;
            }
            set_burst();
            return true;
        }

        // Spill-fill (scratch) size a context binary needs, read from its HTP graph blob info.
        uint64_t spill_fill_size(const std::string& path) {
            uint64_t result = 0;
            int fd          = open(path.c_str(), O_RDONLY);
            if (fd < 0) {
                return 0;
            }
            struct stat st;
            fstat(fd, &st);
            void* data = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (data == MAP_FAILED) {
                return 0;
            }
            const QnnSystemContext_BinaryInfo_t* bi = nullptr;
            Qnn_ContextBinarySize_t bi_size         = 0;
            if (sys_.systemContextGetBinaryInfo(sys_context_, data, static_cast<uint64_t>(st.st_size), &bi, &bi_size) == QNN_SUCCESS &&
                bi != nullptr && bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
                const auto& info = bi->contextBinaryInfoV3;
                for (uint32_t i = 0; i < info.numGraphs; i++) {
                    if (info.graphs[i].version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 ||
                        info.graphs[i].graphInfoV3.graphBlobInfo == nullptr) {
                        continue;
                    }
                    const auto* blob = static_cast<const QnnHtpSystemContext_GraphBlobInfo_t*>(info.graphs[i].graphInfoV3.graphBlobInfo);
                    if (blob->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
                        result = std::max(result, blob->contextBinaryGraphBlobInfoV1.spillFillBufferSize);
                    }
                }
            }
            munmap(data, static_cast<size_t>(st.st_size));
            return result;
        }

        // Contexts of one group share a single spill-fill buffer of max_spill_fill bytes instead of
        // reserving one each, which keeps a chunked transformer inside the HTP's 32-bit VA space.
        bool load_graph(const std::string& path,
                        const std::string& graph_name,
                        bool quiet                  = false,
                        const std::string& group    = "",
                        uint64_t max_spill_fill     = 0) {
            int fd = open(path.c_str(), O_RDONLY);
            if (fd < 0) {
                LOG_ERROR("ltxv qnn: cannot open '%s'", path.c_str());
                return false;
            }
            struct stat st;
            fstat(fd, &st);
            size_t size = static_cast<size_t>(st.st_size);
            void* data  = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (data == MAP_FAILED) {
                LOG_ERROR("ltxv qnn: mmap failed for '%s'", path.c_str());
                return false;
            }

            Graph g;
            g.name                                  = graph_name;
            const QnnSystemContext_BinaryInfo_t* bi = nullptr;
            Qnn_ContextBinarySize_t bi_size         = 0;
            bool ok                                 = sys_.systemContextGetBinaryInfo(sys_context_, data, size, &bi, &bi_size) == QNN_SUCCESS;
            ok                                      = ok && copy_graph_info(bi, graph_name, g);
            QnnHtpContext_CustomConfig_t htp_cfg;
            memset(&htp_cfg, 0, sizeof(htp_cfg));
            QnnContext_Config_t ctx_cfg;
            memset(&ctx_cfg, 0, sizeof(ctx_cfg));
            const QnnContext_Config_t* cfgs[] = {&ctx_cfg, nullptr};
            bool grouped                      = !group.empty() && max_spill_fill > 0;
            if (grouped) {
                htp_cfg.option                                 = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
                htp_cfg.groupRegistration.firstGroupHandle   = groups_.count(group) ? groups_[group] : nullptr;
                htp_cfg.groupRegistration.maxSpillFillBuffer = max_spill_fill;
                ctx_cfg.option                                 = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
                ctx_cfg.customConfig                           = &htp_cfg;
            }
            Qnn_ErrorHandle_t create_err = QNN_SUCCESS;
            if (ok) {
                create_err = qnn_.contextCreateFromBinary(backend_, device_, grouped ? cfgs : nullptr, data, size, &g.context, nullptr);
                ok         = create_err == QNN_SUCCESS;
                if (!ok && !quiet) {
                    LOG_ERROR("ltxv qnn: contextCreateFromBinary(%s) error %d (group %s, spill-fill %" PRIu64 " MB)",
                              graph_name.c_str(), static_cast<int>(create_err), grouped ? group.c_str() : "none", max_spill_fill >> 20);
                }
            }
            if (ok && grouped && !groups_.count(group)) {
                groups_[group] = g.context;
            }
            ok                                      = ok && qnn_.graphRetrieve(g.context, graph_name.c_str(), &g.graph) == QNN_SUCCESS;
            munmap(data, size);
            if (!ok) {
                if (quiet) {
                    LOG_INFO("ltxv qnn: '%s' does not fit next to the loaded graphs", graph_name.c_str());
                } else {
                    LOG_ERROR("ltxv qnn: failed to load graph '%s' from '%s'", graph_name.c_str(), path.c_str());
                }
                return false;
            }
            for (auto& t : g.inputs) {
                g.buffers.emplace_back(tensor_bytes(t));
            }
            for (auto& t : g.outputs) {
                g.buffers.emplace_back(tensor_bytes(t));
            }
            graphs_[graph_name] = std::move(g);
            LOG_INFO("ltxv qnn: loaded %s (%zu MB)", graph_name.c_str(), size >> 20);
            return true;
        }

        Graph* graph(const std::string& name) {
            auto it = graphs_.find(name);
            return it == graphs_.end() ? nullptr : &it->second;
        }

        // Frees a context so its HTP memory can be reused (the HTP address space is 32-bit).
        void unload(const std::string& name) {
            auto it = graphs_.find(name);
            if (it == graphs_.end()) {
                return;
            }
            if (it->second.context != nullptr) {
                for (auto g = groups_.begin(); g != groups_.end(); ++g) {
                    if (g->second == it->second.context) {
                        groups_.erase(g);  // the next context loaded into this group starts a new one
                        break;
                    }
                }
                qnn_.contextFree(it->second.context, nullptr);
            }
            graphs_.erase(it);
            LOG_INFO("ltxv qnn: unloaded %s", name.c_str());
        }

        // Copies fp32 data into the named input, converting to the tensor's element type.
        static bool set_input(Graph& g, const std::string& name, const float* src, size_t count) {
            for (size_t i = 0; i < g.inputs.size(); i++) {
                if (name == tensor_name(g.inputs[i])) {
                    return write_tensor(g.inputs[i], g.buffers[i], src, count);
                }
            }
            LOG_ERROR("ltxv qnn: graph %s has no input '%s'", g.name.c_str(), name.c_str());
            return false;
        }

        // Copies the named output into dst as fp32.
        static bool get_output(Graph& g, const std::string& name, float* dst, size_t count) {
            for (size_t i = 0; i < g.outputs.size(); i++) {
                if (name == tensor_name(g.outputs[i])) {
                    return read_tensor(g.outputs[i], g.buffers[g.inputs.size() + i], dst, count);
                }
            }
            LOG_ERROR("ltxv qnn: graph %s has no output '%s'", g.name.c_str(), name.c_str());
            return false;
        }

        // Passes an output buffer straight to the next graph's input when the element types match and,
        // for fixed-point tensors, so do the scale and offset (graphs are quantized separately);
        // otherwise converts through fp32.
        static bool forward_tensor(Graph& from, const std::string& out_name, Graph& to, const std::string& in_name) {
            int oi = -1, ii = -1;
            for (size_t i = 0; i < from.outputs.size(); i++) {
                if (out_name == tensor_name(from.outputs[i])) oi = static_cast<int>(i);
            }
            for (size_t i = 0; i < to.inputs.size(); i++) {
                if (in_name == tensor_name(to.inputs[i])) ii = static_cast<int>(i);
            }
            if (oi < 0 || ii < 0) {
                LOG_ERROR("ltxv qnn: cannot forward %s.%s -> %s.%s", from.name.c_str(), out_name.c_str(), to.name.c_str(), in_name.c_str());
                return false;
            }
            auto& src = from.buffers[from.inputs.size() + oi];
            auto& dst = to.buffers[ii];
            float src_scale = 0.f, dst_scale = 0.f;
            int32_t src_offset = 0, dst_offset = 0;
            const bool src_q   = scale_offset(from.outputs[oi], &src_scale, &src_offset);
            const bool dst_q   = scale_offset(to.inputs[ii], &dst_scale, &dst_offset);
            const bool same_q  = src_q == dst_q && (!src_q || (src_scale == dst_scale && src_offset == dst_offset));
            if (tensor_type(from.outputs[oi]) == tensor_type(to.inputs[ii]) && src.size() == dst.size() && same_q) {
                memcpy(dst.data(), src.data(), src.size());
                return true;
            }
            std::vector<float> tmp(element_count(from.outputs[oi]));
            return read_tensor(from.outputs[oi], src, tmp.data(), tmp.size()) &&
                   write_tensor(to.inputs[ii], dst, tmp.data(), tmp.size());
        }

        bool execute(Graph& g) {
            for (size_t i = 0; i < g.inputs.size(); i++) {
                bind(g.inputs[i], g.buffers[i]);
            }
            for (size_t i = 0; i < g.outputs.size(); i++) {
                bind(g.outputs[i], g.buffers[g.inputs.size() + i]);
            }
            Qnn_ErrorHandle_t err = qnn_.graphExecute(g.graph,
                                                      g.inputs.data(),
                                                      static_cast<uint32_t>(g.inputs.size()),
                                                      g.outputs.data(),
                                                      static_cast<uint32_t>(g.outputs.size()),
                                                      nullptr,
                                                      nullptr);
            if (err != QNN_SUCCESS) {
                LOG_ERROR("ltxv qnn: graphExecute(%s) failed: %d", g.name.c_str(), static_cast<int>(err));
                return false;
            }
            return true;
        }

    private:
        void* htp_lib_                   = nullptr;
        void* sys_lib_                   = nullptr;
        QNN_INTERFACE_VER_TYPE qnn_      = QNN_INTERFACE_VER_TYPE_INIT;
        QNN_SYSTEM_INTERFACE_VER_TYPE sys_ = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
        Qnn_BackendHandle_t backend_     = nullptr;
        Qnn_DeviceHandle_t device_       = nullptr;
        QnnSystemContext_Handle_t sys_context_ = nullptr;
        std::map<std::string, Graph> graphs_;
        std::map<std::string, Qnn_ContextHandle_t> groups_;

        void set_burst() {
            QnnDevice_Infrastructure_t infra = nullptr;
            if (qnn_.deviceGetInfrastructure == nullptr || qnn_.deviceGetInfrastructure(&infra) != QNN_SUCCESS || infra == nullptr) {
                LOG_WARN("ltxv qnn: no HTP perf infrastructure; running with default clocks");
                return;
            }
            auto* htp = static_cast<QnnHtpDevice_Infrastructure_t*>(infra);
            if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
                return;
            }
            uint32_t power_id = 0;
            if (htp->perfInfra.createPowerConfigId(0, 0, &power_id) != QNN_SUCCESS) {
                return;
            }
            QnnHtpPerfInfrastructure_PowerConfig_t cfg;
            memset(&cfg, 0, sizeof(cfg));
            cfg.option                               = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
            auto& d                                  = cfg.dcvsV3Config;
            d.contextId                              = power_id;
            d.setDcvsEnable                          = 1;
            d.dcvsEnable                             = 0;
            d.powerMode                              = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
            d.setSleepLatency                        = 1;
            d.sleepLatency                           = 40;
            d.setSleepDisable                        = 1;
            d.sleepDisable                           = 1;
            d.setBusParams                           = 1;
            d.busVoltageCornerMin                    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
            d.busVoltageCornerTarget                 = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
            d.busVoltageCornerMax                    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
            d.setCoreParams                          = 1;
            d.coreVoltageCornerMin                   = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
            d.coreVoltageCornerTarget                = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
            d.coreVoltageCornerMax                   = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
            const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&cfg, nullptr};
            if (htp->perfInfra.setPowerConfig(power_id, cfgs) != QNN_SUCCESS) {
                LOG_WARN("ltxv qnn: failed to set HTP burst mode");
            }
        }

        static const char* tensor_name(const Qnn_Tensor_t& t) {
            return t.version == QNN_TENSOR_VERSION_2 ? t.v2.name : t.v1.name;
        }
        static Qnn_DataType_t tensor_type(const Qnn_Tensor_t& t) {
            return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType;
        }
        static size_t element_count(const Qnn_Tensor_t& t) {
            uint32_t rank   = t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
            uint32_t* dims  = t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
            size_t n        = 1;
            for (uint32_t i = 0; i < rank; i++) {
                n *= dims[i];
            }
            return n;
        }
        static size_t element_size(Qnn_DataType_t type) {
            switch (type) {
                case QNN_DATATYPE_FLOAT_16:
                case QNN_DATATYPE_UFIXED_POINT_16:
                case QNN_DATATYPE_SFIXED_POINT_16:
                    return 2;
                case QNN_DATATYPE_UFIXED_POINT_8:
                case QNN_DATATYPE_SFIXED_POINT_8:
                    return 1;
                default:
                    return 4;
            }
        }
        static std::vector<uint8_t> tensor_bytes(const Qnn_Tensor_t& t) {
            return std::vector<uint8_t>(element_count(t) * element_size(tensor_type(t)));
        }
        static void bind(Qnn_Tensor_t& t, std::vector<uint8_t>& buf) {
            Qnn_ClientBuffer_t cb = {buf.data(), static_cast<uint32_t>(buf.size())};
            if (t.version == QNN_TENSOR_VERSION_2) {
                t.v2.memType   = QNN_TENSORMEMTYPE_RAW;
                t.v2.clientBuf = cb;
            } else {
                t.v1.memType   = QNN_TENSORMEMTYPE_RAW;
                t.v1.clientBuf = cb;
            }
        }
        // float = (quantized + offset) * scale for QNN scale/offset encodings.
        static bool scale_offset(const Qnn_Tensor_t& t, float* scale, int32_t* offset) {
            const Qnn_QuantizeParams_t& q = t.version == QNN_TENSOR_VERSION_2 ? t.v2.quantizeParams : t.v1.quantizeParams;
            if (q.encodingDefinition != QNN_DEFINITION_DEFINED || q.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                return false;
            }
            *scale  = q.scaleOffsetEncoding.scale;
            *offset = q.scaleOffsetEncoding.offset;
            return true;
        }

        static bool write_tensor(const Qnn_Tensor_t& t, std::vector<uint8_t>& buf, const float* src, size_t count) {
            if (count != element_count(t)) {
                LOG_ERROR("ltxv qnn: input %s expects %zu values, got %zu", tensor_name(t), element_count(t), count);
                return false;
            }
            Qnn_DataType_t type = tensor_type(t);
            if (type == QNN_DATATYPE_FLOAT_16) {
                ggml_fp32_to_fp16_row(src, reinterpret_cast<ggml_fp16_t*>(buf.data()), static_cast<int64_t>(count));
            } else if (type == QNN_DATATYPE_FLOAT_32) {
                memcpy(buf.data(), src, count * sizeof(float));
            } else if (type == QNN_DATATYPE_UFIXED_POINT_16 || type == QNN_DATATYPE_UFIXED_POINT_8) {
                float scale    = 1.f;
                int32_t offset = 0;
                if (!scale_offset(t, &scale, &offset)) {
                    LOG_ERROR("ltxv qnn: input %s has no scale/offset encoding", tensor_name(t));
                    return false;
                }
                const float qmax = type == QNN_DATATYPE_UFIXED_POINT_16 ? 65535.f : 255.f;
                for (size_t i = 0; i < count; i++) {
                    float q = std::round(src[i] / scale - static_cast<float>(offset));
                    q       = std::min(std::max(q, 0.f), qmax);
                    if (type == QNN_DATATYPE_UFIXED_POINT_16) {
                        reinterpret_cast<uint16_t*>(buf.data())[i] = static_cast<uint16_t>(q);
                    } else {
                        buf[i] = static_cast<uint8_t>(q);
                    }
                }
            } else {
                LOG_ERROR("ltxv qnn: unsupported input type %d for %s", static_cast<int>(type), tensor_name(t));
                return false;
            }
            return true;
        }
        static bool read_tensor(const Qnn_Tensor_t& t, const std::vector<uint8_t>& buf, float* dst, size_t count) {
            if (count != element_count(t)) {
                LOG_ERROR("ltxv qnn: output %s has %zu values, expected %zu", tensor_name(t), element_count(t), count);
                return false;
            }
            Qnn_DataType_t type = tensor_type(t);
            if (type == QNN_DATATYPE_FLOAT_16) {
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(buf.data()), dst, static_cast<int64_t>(count));
            } else if (type == QNN_DATATYPE_FLOAT_32) {
                memcpy(dst, buf.data(), count * sizeof(float));
            } else if (type == QNN_DATATYPE_UFIXED_POINT_16 || type == QNN_DATATYPE_UFIXED_POINT_8) {
                float scale    = 1.f;
                int32_t offset = 0;
                if (!scale_offset(t, &scale, &offset)) {
                    LOG_ERROR("ltxv qnn: output %s has no scale/offset encoding", tensor_name(t));
                    return false;
                }
                for (size_t i = 0; i < count; i++) {
                    int32_t q = type == QNN_DATATYPE_UFIXED_POINT_16 ? reinterpret_cast<const uint16_t*>(buf.data())[i] : buf[i];
                    dst[i]    = static_cast<float>(q + offset) * scale;
                }
            } else {
                LOG_ERROR("ltxv qnn: unsupported output type %d for %s", static_cast<int>(type), tensor_name(t));
                return false;
            }
            return true;
        }

        // Deep-copies the tensor descriptors of one graph out of the system-context binary info.
        static bool copy_graph_info(const QnnSystemContext_BinaryInfo_t* bi, const std::string& name, Graph& g) {
            if (bi == nullptr) {
                return false;
            }
            uint32_t num_graphs                        = 0;
            const QnnSystemContext_GraphInfo_t* infos = nullptr;
            switch (bi->version) {
                case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
                    num_graphs = bi->contextBinaryInfoV1.numGraphs;
                    infos      = bi->contextBinaryInfoV1.graphs;
                    break;
                case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
                    num_graphs = bi->contextBinaryInfoV2.numGraphs;
                    infos      = bi->contextBinaryInfoV2.graphs;
                    break;
                default:
                    num_graphs = bi->contextBinaryInfoV3.numGraphs;
                    infos      = bi->contextBinaryInfoV3.graphs;
                    break;
            }
            for (uint32_t i = 0; i < num_graphs; i++) {
                // graphName / inputs / outputs lead every GraphInfo version with the same layout.
                const auto& gi = infos[i].graphInfoV1;
                if (name != gi.graphName) {
                    continue;
                }
                auto copy = [&](const Qnn_Tensor_t* src, uint32_t n, std::vector<Qnn_Tensor_t>& dst) {
                    for (uint32_t k = 0; k < n; k++) {
                        Qnn_Tensor_t t = src[k];
                        uint32_t rank  = t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
                        uint32_t* dims = t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
                        g.dims.emplace_back(dims, dims + rank);
                        dst.push_back(t);
                    }
                };
                copy(gi.graphInputs, gi.numGraphInputs, g.inputs);
                copy(gi.graphOutputs, gi.numGraphOutputs, g.outputs);
                // Point every descriptor at our own dimension arrays and names, which outlive the binary info.
                size_t d = 0;
                for (auto* list : {&g.inputs, &g.outputs}) {
                    for (auto& t : *list) {
                        auto& dims = g.dims[d++];
                        if (t.version == QNN_TENSOR_VERSION_2) {
                            t.v2.dimensions = dims.data();
                            t.v2.name       = strdup(t.v2.name);
                        } else {
                            t.v1.dimensions = dims.data();
                            t.v1.name       = strdup(t.v1.name);
                        }
                    }
                }
                return true;
            }
            LOG_ERROR("ltxv qnn: graph '%s' not found in context binary", name.c_str());
            return false;
        }
    };

    struct LTXVQnnRunner : public DiffusionModelRunner {
        static constexpr int kChunks      = 4;
        static constexpr int kTextTokens  = 128;
        static constexpr float kMaskValue = -10000.f;

        LTXVConfig config;
        std::shared_ptr<QnnRuntime> qnn;
        std::string qnn_dir;
        bool ready = false;
        std::map<float, std::vector<float>> t_freq_cache;
        std::string rope_key;
        std::vector<float> rope_cos, rope_sin;

        LTXVQnnRunner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map,
                      const std::string& prefix,
                      const std::string& qnn_dir,
                      const std::string& qnn_lib_dir,
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(LTXVConfig::detect_from_weights(tensor_storage_map, prefix)),
              qnn(QnnRuntime::acquire(qnn_lib_dir)),
              qnn_dir(qnn_dir) {
            ready = qnn != nullptr && ensure_loaded();
            if (!ready) {
                LOG_ERROR("ltxv qnn: NPU transformer unavailable");
            }
            active() = this;
        }

        ~LTXVQnnRunner() override {
            if (active() == this) {
                active() = nullptr;
            }
        }

        // The transformer currently bound to the HTP, so the VAE can borrow its address space.
        static LTXVQnnRunner*& active() {
            static LTXVQnnRunner* runner = nullptr;
            return runner;
        }

        static std::string chunk_name(int i) { return "ltxv_chunk" + std::to_string(i); }

        // Chunks [0, resident) stay loaded. When the whole transformer does not fit the HTP address
        // space (Hexagon v69), the remaining chunks take turns in one slot, loaded on every step.
        int resident        = kChunks;
        bool streaming      = false;
        uint64_t spill_fill = 0;

        std::string chunk_path(int i) const { return qnn_dir + "/" + chunk_name(i) + ".bin"; }

        bool ensure_loaded() {
            bool all = true;
            for (int i = 0; i < resident; i++) {
                all = all && qnn->graph(chunk_name(i)) != nullptr;
            }
            if (all && (streaming || resident == kChunks)) {
                return true;
            }
            release();  // (re)load the whole group so the shared spill-fill buffer is registered first
            if (spill_fill == 0) {
                for (int i = 0; i < kChunks; i++) {
                    spill_fill = std::max(spill_fill, qnn->spill_fill_size(chunk_path(i)));
                }
                LOG_INFO("ltxv qnn: largest chunk spill-fill buffer %" PRIu64 " MB", spill_fill >> 20);
            }
            const int want = streaming ? resident : kChunks;
            for (int i = 0; i < want; i++) {
                if (qnn->load_graph(chunk_path(i), chunk_name(i), i > 0 && !streaming, "ltxv_dit", spill_fill)) {
                    continue;
                }
                if (i == 0 || streaming) {
                    LOG_ERROR("ltxv qnn: failed to load %s", chunk_name(i).c_str());
                    return false;
                }
                // Out of HTP address space: keep [0, i-1) resident and stream the rest through one slot.
                qnn->unload(chunk_name(i - 1));
                resident  = i - 1;
                streaming = true;
                LOG_WARN("ltxv qnn: the transformer does not fit the HTP at once; %d chunk(s) stay loaded, %d stream per step",
                         resident, kChunks - resident);
                return true;
            }
            if (spill_fill > 0) {
                LOG_INFO("ltxv qnn: transformer chunks share a %" PRIu64 " MB spill-fill buffer", spill_fill >> 20);
            }
            return true;
        }

        // The graph for chunk i, loading it into the streaming slot when it is not resident.
        QnnRuntime::Graph* chunk(int i) {
            if (auto* g = qnn->graph(chunk_name(i))) {
                return g;
            }
            for (int j = resident; j < kChunks; j++) {
                qnn->unload(chunk_name(j));
            }
            if (!qnn->load_graph(chunk_path(i), chunk_name(i), false, "ltxv_dit", spill_fill)) {
                return nullptr;
            }
            return qnn->graph(chunk_name(i));
        }

        void release() {
            for (int i = 0; i < kChunks; i++) {
                qnn->unload(chunk_name(i));
            }
        }

        std::string get_desc() override {
            return "ltxv_qnn";
        }

        // Weights live inside the context binaries; nothing to load through ggml.
        void get_param_tensors(std::map<std::string, ggml_tensor*>&, const std::string&) override {}

        const std::vector<float>& t_freq(float t) {
            auto it = t_freq_cache.find(t);
            if (it != t_freq_cache.end()) {
                return it->second;
            }
            // diffusers Timesteps(256, flip_sin_to_cos=True, shift=0) of t * timestep_scale_multiplier.
            std::vector<float> f(256);
            const int half = 128;
            for (int i = 0; i < half; i++) {
                double freq = std::exp(-std::log(10000.0) * i / half);
                double arg  = static_cast<double>(t) * config.timestep_scale_multiplier * freq;
                f[i]        = static_cast<float>(std::cos(arg));
                f[half + i] = static_cast<float>(std::sin(arg));
            }
            return t_freq_cache.emplace(t, std::move(f)).first->second;
        }

        void build_rope(int64_t w, int64_t h, int64_t frames, float fps) {
            std::string key = std::to_string(w) + "x" + std::to_string(h) + "x" + std::to_string(frames) + "@" + std::to_string(fps);
            if (key == rope_key) {
                return;
            }
            const int dim   = static_cast<int>(config.hidden_size);
            const int n     = dim / 6;
            const int pad   = dim / 2 - 3 * n;
            const auto& mp  = config.positional_embedding_max_pos;
            const int st    = std::get<0>(config.vae_scale_factors);
            const int ss    = std::get<1>(config.vae_scale_factors);
            const int64_t T = w * h * frames;
            std::vector<double> indices(n);
            for (int i = 0; i < n; i++) {
                double ratio = n > 1 ? static_cast<double>(i) / (n - 1) : 0.0;
                indices[i]   = std::exp(std::log(config.positional_embedding_theta) * ratio) * (M_PI / 2);
            }
            rope_cos.assign(static_cast<size_t>(T) * dim / 2, 1.f);
            rope_sin.assign(static_cast<size_t>(T) * dim / 2, 0.f);
            size_t token = 0;
            for (int64_t t = 0; t < frames; t++) {
                double pt = std::max(0.0, static_cast<double>(t * st + 1 - st)) / fps / mp[0];
                for (int64_t y = 0; y < h; y++) {
                    for (int64_t x = 0; x < w; x++, token++) {
                        double coords[3] = {pt, static_cast<double>(y * ss) / mp[1], static_cast<double>(x * ss) / mp[2]};
                        size_t base      = token * dim / 2 + pad;
                        for (int i = 0; i < n; i++) {
                            for (int a = 0; a < 3; a++) {
                                double f                 = indices[i] * (coords[a] * 2 - 1);
                                rope_cos[base + i * 3 + a] = static_cast<float>(std::cos(f));
                                rope_sin[base + i * 3 + a] = static_cast<float>(std::sin(f));
                            }
                        }
                    }
                }
            }
            rope_key = key;
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& p) override {
            SD_UNUSED(n_threads);
            GGML_ASSERT(p.x != nullptr && p.timesteps != nullptr);
            if (!ready || !ensure_loaded()) {
                return {};
            }
            const auto* extra = diffusion_extra_as<LTXAVDiffusionExtra>(p);
            if (extra->video_positions != nullptr && !extra->video_positions->empty()) {
                LOG_ERROR("ltxv qnn: keyframe (end image) conditioning is not supported on the NPU");
                return {};
            }
            const auto& x       = *p.x;
            const int64_t W     = x.shape()[0];
            const int64_t H     = x.shape()[1];
            const int64_t F     = x.shape()[2];
            const int64_t C     = config.in_channels;
            const int64_t T     = W * H * F;
            const int64_t D     = config.hidden_size;
            if (x.shape()[3] != C) {
                LOG_ERROR("ltxv qnn: batched inputs are not supported (channels %" PRId64 ")", x.shape()[3]);
                return {};
            }
            if (p.context == nullptr || p.context->numel() != kTextTokens * config.caption_channels) {
                LOG_ERROR("ltxv qnn: expected %d T5 tokens", kTextTokens);
                return {};
            }

            // ggml [W, H, F, C] -> token-major [T, C]
            std::vector<float> latent(static_cast<size_t>(T * C));
            const float* xs = x.data();
            for (int64_t c = 0; c < C; c++) {
                for (int64_t tok = 0; tok < T; tok++) {
                    latent[tok * C + c] = xs[c * T + tok];
                }
            }

            const auto& ts = *p.timesteps;
            std::vector<float> freq(static_cast<size_t>(T * 256));
            for (int64_t tok = 0; tok < T; tok++) {
                float t        = ts.numel() == T ? ts.data()[tok] : ts.data()[0];
                const auto& tf = t_freq(t);
                std::copy(tf.begin(), tf.end(), freq.begin() + tok * 256);
            }

            std::vector<float> mask(kTextTokens, 0.f);
            if (p.y != nullptr && p.y->numel() == kTextTokens) {
                for (int i = 0; i < kTextTokens; i++) {
                    mask[i] = std::isfinite(p.y->data()[i]) ? p.y->data()[i] : kMaskValue;
                }
            }
            build_rope(W, H, F, extra->frame_rate > 0.f ? extra->frame_rate : 24.f);

            auto set_common = [&](QnnRuntime::Graph& g) {
                return QnnRuntime::set_input(g, "mask", mask.data(), mask.size()) &&
                       QnnRuntime::set_input(g, "cos", rope_cos.data(), rope_cos.size()) &&
                       QnnRuntime::set_input(g, "sin", rope_sin.data(), rope_sin.size());
            };
            QnnRuntime::Graph* g0 = chunk(0);
            bool ok               = g0 != nullptr &&
                      QnnRuntime::set_input(*g0, "latent", latent.data(), latent.size()) &&
                      QnnRuntime::set_input(*g0, "t_freq", freq.data(), freq.size()) &&
                      QnnRuntime::set_input(*g0, "caption", p.context->data(), p.context->numel()) &&
                      set_common(*g0) && qnn->execute(*g0);
            std::vector<float> out(static_cast<size_t>(T * C));
            if (!streaming) {
                // All chunks resident: hand device buffers straight from graph to graph.
                for (int i = 1; ok && i < kChunks; i++) {
                    auto& prev = *qnn->graph(chunk_name(i - 1));
                    auto& g    = *qnn->graph(chunk_name(i));
                    ok         = set_common(g) && QnnRuntime::forward_tensor(prev, i == 1 ? "x" : "x_out", g, "x") &&
                         QnnRuntime::forward_tensor(*g0, "tmod", g, "tmod") &&
                         QnnRuntime::forward_tensor(*g0, "context", g, "context") &&
                         (i != kChunks - 1 || QnnRuntime::forward_tensor(*g0, "emb", g, "emb")) &&
                         qnn->execute(g);
                }
                ok = ok && QnnRuntime::get_output(*qnn->graph(chunk_name(kChunks - 1)), "out", out.data(), out.size());
            } else {
                // Streamed chunks lose their buffers when unloaded, so pass activations through host memory.
                std::vector<float> h(static_cast<size_t>(T * D)), tmod(static_cast<size_t>(T * 6 * D));
                std::vector<float> emb(static_cast<size_t>(T * D)), ctx(static_cast<size_t>(kTextTokens * D));
                ok = ok && QnnRuntime::get_output(*g0, "x", h.data(), h.size()) &&
                     QnnRuntime::get_output(*g0, "tmod", tmod.data(), tmod.size()) &&
                     QnnRuntime::get_output(*g0, "emb", emb.data(), emb.size()) &&
                     QnnRuntime::get_output(*g0, "context", ctx.data(), ctx.size());
                for (int i = 1; ok && i < kChunks; i++) {
                    const bool last      = i == kChunks - 1;
                    QnnRuntime::Graph* g = chunk(i);
                    ok                   = g != nullptr && set_common(*g) &&
                         QnnRuntime::set_input(*g, "x", h.data(), h.size()) &&
                         QnnRuntime::set_input(*g, "tmod", tmod.data(), tmod.size()) &&
                         QnnRuntime::set_input(*g, "context", ctx.data(), ctx.size()) &&
                         (!last || QnnRuntime::set_input(*g, "emb", emb.data(), emb.size())) &&
                         qnn->execute(*g) &&
                         (last ? QnnRuntime::get_output(*g, "out", out.data(), out.size())
                               : QnnRuntime::get_output(*g, "x_out", h.data(), h.size()));
                }
            }
            if (!ok) {
                return {};
            }

            // token-major [T, C] -> ggml [W, H, F, C]
            sd::Tensor<float> result(x.shape());
            float* rs = result.data();
            for (int64_t c = 0; c < C; c++) {
                for (int64_t tok = 0; tok < T; tok++) {
                    rs[c * T + tok] = out[tok * C + c];
                }
            }
            SD_UNUSED(D);
            return result;
        }
    };

}  // namespace LTXV

// LTX-Video VAE whose decoder runs from a QNN context binary (ltxv_vae_<W>x<H>x<F>.bin). Encoding
// (image-to-video) and any size without a compiled decoder fall back to the ggml implementation.
struct LTXVideoQnnVAE : public LTXVideoVAE {
    std::shared_ptr<LTXV::QnnRuntime> qnn;
    std::string qnn_dir;

    LTXVideoQnnVAE(ggml_backend_t backend,
                   const String2TensorStorage& tensor_storage_map,
                   const std::string& prefix,
                   SDVersion version,
                   const std::string& qnn_dir,
                   const std::string& qnn_lib_dir,
                   std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : LTXVideoVAE(backend, tensor_storage_map, prefix, false, version, weight_manager),
          qnn(LTXV::QnnRuntime::acquire(qnn_lib_dir)),
          qnn_dir(qnn_dir) {}

    std::string get_desc() override {
        return "ltx_video_vae_qnn";
    }

    using LTXVideoVAE::encode;

    // Image-to-video start frames are encoded on the CPU, followed by the T5 pass. Together with the
    // resident transformer chunks that exceeds the RAM of 12 GB phones, so free the transformer
    // first; it reloads lazily before sampling (a few seconds).
    sd::Tensor<float> encode(int n_threads,
                             const sd::Tensor<float>& x,
                             sd_tiling_params_t tiling_params,
                             bool circular_x,
                             bool circular_y) override {
        if (auto* dit = LTXV::LTXVQnnRunner::active()) {
            dit->release();
        }
        return LTXVideoVAE::encode(n_threads, x, tiling_params, circular_x, circular_y);
    }

    sd::Tensor<float> decode(int n_threads,
                             const sd::Tensor<float>& x,
                             sd_tiling_params_t tiling_params,
                             bool decode_video,
                             bool circular_x,
                             bool circular_y,
                             bool silent) override {
        const int64_t W = x.shape()[0], H = x.shape()[1], T = x.shape()[2];
        const int64_t F = (T - 1) * 8 + 1;
        std::string name = "ltxv_vae_" + std::to_string(W * 32) + "x" + std::to_string(H * 32) + "x" + std::to_string(F);
        std::string path = qnn_dir + "/" + name + ".bin";
        if (qnn == nullptr || access(path.c_str(), R_OK) != 0 || x.shape()[3] != 128) {
            return LTXVideoVAE::decode(n_threads, x, tiling_params, decode_video, circular_x, circular_y, silent);
        }
        int64_t t0 = ggml_time_ms();
        bool borrowed = false;
        if (qnn->graph(name) == nullptr && !qnn->load_graph(path, name, true)) {
            // Not enough HTP address space next to the transformer: free it, it reloads lazily.
            if (auto* dit = LTXV::LTXVQnnRunner::active()) {
                dit->release();
                borrowed = true;
            }
            if (!qnn->load_graph(path, name)) {
                LOG_WARN("ltxv qnn: NPU VAE unavailable, decoding on the CPU");
                return LTXVideoVAE::decode(n_threads, x, tiling_params, decode_video, circular_x, circular_y, silent);
            }
        }
        // Latents arrive per-channel normalized; [W, H, T, C] in ggml order is [1, C, T, H, W] in memory.
        auto z = un_normalize_latents(n_threads, x);
        runner_end();
        auto* g = qnn->graph(name);
        const float temb = decode_timestep_tensor.values()[0];
        sd::Tensor<float> out(x.dim() == 5 ? std::vector<int64_t>{W * 32, H * 32, F, 3, 1}
                                           : std::vector<int64_t>{W * 32, H * 32, F, 3});
        bool ok = !z.empty() &&
                  LTXV::QnnRuntime::set_input(*g, "z", z.data(), z.numel()) &&
                  LTXV::QnnRuntime::set_input(*g, "temb", &temb, 1) &&
                  qnn->execute(*g) &&
                  LTXV::QnnRuntime::get_output(*g, "video", out.data(), out.numel());
        if (borrowed) {
            qnn->unload(name);
        }
        if (!ok) {
            LOG_ERROR("ltxv qnn: NPU VAE decode failed");
            return {};
        }
        if (scale_input) {
            scale_tensor_to_0_1(&out);
        }
        LOG_INFO("ltxv qnn: VAE decode on the NPU took %.2fs", (ggml_time_ms() - t0) / 1000.f);
        return out;
    }
};

#endif  // SD_LTXV_QNN
#endif  // __SD_MODEL_DIFFUSION_LTXV_QNN_HPP__
