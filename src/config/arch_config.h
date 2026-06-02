#pragma once
#include <cstdint>
#include <string>

namespace sim {

struct SystolicConfig {
    uint32_t    rows          = 128;
    uint32_t    cols          = 128;
    std::string precision     = "BF16";   // FP8 | FP16 | BF16 | FP32
    bool        bidirectional = false;    // halves pipeline-fill latency
    uint32_t    d_head        = 128;      // attention head dim (= K dimension)
};

struct SramConfig {
    uint32_t ibuf_kb           = 4096;
    uint32_t obuf_kb           = 4096;
    uint32_t banking_factor    = 8;
    uint32_t private_vector_kb = 512;
};

struct HbmConfig {
    double   bandwidth_tb_s = 2.0;
    uint32_t latency_cycles = 200;
};

struct DmaConfig {
    uint32_t channels = 1;
};

struct VectorCoreConfig {
    uint32_t simd_width  = 64;
    uint32_t exp_latency = 4;
};

struct AccessCoreConfig {
    uint32_t bandwidth = 64;   // elements per cycle
};

struct ArchConfig {
    double           clock_ghz    = 1.0;
    SystolicConfig   systolic;
    uint32_t         systolic_units = 1;
    uint32_t         vector_cores = 3;
    uint32_t         access_cores = 1;
    SramConfig       sram;
    HbmConfig        hbm;
    DmaConfig        dma;
    VectorCoreConfig vector_core;
    AccessCoreConfig access_core;

    // Derived: bytes transferred per HBM cycle at this clock frequency.
    double hbm_bytes_per_cycle() const {
        return (hbm.bandwidth_tb_s * 1e12) / (clock_ghz * 1e9);
    }

    static ArchConfig from_yaml_file(const std::string& path);
    static ArchConfig from_yaml_string(const std::string& yaml);
    std::string       to_yaml_string() const;
};

}  // namespace sim
