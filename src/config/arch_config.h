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

    // ---- Dataflow / GEMM latency model (P0.1, P1.4) ----------------------
    // weight_stationary (default): weights K×N held in the PE array, M rows of
    //   activations stream through. Fragmentation is over (K, N); M streams.
    //   lat = ceil(N/cols) * ( ceil(K/rows)*(weight_load + M) + fill ).
    //   This is what production NN accelerators (TPU, etc.) actually do and is
    //   what makes prefill (large M) and decode (M=1) price correctly.
    // output_stationary: legacy K+fill model, M-insensitive. Kept selectable so
    //   dataflow itself can be a sweep axis.
    std::string dataflow             = "weight_stationary";
    // Cycles to load one K-block of weights into the array. 0 => auto (= rows),
    // i.e. one weight row per cycle over `rows` load lanes.
    uint32_t    weight_load_cycles   = 0;
    // P1.4: weight-FIFO double buffer. When true, weight-load(i+1) is prefetched
    // behind stream(i), so only the first load is exposed and each K-block then
    // costs max(weight_load, M). When false, load and stream are serial (pessimistic).
    bool        weight_double_buffer = true;
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
    // P1.5: pipelined HBM. When true, a DMA occupies its channel only for the
    // bandwidth term (ceil(bytes/bw)); the access latency is pipeline fill that
    // overlaps adjacent transfers, so a stream of loads is bandwidth-bound
    // (≈ latency + N·bytes/bw) rather than paying full latency every time
    // (N·(latency + bytes/bw)). Data is still ready only after latency+bw.
    // This is the accurate model for HBM-bandwidth sweeps. When false, the
    // channel is held for the full latency+bandwidth (legacy, pessimistic).
    bool     pipelined = true;
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

    // ---- P1.2: SRAM-pressure / partial-sum modeling toggles ---------------
    // structural_k_tiling: when true, Tiler splits oversized-K GEMMs into
    //   explicit per-K-block sub-GEMMs writing partial sums + `accumulate` ops,
    //   so partial-sum OBUF traffic is modeled as real instructions/cycles
    //   (instead of folded into the analytical latency term). Default off:
    //   in-array accumulation suffices for baseline timing.
    bool structural_k_tiling = false;
    // model_sram: when true, IBUF/OBUF capacity is bound at registration and
    //   GEMM/stage operands occupy bytes for their lifetime; over-capacity
    //   working sets stall/spill instead of being free. Makes SRAM size a real
    //   swept axis. Default off (infinite SRAM, original behavior).
    bool model_sram = false;
    // stage_double_buffer (S2): ping-pong the array operand staging buffer so
    //   stage(i+1) overlaps gemm(i) instead of waiting for the array to drain.
    bool stage_double_buffer = false;

    // Derived: bytes transferred per HBM cycle at this clock frequency.
    double hbm_bytes_per_cycle() const {
        return (hbm.bandwidth_tb_s * 1e12) / (clock_ghz * 1e9);
    }

    static ArchConfig from_yaml_file(const std::string& path);
    static ArchConfig from_yaml_string(const std::string& yaml);
    std::string       to_yaml_string() const;
};

}  // namespace sim
