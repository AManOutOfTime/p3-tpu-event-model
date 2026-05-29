#pragma once

#include "schedule/schedule.h"
#include <string>

namespace sim {

struct LlamaScheduleConfig {
    std::string mode = "attention";      // attention | layer | prefill | decode
    std::string schedule_granularity = "detailed";  // detailed | coarse
    uint32_t seq_len = 128;
    uint32_t prompt_len = 128;
    uint32_t generation_steps = 1;
    uint32_t num_layers = 1;
    uint32_t num_q_heads = 1;
    uint32_t num_kv_heads = 1;
    uint32_t gqa_group_size = 0;
    uint32_t head_dim = 0;
    uint32_t hidden_dim = 128;
    uint32_t intermediate_dim = 256;
    uint32_t vocab_size = 32000;
    uint32_t tile_rows = 128;
    uint32_t tile_cols = 128;
    uint32_t linear_tile_rows = 128;
    uint32_t linear_tile_cols = 128;
    uint32_t max_seq_len = 0;
    uint32_t dtype_bytes = 2;
    uint32_t sram_kv_capacity_kb = 0;
    uint32_t kv_cache_block_tokens = 0;
    uint32_t kv_stage_buffers = 2;
    bool kv_cache_enabled = false;
    std::string kv_cache_location = "sram";  // sram | hbm
    std::string kv_prefetch = "double_buffer";  // none | double_buffer
    std::string kv_cache_eviction_policy = "fail";  // fail | spill_to_hbm
};

Schedule build_attention_schedule(const LlamaScheduleConfig& cfg);
Schedule build_transformer_layer_schedule(const LlamaScheduleConfig& cfg);
Schedule build_prefill_decode_schedule(const LlamaScheduleConfig& cfg);
Schedule build_llama_schedule(const LlamaScheduleConfig& cfg);

LlamaScheduleConfig llama_config_from_yaml_file(const std::string& path);
LlamaScheduleConfig llama_config_from_yaml_string(const std::string& yaml);

}  // namespace sim
