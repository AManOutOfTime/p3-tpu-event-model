#include "config/arch_config.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace sim {

namespace {

template <typename T>
void try_load(const YAML::Node& n, const char* key, T& out) {
    if (n && n[key]) out = n[key].as<T>();
}

ArchConfig parse(const YAML::Node& root) {
    ArchConfig c;
    try_load(root, "clock_ghz", c.clock_ghz);
    if (c.clock_ghz <= 0.0)
        throw std::runtime_error("clock_ghz must be > 0");

    if (auto s = root["systolic"]) {
        try_load(s, "rows",          c.systolic.rows);
        try_load(s, "cols",          c.systolic.cols);
        try_load(s, "precision",     c.systolic.precision);
        try_load(s, "bidirectional", c.systolic.bidirectional);
        try_load(s, "d_head",        c.systolic.d_head);
        try_load(s, "dataflow",             c.systolic.dataflow);
        try_load(s, "weight_load_cycles",   c.systolic.weight_load_cycles);
        try_load(s, "weight_double_buffer", c.systolic.weight_double_buffer);
    }
    try_load(root, "structural_k_tiling", c.structural_k_tiling);
    try_load(root, "model_sram",          c.model_sram);
    try_load(root, "stage_double_buffer", c.stage_double_buffer);
    try_load(root, "systolic_units", c.systolic_units);
    if (c.systolic_units == 0)
        throw std::runtime_error("systolic_units must be > 0");
    try_load(root, "vector_cores", c.vector_cores);
    try_load(root, "access_cores", c.access_cores);

    if (auto s = root["sram"]) {
        try_load(s, "ibuf_kb",           c.sram.ibuf_kb);
        try_load(s, "obuf_kb",           c.sram.obuf_kb);
        try_load(s, "banking_factor",    c.sram.banking_factor);
        try_load(s, "private_vector_kb", c.sram.private_vector_kb);
    }
    if (auto s = root["hbm"]) {
        try_load(s, "bandwidth_tb_s", c.hbm.bandwidth_tb_s);
        try_load(s, "latency_cycles", c.hbm.latency_cycles);
        try_load(s, "pipelined",      c.hbm.pipelined);
    }
    if (auto s = root["dma"]) {
        try_load(s, "channels", c.dma.channels);
    }
    if (auto s = root["vector_core"]) {
        try_load(s, "simd_width",  c.vector_core.simd_width);
        try_load(s, "exp_latency", c.vector_core.exp_latency);
    }
    if (auto s = root["access_core"]) {
        try_load(s, "bandwidth", c.access_core.bandwidth);
    }
    return c;
}

}  // namespace

ArchConfig ArchConfig::from_yaml_file(const std::string& path) {
    return parse(YAML::LoadFile(path));
}

ArchConfig ArchConfig::from_yaml_string(const std::string& yaml) {
    return parse(YAML::Load(yaml));
}

std::string ArchConfig::to_yaml_string() const {
    YAML::Emitter out;
    out << YAML::BeginMap
        << YAML::Key << "clock_ghz"    << YAML::Value << clock_ghz
        << YAML::Key << "systolic"     << YAML::BeginMap
            << YAML::Key << "rows"      << YAML::Value << systolic.rows
            << YAML::Key << "cols"      << YAML::Value << systolic.cols
            << YAML::Key << "precision" << YAML::Value << systolic.precision
            << YAML::Key << "bidirectional"       << YAML::Value << systolic.bidirectional
            << YAML::Key << "d_head"              << YAML::Value << systolic.d_head
            << YAML::Key << "dataflow"            << YAML::Value << systolic.dataflow
            << YAML::Key << "weight_load_cycles"  << YAML::Value << systolic.weight_load_cycles
            << YAML::Key << "weight_double_buffer"<< YAML::Value << systolic.weight_double_buffer
            << YAML::EndMap
        << YAML::Key << "structural_k_tiling" << YAML::Value << structural_k_tiling
        << YAML::Key << "model_sram"          << YAML::Value << model_sram
        << YAML::Key << "stage_double_buffer" << YAML::Value << stage_double_buffer
        << YAML::Key << "systolic_units" << YAML::Value << systolic_units
        << YAML::Key << "vector_cores" << YAML::Value << vector_cores
        << YAML::Key << "access_cores" << YAML::Value << access_cores
        << YAML::Key << "sram"         << YAML::BeginMap
            << YAML::Key << "ibuf_kb"           << YAML::Value << sram.ibuf_kb
            << YAML::Key << "obuf_kb"           << YAML::Value << sram.obuf_kb
            << YAML::Key << "banking_factor"    << YAML::Value << sram.banking_factor
            << YAML::Key << "private_vector_kb" << YAML::Value << sram.private_vector_kb
            << YAML::EndMap
        << YAML::Key << "hbm"          << YAML::BeginMap
            << YAML::Key << "bandwidth_tb_s" << YAML::Value << hbm.bandwidth_tb_s
            << YAML::Key << "latency_cycles" << YAML::Value << hbm.latency_cycles
            << YAML::Key << "pipelined"      << YAML::Value << hbm.pipelined
            << YAML::EndMap
        << YAML::Key << "dma"          << YAML::BeginMap
            << YAML::Key << "channels" << YAML::Value << dma.channels
            << YAML::EndMap
        << YAML::EndMap;
    return out.c_str();
}

}  // namespace sim
