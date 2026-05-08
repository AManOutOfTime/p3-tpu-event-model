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
    }
    try_load(root, "vector_cores", c.vector_cores);
    try_load(root, "access_cores", c.access_cores);

    if (auto s = root["sram"]) {
        try_load(s, "ibuf_kb",           c.sram.ibuf_kb);
        try_load(s, "obuf_kb",           c.sram.obuf_kb);
        try_load(s, "banking_factor",    c.sram.banking_factor);
        try_load(s, "private_tandem_kb", c.sram.private_tandem_kb);
    }
    if (auto s = root["hbm"]) {
        try_load(s, "bandwidth_tb_s", c.hbm.bandwidth_tb_s);
        try_load(s, "latency_cycles", c.hbm.latency_cycles);
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
            << YAML::Key << "rows"          << YAML::Value << systolic.rows
            << YAML::Key << "cols"          << YAML::Value << systolic.cols
            << YAML::Key << "precision"     << YAML::Value << systolic.precision
            << YAML::Key << "bidirectional" << YAML::Value << systolic.bidirectional
            << YAML::EndMap
        << YAML::Key << "vector_cores" << YAML::Value << vector_cores
        << YAML::Key << "access_cores" << YAML::Value << access_cores
        << YAML::Key << "sram"         << YAML::BeginMap
            << YAML::Key << "ibuf_kb"           << YAML::Value << sram.ibuf_kb
            << YAML::Key << "obuf_kb"           << YAML::Value << sram.obuf_kb
            << YAML::Key << "banking_factor"    << YAML::Value << sram.banking_factor
            << YAML::Key << "private_tandem_kb" << YAML::Value << sram.private_tandem_kb
            << YAML::EndMap
        << YAML::Key << "hbm"          << YAML::BeginMap
            << YAML::Key << "bandwidth_tb_s" << YAML::Value << hbm.bandwidth_tb_s
            << YAML::Key << "latency_cycles" << YAML::Value << hbm.latency_cycles
            << YAML::EndMap
        << YAML::Key << "dma"          << YAML::BeginMap
            << YAML::Key << "channels" << YAML::Value << dma.channels
            << YAML::EndMap
        << YAML::Key << "vector_core"  << YAML::BeginMap
            << YAML::Key << "simd_width"  << YAML::Value << vector_core.simd_width
            << YAML::Key << "exp_latency" << YAML::Value << vector_core.exp_latency
            << YAML::EndMap
        << YAML::Key << "access_core"  << YAML::BeginMap
            << YAML::Key << "bandwidth" << YAML::Value << access_core.bandwidth
            << YAML::EndMap
        << YAML::EndMap;
    return out.c_str();
}

}  // namespace sim
