#include <doctest/doctest.h>
#include <stdexcept>
#include "config/arch_config.h"

using namespace sim;

TEST_CASE("ArchConfig has correct defaults") {
    ArchConfig c;
    REQUIRE(c.clock_ghz              == doctest::Approx(1.0));
    REQUIRE(c.systolic.rows          == 128);
    REQUIRE(c.systolic.cols          == 128);
    REQUIRE(c.systolic.precision     == "BF16");
    REQUIRE(c.vector_cores           == 3);
    REQUIRE(c.access_cores           == 2);
    REQUIRE(c.sram.ibuf_kb           == 4096);
    REQUIRE(c.sram.obuf_kb           == 4096);
    REQUIRE(c.sram.banking_factor    == 8);
    REQUIRE(c.sram.private_tandem_kb == 512);
    REQUIRE(c.hbm.bandwidth_tb_s     == doctest::Approx(2.0));
    REQUIRE(c.hbm.latency_cycles     == 200);
    REQUIRE(c.dma.channels           == 1);
}

TEST_CASE("ArchConfig loads all fields from YAML string") {
    const char* yaml = R"(
clock_ghz: 1.5
systolic:
  rows: 256
  cols: 256
  precision: FP8
vector_cores: 4
access_cores: 1
sram:
  ibuf_kb: 8192
  obuf_kb: 2048
  banking_factor: 16
  private_tandem_kb: 1024
hbm:
  bandwidth_tb_s: 3.35
  latency_cycles: 150
dma:
  channels: 2
)";
    ArchConfig c = ArchConfig::from_yaml_string(yaml);
    REQUIRE(c.clock_ghz              == doctest::Approx(1.5));
    REQUIRE(c.systolic.rows          == 256);
    REQUIRE(c.systolic.cols          == 256);
    REQUIRE(c.systolic.precision     == "FP8");
    REQUIRE(c.vector_cores           == 4);
    REQUIRE(c.access_cores           == 1);
    REQUIRE(c.sram.ibuf_kb           == 8192);
    REQUIRE(c.sram.obuf_kb           == 2048);
    REQUIRE(c.sram.banking_factor    == 16);
    REQUIRE(c.sram.private_tandem_kb == 1024);
    REQUIRE(c.hbm.bandwidth_tb_s     == doctest::Approx(3.35));
    REQUIRE(c.hbm.latency_cycles     == 150);
    REQUIRE(c.dma.channels           == 2);
}

TEST_CASE("ArchConfig round-trips through YAML serialization") {
    ArchConfig orig;
    orig.clock_ghz          = 2.0;
    orig.systolic.rows      = 64;
    orig.systolic.precision = "FP16";
    orig.hbm.bandwidth_tb_s = 1.2;
    orig.dma.channels       = 4;

    std::string s = orig.to_yaml_string();
    ArchConfig  c = ArchConfig::from_yaml_string(s);
    REQUIRE(c.clock_ghz          == doctest::Approx(2.0));
    REQUIRE(c.systolic.rows      == 64);
    REQUIRE(c.systolic.precision == "FP16");
    REQUIRE(c.hbm.bandwidth_tb_s == doctest::Approx(1.2));
    REQUIRE(c.dma.channels       == 4);
}

TEST_CASE("hbm_bytes_per_cycle is correctly derived") {
    ArchConfig c;
    c.clock_ghz          = 1.0;  // 1 GHz -> 1 ns/cycle
    c.hbm.bandwidth_tb_s = 1.0;  // 1 TB/s = 1e12 B/s = 1000 B/cycle at 1 GHz
    REQUIRE(c.hbm_bytes_per_cycle() == doctest::Approx(1000.0));

    c.clock_ghz          = 2.0;  // 2 GHz -> 0.5 ns/cycle
    c.hbm.bandwidth_tb_s = 2.0;  // 2 TB/s -> still 1000 B/cycle
    REQUIRE(c.hbm_bytes_per_cycle() == doctest::Approx(1000.0));
}

TEST_CASE("ArchConfig rejects clock_ghz <= 0") {
    REQUIRE_THROWS_AS(ArchConfig::from_yaml_string("clock_ghz: 0.0"), std::runtime_error);
    REQUIRE_THROWS_AS(ArchConfig::from_yaml_string("clock_ghz: -1.0"), std::runtime_error);
}
