#include <doctest/doctest.h>
#include <stdexcept>
#include "schedule/schedule.h"

using namespace sim;

TEST_CASE("schedule parses instructions and typed params correctly") {
    const char* yaml = R"(
schedule:
  - id: 0
    op: dma_load
    unit: dma
    params: { bytes: 1024, src: hbm, dst: ibuf, overlap: true }
    label: "load K tile"
  - id: 1
    op: gemm
    unit: systolic
    params: { M: 128, K: 128, N: 128, scale: 0.5 }
    depends_on: [0]
    label: "QK^T"
)";
    Schedule s = Schedule::from_yaml_string(yaml);
    REQUIRE(s.instructions.size() == 2);

    const auto& i0 = s.instructions[0];
    REQUIRE(i0.id    == 0);
    REQUIRE(i0.op    == "dma_load");
    REQUIRE(i0.unit  == "dma");
    REQUIRE(i0.label == "load K tile");
    REQUIRE(pget_int(i0.params, "bytes")   == 1024);
    REQUIRE(pget_str(i0.params, "src")     == "hbm");
    REQUIRE(pget_bool(i0.params, "overlap") == true);

    const auto& i1 = s.instructions[1];
    REQUIRE(i1.depends_on == std::vector<InstructionId>{0});
    REQUIRE(pget_int(i1.params, "M")     == 128);
    REQUIRE(pget_dbl(i1.params, "scale") == doctest::Approx(0.5));
}

TEST_CASE("schedule auto-assigns sequential ids when omitted") {
    const char* yaml = R"(
schedule:
  - op: a
  - op: b
  - op: c
)";
    Schedule s = Schedule::from_yaml_string(yaml);
    REQUIRE(s.instructions[0].id == 0);
    REQUIRE(s.instructions[1].id == 1);
    REQUIRE(s.instructions[2].id == 2);
}

TEST_CASE("schedule validate rejects duplicate instruction ids") {
    const char* yaml = R"(
schedule:
  - id: 0
    op: a
  - id: 0
    op: b
)";
    REQUIRE_THROWS_AS(Schedule::from_yaml_string(yaml), std::runtime_error);
}

TEST_CASE("schedule validate rejects unknown dependency id") {
    const char* yaml = R"(
schedule:
  - id: 0
    op: a
    depends_on: [99]
)";
    REQUIRE_THROWS_AS(Schedule::from_yaml_string(yaml), std::runtime_error);
}

TEST_CASE("schedule validate rejects dependency cycle") {
    const char* yaml = R"(
schedule:
  - id: 0
    op: a
    depends_on: [1]
  - id: 1
    op: b
    depends_on: [0]
)";
    REQUIRE_THROWS_AS(Schedule::from_yaml_string(yaml), std::runtime_error);
}

TEST_CASE("schedule validate accepts a valid diamond graph") {
    // A -> B, A -> C, B -> D, C -> D
    const char* yaml = R"(
schedule:
  - id: 0
    op: A
  - id: 1
    op: B
    depends_on: [0]
  - id: 2
    op: C
    depends_on: [0]
  - id: 3
    op: D
    depends_on: [1, 2]
)";
    REQUIRE_NOTHROW(Schedule::from_yaml_string(yaml));
    Schedule s = Schedule::from_yaml_string(yaml);
    REQUIRE(s.instructions.size() == 4);
    REQUIRE(s.instructions[3].depends_on.size() == 2);
}

TEST_CASE("schedule missing 'schedule' key throws") {
    REQUIRE_THROWS_AS(Schedule::from_yaml_string("{}"), std::runtime_error);
}
