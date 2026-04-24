#include "schedule/dummy_schedule.h"

namespace tpu {

namespace {
// Symbolic unit ids — the real simulator will source these from a config.
// Keeping them here makes the dummy trace easy to eyeball.
constexpr UnitId kSystolic = 1;
constexpr UnitId kVector0  = 10;
constexpr UnitId kVector1  = 11;
constexpr UnitId kDma      = 20;
} // namespace

void load_dummy_schedule(EventEngine& eng) {
    // Intentionally inserted out of order to prove the priority queue
    // re-sorts into (time, unit, seq) order at dispatch.
    eng.schedule(2000, kVector0,  EventKind::PrintMessage, "vec0: layernorm tile 0");
    eng.schedule( 500, kDma,      EventKind::PrintMessage, "dma : HBM -> SRAM load A");
    eng.schedule(1000, kSystolic, EventKind::PrintMessage, "sys : matmul tile (0,0)");

    // Same-time collision across two units. Expected dispatch order:
    // Vector0 first (unit=10) then Vector1 (unit=11).
    eng.schedule(1500, kVector1,  EventKind::PrintMessage, "vec1: softmax tile 0");
    eng.schedule(1500, kVector0,  EventKind::PrintMessage, "vec0: softmax tile 0");

    // Same (time, unit) collision — broken by insertion order via Seq.
    eng.schedule(3000, kSystolic, EventKind::PrintMessage, "sys : matmul tile (0,1) issue");
    eng.schedule(3000, kSystolic, EventKind::Tick,         "sys : matmul tile (0,1) retire");
}

} // namespace tpu
