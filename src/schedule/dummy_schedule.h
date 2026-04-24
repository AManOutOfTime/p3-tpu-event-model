#pragma once

#include "engine/engine.h"

namespace tpu {

// Populate the engine with a small, hand-written set of events that
// exercise ordering: same-time collisions, out-of-order insertion, and
// multiple "units". This is the Phase 0 stand-in for a real schedule
// emitted from an op graph.
void load_dummy_schedule(EventEngine& eng);

} // namespace tpu
