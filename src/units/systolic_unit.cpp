#include "units/systolic_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <stdexcept>

namespace sim {

SystolicUnit::SystolicUnit(std::string name, const SystolicConfig& cfg,
                           Scheduler* sched, TensorStore*, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), os_(os) {}

Cycle SystolicUnit::fill_latency() const {
    if (cfg_.bidirectional)
        return (cfg_.rows-1+1)/2 + (cfg_.cols-1+1)/2;
    return (cfg_.rows-1) + (cfg_.cols-1);
}

Cycle SystolicUnit::compute_latency(uint32_t M, uint32_t K, uint32_t N) const {
    if (!M||!K||!N) return 0;
    if (M > cfg_.rows || N > cfg_.cols)
        throw std::runtime_error("SystolicUnit: oversized GEMM was not subtiled");
    return static_cast<Cycle>(K) + fill_latency();
}

void SystolicUnit::handle(const Event& e, EventEngine& engine) {
    if (e.type == EventType::OP_START) {
        uint32_t M=cfg_.rows, K=cfg_.d_head, N=cfg_.cols;
        if (const auto* s=std::any_cast<GemmShape>(&e.payload)) {
            M=s->M; K=s->K; N=s->N;
        }
        const Cycle lat = compute_latency(M, K, N);
        os_<<"  ["<<name()<<"]  GEMM_START  instr="<<e.instr
           <<"  @cycle="<<e.cycle<<"  shape=["<<M<<"x"<<K<<"x"<<N<<"]"
           <<"  array=["<<cfg_.rows<<"x"<<cfg_.cols<<"]"
           <<"  mode="<<(cfg_.bidirectional?"bidir":"unidir")
           <<"  fill="<<fill_latency()<<"  lat="<<lat
           <<(e.label.empty()?"":" \""+e.label+"\"")<<"\n";

        Event done=e;
        done.type=EventType::OP_DONE;
        done.cycle=e.cycle+lat;
        done.seq=engine.next_seq();
        engine.schedule(done);

    } else if (e.type == EventType::OP_DONE) {
        os_<<"  ["<<name()<<"]  GEMM_DONE  instr="<<e.instr
           <<"  @cycle="<<e.cycle
           <<(e.label.empty()?"":" \""+e.label+"\"")<<"\n";
        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
