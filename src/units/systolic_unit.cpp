#include "units/systolic_unit.h"
#include "core/event_engine.h"
#include "schedule/scheduler.h"
#include <cmath>
#include <algorithm>

namespace sim {

SystolicUnit::SystolicUnit(std::string name, const SystolicConfig& cfg,
                           Scheduler* sched, TensorStore* ts, std::ostream& os)
    : Unit(std::move(name)), cfg_(cfg), sched_(sched), ts_(ts), os_(os) {}

Cycle SystolicUnit::fill_latency() const {
    if (cfg_.bidirectional)
        return (cfg_.rows-1+1)/2 + (cfg_.cols-1+1)/2;
    return (cfg_.rows-1) + (cfg_.cols-1);
}

Cycle SystolicUnit::compute_latency(uint32_t M, uint32_t K, uint32_t N) const {
    if (!M||!K||!N) return 0;
    Cycle tm = (M+cfg_.rows-1)/cfg_.rows;
    Cycle tn = (N+cfg_.cols-1)/cfg_.cols;
    return tm * tn * (static_cast<Cycle>(K) + fill_latency());
}

void SystolicUnit::do_gemm(const GemmShape& s) {
    if (!ts_->has(s.src_a)||!ts_->has(s.src_b)) {
        os_<<"  ["<<name()<<"]  GEMM_COMPUTE SKIPPED (buffers not found)\n";
        return;
    }
    const auto& A = ts_->get(s.src_a);
    const auto& B = ts_->get(s.src_b);
    const uint32_t M=s.M, K=s.K, N=s.N;
    if (A.size()<(size_t)M*K || B.size()<(size_t)K*N) {
        os_<<"  ["<<name()<<"]  GEMM_COMPUTE ERROR size mismatch\n"; return;
    }
    std::vector<float> C(M*N, 0.f);
    const uint32_t TM=cfg_.rows, TN=cfg_.cols;
    for (uint32_t ti=0;ti<M;ti+=TM) {
        uint32_t ib=std::min(TM,M-ti);
        for (uint32_t tj=0;tj<N;tj+=TN) {
            uint32_t jb=std::min(TN,N-tj);
            for (uint32_t k=0;k<K;k++)
                for (uint32_t i=0;i<ib;i++) {
                    float a=A[(ti+i)*K+k];
                    for (uint32_t j=0;j<jb;j++)
                        C[(ti+i)*N+(tj+j)] += a*B[k*N+(tj+j)];
                }
        }
    }
    ts_->set(s.dst_c, std::move(C));
    os_<<"  ["<<name()<<"]  GEMM_COMPUTE \""<<s.src_a<<"\" ["<<M<<"x"<<K<<"]"
       <<" x \""<<s.src_b<<"\" ["<<K<<"x"<<N<<"] → \""<<s.dst_c<<"\"\n";
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
        if (ts_)
            if (const auto* s=std::any_cast<GemmShape>(&e.payload))
                if (!s->src_a.empty()&&!s->src_b.empty()&&!s->dst_c.empty())
                    do_gemm(*s);

        os_<<"  ["<<name()<<"]  GEMM_DONE  instr="<<e.instr
           <<"  @cycle="<<e.cycle
           <<(e.label.empty()?"":" \""+e.label+"\"")<<"\n";
        if (sched_) sched_->notify_done(e.instr);
    }
}

}  // namespace sim
