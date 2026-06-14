#include "schedule/op_registry.h"
#include <stdexcept>

namespace sim {

void OpRegistry::register_op(const std::string& name, OpHandler handler) {
    fast_ops_[OpStr(name).code()] = handler;
    ops_[name] = std::move(handler);
}

bool OpRegistry::has(const std::string& name) const {
    return ops_.count(name) > 0;
}

const OpHandler& OpRegistry::get(const std::string& name) const {
    auto it = ops_.find(name);
    if (it == ops_.end())
        throw std::runtime_error("OpRegistry: unknown op '" + name + "'");
    return it->second;
}

const OpHandler& OpRegistry::get(uint8_t code) const {
    auto it = fast_ops_.find(code);
    if (it == fast_ops_.end())
        throw std::runtime_error("OpRegistry: unknown op code " +
                                 std::to_string(code));
    return it->second;
}

}  // namespace sim