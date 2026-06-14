#pragma once
#include "core/types.h"
#include <cstring>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sim {

// ---------------------------------------------------------------------------
// SmallStr<TAG, Code>  — compact intern handle for fixed-vocabulary fields.
//
// Code defaults to uint8_t (1 byte, 255 unique values) which is enough for
// op names (~50), unit names (~10), and param key names (~30).
// LabelStr uses uint16_t (2 bytes, 65535 unique values) because trace-mode
// schedules generate one unique label per tile per instruction, easily
// exceeding 255 entries.
//
// DEFAULT CONSTRUCTOR: zero-initializes Code only — no pool call.
// Pool is only touched on explicit string assignment (once per instruction
// during schedule build). Vector reallocation just copies Code bytes.
// ---------------------------------------------------------------------------
template<int TAG, typename Code = uint8_t>
class SmallStr {
public:
    SmallStr() = default;  // code_ = 0 (= ""), NO pool call
    SmallStr(const std::string& s) : code_(intern(s)) {}
    SmallStr(std::string&&      s) : code_(intern(s)) {}
    SmallStr(const char*        s) : code_(intern(s)) {}

    SmallStr& operator=(const std::string& s) { code_ = intern(s); return *this; }
    SmallStr& operator=(std::string&&      s) { code_ = intern(s); return *this; }
    SmallStr& operator=(const char*        s) { code_ = intern(s); return *this; }

    // Implicit conversion — lets existing callers pass inst.op/unit to
    // functions taking const string& without any source change.
    operator const std::string&() const { return pool().str(code_); }

    bool operator==(const SmallStr&    o) const { return code_ == o.code_; }
    bool operator!=(const SmallStr&    o) const { return code_ != o.code_; }
    bool operator==(const std::string& o) const { return pool().str(code_) == o; }
    bool operator!=(const std::string& o) const { return !(*this == o); }
    bool operator==(const char*        o) const { return pool().str(code_) == o; }
    bool operator!=(const char*        o) const { return !(*this == o); }

    // Concatenation — string::operator+ is a template, so user-defined
    // conversions don't fire for argument deduction. Provide explicit forms.
    std::string operator+(const char*        r) const { return str() + r; }
    std::string operator+(const std::string& r) const { return str() + r; }
    friend std::string operator+(const char*        l, const SmallStr& r)
        { return std::string(l) + r.str(); }
    friend std::string operator+(const std::string& l, const SmallStr& r)
        { return l + r.str(); }

    const std::string& str()   const { return pool().str(code_); }
    bool               empty() const { return code_ == 0; }
    Code               code()  const { return code_; }

    // Forwarded string methods — needed where SmallStr is used as label
    // (tiler truncation, test assertions with .find()).
    std::size_t size() const { return str().size(); }
    std::string substr(std::size_t pos, std::size_t len = std::string::npos) const
        { return str().substr(pos, len); }
    std::size_t find(const std::string& s, std::size_t pos = 0) const
        { return str().find(s, pos); }
    std::size_t find(const char* s, std::size_t pos = 0) const
        { return str().find(s, pos); }

private:
    struct Pool {
        std::vector<std::string>                 table;
        std::unordered_map<std::string, Code>    index;
        Pool() { table.push_back(""); index[""] = 0; }
        Code intern(const std::string& s) {
            auto it = index.find(s);
            if (it != index.end()) return it->second;
            auto idx = static_cast<Code>(table.size());
            table.push_back(s);
            index[s] = idx;
            return idx;
        }
        const std::string& str(Code c) const { return table[c]; }
    };
    static Pool& pool() { static Pool p; return p; }
    static Code intern(const std::string& s) { return pool().intern(s); }

    Code code_ = 0;  // 0 == ""
};

using OpStr    = SmallStr<0>;                    // 1 byte — ~50 op names
using UnitStr  = SmallStr<1>;                    // 1 byte — ~10 unit names
using LabelStr = SmallStr<2, uint16_t>;          // 2 bytes — many unique labels in trace mode
using KeyStr   = SmallStr<3>;

// ---------------------------------------------------------------------------
// CompactParamVal  — 16-byte tagged union.
//
// Replaces variant<int64_t, double, std::string, bool> which is always
// 40 bytes regardless of which type is stored (the variant must reserve space
// for the largest member, std::string = 32 bytes).
//
// Layout:  type_(1) + pad(7) + data_(8)  = 16 bytes
// Savings: 24 bytes × avg 2.5 params × 11.1M instructions ≈ 667 MB.
//
// String values are heap-allocated as null-terminated char arrays.
// In --no-trace (minimal) mode string params are dropped by the builder
// before reaching inst.params, so char* allocations are rare in practice.
// ---------------------------------------------------------------------------
class CompactParamVal {
public:
    enum Type : uint8_t { T_INT64 = 0, T_DOUBLE, T_STRING, T_BOOL };

    CompactParamVal()                    : type_(T_INT64)  { data_.i = 0; }
    CompactParamVal(int64_t       v)     : type_(T_INT64)  { data_.i = v; }
    CompactParamVal(double        v)     : type_(T_DOUBLE) { data_.d = v; }
    CompactParamVal(bool          v)     : type_(T_BOOL)   { data_.b = v; }
    CompactParamVal(const std::string& v): type_(T_STRING) { alloc(v.data(), v.size()); }
    CompactParamVal(std::string&&      v): type_(T_STRING) { alloc(v.data(), v.size()); }
    CompactParamVal(const char*        v): type_(T_STRING) { alloc(v, std::strlen(v)); }

    CompactParamVal(const CompactParamVal& o) : type_(o.type_) {
        if (type_ == T_STRING) alloc(o.data_.s, std::strlen(o.data_.s));
        else                   data_.i = o.data_.i;
    }
    CompactParamVal(CompactParamVal&& o) noexcept : type_(o.type_), data_(o.data_) {
        o.type_ = T_INT64; o.data_.i = 0;
    }
    ~CompactParamVal() { if (type_ == T_STRING) delete[] data_.s; }

    CompactParamVal& operator=(const CompactParamVal& o) {
        if (this != &o) { this->~CompactParamVal(); new(this) CompactParamVal(o); }
        return *this;
    }
    CompactParamVal& operator=(CompactParamVal&& o) noexcept {
        if (this != &o) { this->~CompactParamVal(); new(this) CompactParamVal(std::move(o)); }
        return *this;
    }
    CompactParamVal& operator=(int64_t v) {
        if (type_==T_STRING) delete[] data_.s;
        type_=T_INT64;  data_.i=v; return *this;
    }
    CompactParamVal& operator=(double v) {
        if (type_==T_STRING) delete[] data_.s;
        type_=T_DOUBLE; data_.d=v; return *this;
    }
    CompactParamVal& operator=(bool v) {
        if (type_==T_STRING) delete[] data_.s;
        type_=T_BOOL;   data_.b=v; return *this;
    }
    CompactParamVal& operator=(const std::string& v) {
        if (type_==T_STRING) delete[] data_.s;
        type_=T_STRING; alloc(v.data(), v.size()); return *this;
    }
    CompactParamVal& operator=(std::string&& v)  { return *this = static_cast<const std::string&>(v); }
    CompactParamVal& operator=(const char*   v) {
        if (type_==T_STRING) delete[] data_.s;
        type_=T_STRING; alloc(v, std::strlen(v)); return *this;
    }

    Type           type()          const { return type_; }
    bool           is_string()     const { return type_ == T_STRING; }
    const int64_t* get_int_if()    const { return type_==T_INT64  ? &data_.i : nullptr; }
    const double*  get_double_if() const { return type_==T_DOUBLE ? &data_.d : nullptr; }
    const bool*    get_bool_if()   const { return type_==T_BOOL   ? &data_.b : nullptr; }
    const char*    get_cstr()      const { return type_==T_STRING ? data_.s  : nullptr; }
    int64_t*       get_int_if()          { return type_==T_INT64  ? &data_.i : nullptr; }
    double*        get_double_if()       { return type_==T_DOUBLE ? &data_.d : nullptr; }

private:
    void alloc(const char* s, std::size_t n) {
        data_.s = new char[n+1]; std::memcpy(data_.s, s, n+1);
    }
    Type type_;           // 1 byte  — declared first to avoid -Wreorder
    // 7 bytes implicit padding (union needs 8-byte alignment from int64_t)
    union Data { int64_t i; double d; bool b; char* s; } data_;  // 8 bytes
    // sizeof(CompactParamVal) == 16
};

using ParamVal = CompactParamVal;

// ---------------------------------------------------------------------------
// ParamMap — flat vector-backed map with 1-byte intern keys.
//
// Key changed from std::string (32 bytes) to KeyStr (1 byte, SmallStr<3>).
// pair<KeyStr(1B+7Bpad), CompactParamVal(16B)> = 24 bytes per entry
// was pair<string(32B), variant(40B)> = 72 bytes → saves 48 B/entry.
// At ~2.5 params × 11.1M instructions: ≈ 1.33 GB total savings.
//
// Public interface is unchanged: find/operator[]/count still take const
// string& and intern on the fly. One extra overload accepts KeyStr directly
// for the minimal-mode filter loop that already holds a KeyStr value.
// ---------------------------------------------------------------------------
class ParamMap {
public:
    using value_type     = std::pair<KeyStr, ParamVal>;
    using storage        = std::vector<value_type>;
    using iterator       = storage::iterator;
    using const_iterator = storage::const_iterator;

    ParamMap() = default;
    ParamMap(std::initializer_list<value_type> init) : data_(init) {}

    iterator       begin()       { return data_.begin(); }
    iterator       end()         { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end()   const { return data_.end(); }

    iterator find(const std::string& k) {
        uint8_t code = KeyStr(k).code();
        for (auto it = data_.begin(); it != data_.end(); ++it)
            if (it->first.code() == code) return it;
        return data_.end();
    }
    const_iterator find(const std::string& k) const {
        uint8_t code = KeyStr(k).code();
        for (auto it = data_.begin(); it != data_.end(); ++it)
            if (it->first.code() == code) return it;
        return data_.end();
    }

    // Single overload handles const char*, std::string, and KeyStr —
    // all three implicitly convert to const string& with no ambiguity.
    ParamVal& operator[](const std::string& k) {
        uint8_t code = KeyStr(k).code();
        for (auto& kv : data_)
            if (kv.first.code() == code) return kv.second;
        data_.emplace_back(k, ParamVal{});
        return data_.back().second;
    }

    std::size_t count(const std::string& k) const { return find(k)==end() ? 0u : 1u; }
    std::size_t size()  const { return data_.size(); }
    bool        empty() const { return data_.empty(); }
    void        shrink() { storage{data_}.swap(data_); }

private:
    storage data_;
};

// Param accessors — updated to use CompactParamVal typed accessors.
inline int64_t pget_int(const ParamMap& p, const std::string& k, int64_t def = 0) {
    auto it = p.find(k);
    if (it == p.end()) return def;
    if (auto* v = it->second.get_int_if())    return *v;
    if (auto* v = it->second.get_double_if()) return static_cast<int64_t>(*v);
    return def;
}
inline double pget_dbl(const ParamMap& p, const std::string& k, double def = 0.0) {
    auto it = p.find(k);
    if (it == p.end()) return def;
    if (auto* v = it->second.get_double_if()) return *v;
    if (auto* v = it->second.get_int_if())    return static_cast<double>(*v);
    return def;
}
inline std::string pget_str(const ParamMap& p, const std::string& k,
                             const std::string& def = "") {
    auto it = p.find(k);
    if (it == p.end()) return def;
    const char* s = it->second.get_cstr();
    return s ? std::string(s) : def;
}
inline bool pget_bool(const ParamMap& p, const std::string& k, bool def = false) {
    auto it = p.find(k);
    if (it == p.end()) return def;
    if (auto* v = it->second.get_bool_if()) return *v;
    return def;
}

// ---------------------------------------------------------------------------
// Instruction
//
// Memory layout (64-bit) after all SmallStr substitutions:
//   id(4) op(1) unit(1) label(2) params(24) depends_on(24) = 56 bytes
//   (was 152 bytes with three std::string fields and std::variant params)
//   Savings from struct alone: 96 bytes × 11.1M instructions ≈ 1.07 GB
//   Plus ParamMap key saving: ~24 B/entry × 2.5 × 11.1M ≈ 667 MB
//   Grand total this PR: ~1.7 GB
// ---------------------------------------------------------------------------
struct Instruction {
    InstructionId id = 0;
    OpStr         op;      // 1 byte  (was 32 bytes)
    UnitStr       unit;    // 1 byte  (was 32 bytes)
    LabelStr      label;   // 2 bytes (was 32 bytes, always "" in --no-trace)
    ParamMap      params;
    std::vector<InstructionId> depends_on;
};

}  // namespace sim