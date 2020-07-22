#ifndef MIGRAPHX_GUARD_RTGLIB_MEMORY_COLORING2_HPP
#define MIGRAPHX_GUARD_RTGLIB_MEMORY_COLORING2_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct program;

struct memory_coloring2
{
    std::string allocation_op{};
    std::string name() const { return "memory_coloring2"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
