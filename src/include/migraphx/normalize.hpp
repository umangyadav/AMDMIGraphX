#ifndef MIGRAPHX_GUARD_RTGLIB_NORMALIZE_HPP
#define MIGRAPHX_GUARD_RTGLIB_NORMALIZE_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Normalize all operators
 */
struct normalize
{
    std::string name() const { return "normalize"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
