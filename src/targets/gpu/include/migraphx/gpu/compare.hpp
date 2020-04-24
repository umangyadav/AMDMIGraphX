#ifndef MIGRAPHX_GUARD_RTGLIB_COMPARE_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMPARE_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/compare.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_compare : binary_device<hip_compare, device::compare>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
