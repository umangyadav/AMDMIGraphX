#ifndef MIGRAPH_GUARD_RTGLIB_GEMM_HPP
#define MIGRAPH_GUARD_RTGLIB_GEMM_HPP

#include <migraphx/gpu/lowering.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/add.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/context.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

struct miopen_gemm
{
    op::dot op;
    std::string name() const { return "gpu::gemm"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    int output_alias(const std::vector<shape>& shapes) const { return shapes.size() - 1; }
};

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
