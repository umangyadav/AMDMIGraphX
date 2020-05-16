#include <migraphx/gpu/gather_elements.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/gather_elements.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_gather_elements::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_gather_elements::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    return device::gather_elements(ctx.get_stream().get(), args.back(), args[0], args[1], op.axis);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
