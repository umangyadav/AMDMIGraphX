#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/onehot.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument
onehot(hipStream_t stream, argument result, argument arg_indices, argument arg_value, int axis)
{
    auto out_shape           = result.get_shape();
    int n_rank               = static_cast<int>(out_shape.lens().size());
    int tuned_axis           = (axis < 0) ? (axis + n_rank) : axis;
    auto in_comp_lens        = out_shape.lens();
    int depth                = in_comp_lens[tuned_axis];
    in_comp_lens[tuned_axis] = 1;
    shape in_comp_shape{out_shape.type(), in_comp_lens};
    std::size_t nelements = out_shape.elements();

    visit_all(result, arg_value)([&](auto output, auto val) {
        // retrieve the off_value and on_value
        const auto* val_ptr = device_cast(val.data());
        auto* output_ptr    = device_cast(output.data());

        arg_indices.visit([&](auto ind) {
            const auto* ind_ptr = device_cast(ind.data());
            hip_visit_all(out_shape, in_comp_shape)([&](auto out_s, auto in_s) {
                gs_launch(stream, nelements, 256)([=](auto i)
                                                      __device__ { output_ptr[i] = val_ptr[0]; });
                gs_launch(stream, in_comp_shape.elements(), 256)([=](auto i) __device__ {
                    int axis_idx                 = ind_ptr[i];
                    axis_idx                     = (axis_idx < 0) ? axis_idx + depth : axis_idx;
                    auto idx                     = in_s.multi(i);
                    idx[tuned_axis]              = axis_idx;
                    output_ptr[out_s.index(idx)] = val_ptr[1];
                });
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
