#include <migraphx/gpu/device/gather_elements.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument gather_elements(hipStream_t stream,
                     argument result,
                     argument arg_data,
                     argument arg_ind,
                     int axis)
{
    auto input_shape   = arg_data.get_shape();
    auto output_shape = result.get_shape();
    int n_rank = static_cast<int>(input_shape.lens().size());
    axis = (axis < 0) ? axis + n_rank : axis;
    int depth = static_cast<int>(input_shape.lens()[axis]);

    hip_visit_all(result, arg_data, output_shape, input_shape)([&](auto out, auto in, auto out_s, auto in_s) {
        auto* out_data = device_cast(out.data());
        const auto* in_data = device_cast(in.data());
        arg_ind.visit([&](auto ind) {
            const auto* ind_ptr = device_cast(ind.data());
            gs_launch(stream, output_shape.elements(), 256)([=](auto i) __device__ {
                auto idx = out_s.multi(i);
                auto index = ind_ptr[i];
                index = (index < 0) ? index + depth : index;
                idx[axis] = index;
                out_data[i] = in_data[in_s.index(idx)];
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
