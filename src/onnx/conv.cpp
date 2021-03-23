#include <migraphx/onnx/conv.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

void recalc_conv_attributes(value& v, size_t kdims)
{
    if(v["padding_l"].size() != kdims)
    {
        v["padding_l"].resize(kdims);
        std::fill_n(v["padding_l"].begin(), kdims, 0);
    }
    if(v["padding_r"].size() != kdims)
    {
        v["padding_r"].resize(kdims);
        std::fill_n(v["padding_r"].begin(), kdims, 0);
    }
    if(v["stride"].size() != kdims)
    {
        v["stride"].resize(kdims);
        std::fill_n(v["stride"].begin(), kdims, 1);
    }
    if(v["dilation"].size() != kdims)
    {
        v["dilation"].resize(kdims);
        std::fill_n(v["dilation"].begin(), kdims, 1);
    }
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
