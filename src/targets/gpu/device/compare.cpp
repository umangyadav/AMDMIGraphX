#include <migraphx/gpu/device/compare.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
static bool __device__ float_equal(T x, T y)
{
    return std::isfinite(x) and std::isfinite(y) and
           std::nextafter(x, std::numeric_limits<T>::lowest()) <= y and
           std::nextafter(x, std::numeric_limits<T>::max()) >= y;
}

void compare(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{

    nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ { return x == y; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
