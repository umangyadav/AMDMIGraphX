#ifndef MIGRAPHX_GUARD_OPERATORS_IFF_HPP
#define MIGRAPHX_GUARD_OPERATORS_IFF_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct iff
{
    std::string sub_module;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.sub_module, "sub_graph"));
    }

    std::string name() const { return "if"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        return inputs[0];
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
