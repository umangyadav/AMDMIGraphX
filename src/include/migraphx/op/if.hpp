#ifndef MIGRAPHX_GUARD_OPERATORS_IF_HPP
#define MIGRAPHX_GUARD_OPERATORS_IF_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct iff
{
    module_ref sub_module;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.sub_module_name, "sub_graph"));
    }

    std::string name() const { return "if"; }
    shape compute_shape(const std::vector<shape>&) const
    {
        std::vector<shape> out_shapes = sub_module->get_output_shapes();
        return out_shapes[0];
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
