#ifndef MIGRAPHX_GUARD_OPERATORS_GATHER_ELEMENTS_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHER_ELEMENTS_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>


namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct gather_elements
{
    int axis = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "gather_elements"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).standard().same_dims();
        if (inputs[0].lens().size() == 0)
        {
            MIGRAPHX_THROW("GATHER_ELEMENTS: input rank must be at least 1");
        }

        int n_rank = static_cast<int>(inputs[0].lens().size());
        if (axis <= -n_rank or axis >= n_rank - 1)
        {
            MIGRAPHX_THROW("GATHER_ELEMENTS: axis is out of rank");
        }

        
        return {inputs[0].type(), inputs[1].lens()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto input_shape = args[0].get_shape();
        int n_rank = static_cast<int>(output_shape.lens().size());
        int axis_index = (axis < 0) ? axis + n_rank : axis;
        int depth = static_cast<int>(input_shape.lens()[axis_index]);

        visit_all(result, args[0])([&](auto output, auto input) {
            args[1].visit([&](auto ind) {
                par_for(output_shape.elements(), [&](auto i) {
                    auto idx = output_shape.multi(i);
                    auto index = ind[i];
                    index = (index < 0) ? index + depth : index;
                    idx[axis_index] = index;
                    output[i] = input[input_shape.index(idx.begin(), idx.end())];
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
