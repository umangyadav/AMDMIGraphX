#ifndef MIGRAPHX_GUARD_OPERATORS_ONEHOT_HPP
#define MIGRAPHX_GUARD_OPERATORS_ONEHOT_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct onehot
{
    std::size_t depth = 0;
    int64_t axis      = -1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.depth, "depth"), f(self.axis, "axis"));
    }

    std::string name() const { return "onehot"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).standard();
        if(depth == 0)
        {
            MIGRAPHX_THROW("ONEHOT: Attribute depth cannot be 0!");
        }

        if(inputs[1].elements() != 2)
        {
            MIGRAPHX_THROW("ONEHOT: value argument must contain 2 elements!");
        }

        auto lens      = inputs[0].lens();
        int64_t n_rank = static_cast<int>(lens.size());
        if(axis > n_rank || axis < -(n_rank + 1))
        {
            MIGRAPHX_THROW("ONEHOT: Axis is out of range.");
        }

        // negative axis means counting dimensions from back
        int64_t tuned_axis = (axis < 0) ? (n_rank + 1 + axis) : axis;

        auto type = inputs[1].type();
        lens.insert(lens.begin() + tuned_axis, depth);

        return {type, lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        int64_t n_rank = static_cast<int>(output_shape.lens().size());
        // negative axis means counting dimensions from back
        int64_t tuned_axis = (axis < 0) ? (n_rank + axis) : axis;

        argument result{output_shape};
        // get the off_value and on_value
        visit_all(result, args[1])([=](auto output, auto value) {
            using type = typename decltype(value)::value_type;
            std::vector<type> vec_values;
            vec_values.assign(value.begin(), value.end());
            type off_value = vec_values.front();
            type on_value  = vec_values.back();
            std::fill(output.begin(), output.end(), off_value);

            args[0].visit([&](auto index_buf) {
                shape_for_each(args[0].get_shape(), [&](const auto& in_idx) {
                    auto out_idx = in_idx;
                    int64_t axis_index =
                        static_cast<int64_t>(index_buf(in_idx.begin(), in_idx.end()));
                    axis_index = (axis_index < 0) ? (axis_index + depth) : axis_index;
                    out_idx.insert(out_idx.begin() + tuned_axis, axis_index);
                    output[output_shape.index(out_idx.begin(), out_idx.end())] = on_value;
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
