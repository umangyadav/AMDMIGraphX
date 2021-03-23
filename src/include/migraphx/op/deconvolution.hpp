#ifndef MIGRAPHX_GUARD_OPERATORS_DECONVOLUTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_DECONVOLUTION_HPP

#include <array>
#include <migraphx/op/common.hpp>
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

struct deconvolution
{
    std::vector<std::size_t> padding_l  = {0, 0};
    std::vector<std::size_t> padding_r  = {0, 0};
    std::vector<std::size_t> stride   = {1, 1};
    std::vector<std::size_t> dilation = {1, 1};

    padding_mode_t padding_mode = default_;
    int group                   = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding_l, "padding_left"),
                    f(self.padding_r, "padding_right"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"),
                    f(self.group, "group"));
    }

    std::string name() const { return "deconvolution"; }

    void check_attribute_size() const
    {
        if(not(padding_l.size() == stride.size() and padding_l.size() == dilation.size() and padding_l.size() == padding_r.size()))
        {
            MIGRAPHX_THROW("deconvolution: inconsistent attribute sizes");
        }
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims().min_ndims(3);

        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        auto t               = input.type();
        size_t kdims         = input.lens().size() - 2;
        if(kdims != this->kdims())
        {
            MIGRAPHX_THROW("deconvolution: input k-dims does not match attribute size");
        }

        std::vector<size_t> output_lens{input.lens()[0], weights.lens()[1]};

        for(size_t i = 0; i < kdims; i++)
        {
            output_lens.push_back(std::size_t(std::max<std::ptrdiff_t>(
                1,
                stride[i] * (input.lens()[i + 2] - 1) +
                    ((weights.lens()[i + 2] - 1) * dilation[i] + 1) - (padding_l[i] + padding_r[i]))));
        }
        return {t, output_lens};
    }

    size_t kdims() const
    {
        check_attribute_size();
        return padding_l.size();
    }

    std::vector<size_t> padding() const
    {
        if(padding_l != padding_r)
            MIGRAPHX_THROW("CONVOLUTION: padding is asymmetric");
        return padding_l;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
