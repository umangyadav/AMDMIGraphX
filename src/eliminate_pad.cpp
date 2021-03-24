#include <migraphx/eliminate_pad.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/im2col.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/pad.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_pad::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        const std::string& op_name = ins->name();
        if(op_name != "convolution" and op_name != "im2col" and op_name != "pooling")
            continue;
        auto input = ins->inputs().front();
        if(input->name() != "pad")
            continue;
        if(op_name == "convolution" or op_name == "im2col")
        {
            update_op(input, ins, p);
            p.debug_print();
        }
        else if(op_name == "pooling")
            update_pooling(input, ins, p);
    }
}

void eliminate_pad::update_op(const instruction_ref& input,
                              const instruction_ref& ins,
                              module& p) const
{
    auto pad_op = any_cast<op::pad>(input->get_operator());
    // if(!pad_op.symmetric())
    //     return;

    auto kdims    = input->get_shape().lens().size() - 2;
    auto kdims_it = pad_op.pads.begin() + 2;

    std::vector<size_t> pads_l(kdims_it, kdims_it + kdims);
    std::vector<size_t> pads_r(kdims_it + kdims + 2, pad_op.pads.end());
    

    auto op = any_cast<op::convolution>(ins->get_operator());
 
    std::transform(op.padding_l.begin(), op.padding_l.end(), pads_l.begin(), op.padding_l.begin(), std::plus<size_t>());
    std::transform(op.padding_r.begin(), op.padding_r.end(), pads_r.begin(), op.padding_r.begin(), std::plus<size_t>());

    std::vector<instruction_ref> new_inputs{ins->inputs()};
    new_inputs.front() = input->inputs().front();

    p.replace_instruction(ins, op, new_inputs);
}

void eliminate_pad::update_pooling(const instruction_ref& input,
                                   const instruction_ref& ins,
                                   module& p) const
{
    auto op = any_cast<op::pooling>(ins->get_operator());
    if(op.mode == "average")
    {
        return;
    }

    auto pad_op = any_cast<op::pad>(input->get_operator());
    // if(!pad_op.symmetric())
    //     return;

    auto kdims    = input->get_shape().lens().size() - 2;
    auto kdims_it = pad_op.pads.begin() + 2;

    std::vector<size_t> pads_l(kdims_it, kdims_it + kdims);
    std::vector<size_t> pads_r(kdims_it + kdims, pad_op.pads.end());

    std::transform(op.padding_l.begin(), op.padding_l.end(), pads_l.begin(), op.padding_l.begin(), std::plus<size_t>());
    std::transform(op.padding_r.begin(), op.padding_r.end(), pads_r.begin(), op.padding_r.begin(), std::plus<size_t>());

    std::vector<instruction_ref> new_inputs{ins->inputs()};
    new_inputs.front() = input->inputs().front();

    p.replace_instruction(ins, op, new_inputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
