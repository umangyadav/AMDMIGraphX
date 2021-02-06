#include <unordered_set>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/auto_any_cast.hpp>
#include <migraphx/value.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void normalize_ops::apply(module& m) const
{
    std::cout << "loc1" << std::endl;
    for(auto ins : iterator_for(m))
    {
        std::cout << "loc1, ins" << std::endl;
        auto inputs = ins->inputs();
        if(inputs.empty())
            continue;

        // apply this pass to sub_modules
        auto& module_inputs = ins->module_inputs();
        for(auto& smdl : module_inputs)
        {
            apply(*smdl);
        }

        auto lens                    = inputs[0]->get_shape().lens();
        migraphx::operation tuned_op = ins->get_operator();
        if(normalize_attributes(tuned_op, lens))
        {
            m.replace_instruction(ins, tuned_op, inputs);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
