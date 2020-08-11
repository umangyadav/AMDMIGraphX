#include <migraphx/gpu/adjust_allocation.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void adjust_allocation::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        // skip instruction with no input
        if(ins->inputs().empty())
            continue;

        // Skip non-gpu operators
        if(ins->get_operator().is_context_free())
            continue;

        auto alias_ins = instruction::get_output_alias(ins, true);
        if(alias_ins->name() != "hip::allocate" and alias_ins->name() != "@param")
            continue;
        // shape allocated is different from actual shape
        // of the instruction, reallocate and replace the previous one
        if(alias_ins->get_shape() == ins->get_shape())
            continue;
        auto alloc_ins = p.insert_instruction(ins, hip_allocate{ins->get_shape()});
        p.replace_instruction(alias_ins, alloc_ins);
        // If the memory is an output parameter then copy the memory to the parameter
        if(alias_ins->name() == "@param")
        {
            p.insert_instruction(std::next(ins), hip_copy{}, ins, alias_ins);
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
