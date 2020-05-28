#include <migraphx/normalize.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void normalize::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if (starts_with(ins->name(), "@"))
            continue;
        auto op = ins->get_operator().normalize(to_shapes(ins->inputs()));
        p.replace_instruction(ins, op, ins->inputs());
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
