#ifndef MIGRAPHX_GUARD_RTGLIB_IFF_HPP
#define MIGRAPHX_GUARD_RTGLIB_IFF_HPP

#include <migraphx/shape.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/op/iff.hpp>
#include <migraphx/module_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_iff
{
    op::iff op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::iff"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument compute(
        const std::vector<argument>& args,
        const std::vector<module_ref>& modules,
        std::function<std::vector<argument>(module_ref& mdl, const std::vector<argument>& inputs)>
            run) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
