#include <migraphx/gpu/if.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_if::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    inputs.pop_back();
    return op.compute_shape(inputs, std::move(mods));
}

argument hip_if::compute(
    const std::vector<argument>& args,
    const std::vector<module_ref>& mods,
    std::function<std::vector<argument>(
        module_ref& mdl, const std::unordered_map<std::string, argument>& inputs)>& run) const
{
    auto cond      = args.at(0).at<bool>();
    module_ref mod = cond ? mods[0] : mods[1];
    std::unordered_map<std::string, argument> params;
    auto out_shapes = mod->get_output_shapes();
    assert(out_shapes.size() + 1 == args.size());
    for(std::size_t i = 0; i < out_shapes.size(); ++i)
    {
        std::string out_name = "#output_" + std::to_string(i);
        params[out_name]     = args.at(i + 1);
    }

    auto results = run(mod, params);
    auto res = migraphx::gpu::from_gpu(results[0]);
    std::cout << "res = " << res << std::endl;

    std::cout << "arg_size = " << args.size() << std::endl;
    auto arg1 = migraphx::gpu::from_gpu(args.at(1));
    std::cout << "arg1 = " << arg1 << std::endl;

    // context ctx{};
    // gpu_copy(ctx, results[0], args[1]);

    return args.at(1);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
