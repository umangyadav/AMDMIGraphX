
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

migraphx::instruction_ref
add_layernorm(migraphx::program& p, migraphx::instruction_ref x, std::vector<size_t> dims)
{
    auto scale =
        mm->add_parameter("scale", migraphx::shape{migraphx::shape::float_type, {dims.back()}});
    auto bias =
        mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {dims.back()}});
    auto epsilon  = mm->add_literal(1e-12f);
    auto exponent = mm->add_literal(2.0f);

    auto mean            = mm->add_instruction(migraphx::op::reduce_mean({2}), x);
    auto mean_mbcast     = mm->add_instruction(migraphx::op::multibroadcast{{dims}}, mean);
    auto sub             = mm->add_instruction(migraphx::op::sub{}, x, mean_mbcast);
    auto exponent_mbcast = mm->add_instruction(migraphx::op::multibroadcast{{dims}}, exponent);
    auto pow             = mm->add_instruction(migraphx::op::pow{}, sub, exponent_mbcast);
    auto var             = mm->add_instruction(migraphx::op::reduce_mean({2}), pow);
    auto epsilon_mbcast =
        mm->add_instruction(migraphx::op::multibroadcast{{1, dims.at(1), 1}}, epsilon);
    auto add_epsilon  = mm->add_instruction(migraphx::op::add{}, var, epsilon_mbcast);
    auto sqrt         = mm->add_instruction(migraphx::op::sqrt{}, add_epsilon);
    auto sqrt_mbcast  = mm->add_instruction(migraphx::op::multibroadcast{dims}, sqrt);
    auto div          = mm->add_instruction(migraphx::op::div{}, sub, sqrt_mbcast);
    auto scale_mbcast = mm->add_instruction(migraphx::op::multibroadcast{dims}, scale);
    auto mul          = mm->add_instruction(migraphx::op::mul{}, scale_mbcast, div);
    auto bias_mbcast  = mm->add_instruction(migraphx::op::multibroadcast{dims}, bias);
    return mm->add_instruction(migraphx::op::add{}, mul, bias_mbcast);
}

struct test_layernorm : verify_program<test_layernorm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> dims = {1, 1, 5};
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(p, x, dims);
        return p;
    }
};

struct test_layernorm2 : verify_program<test_layernorm2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> dims = {1, 4, 24};
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(p, x, dims);
        return p;
    }
};

struct test_layernorm_triadd : verify_program<test_layernorm_triadd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> dims = {1, 4, 24};
        auto x    = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        auto y    = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, dims});
        auto z    = p.add_parameter("z", migraphx::shape{migraphx::shape::float_type, dims});
        auto add1 = p.add_instruction(migraphx::op::add{}, x, y);
        auto add2 = p.add_instruction(migraphx::op::add{}, add1, z);
        add_layernorm(p, add2, dims);
        return p;
    }
};
