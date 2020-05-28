#include <migraphx/normalize.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/operators.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::normalize{}});
}

void check_normalize_op(const migraphx::operation& untuned, const migraphx::operation& tuned, std::vector<migraphx::shape> shapes)
{
    migraphx::program p1;
    {
        std::vector<migraphx::instruction_ref> inputs;
        std::size_t n = 0;
        for(auto&& s:shapes)
        {
            auto ins = p1.add_parameter("x" + std::to_string(n), s);
            inputs.push_back(ins);
            n++;
        }
        p1.add_instruction(untuned, inputs);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        std::vector<migraphx::instruction_ref> inputs;
        std::size_t n = 0;
        for(auto&& s:shapes)
        {
            auto ins = p2.add_parameter("x" + std::to_string(n), s);
            inputs.push_back(ins);
            n++;
        }
        p2.add_instruction(tuned, inputs);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(slice)
{
    check_normalize_op(migraphx::op::slice{{-1}, {0}, {2}}, migraphx::op::slice{{3}, {0}, {2}}, {{migraphx::shape::float_type, {1, 2, 3, 4}}});
    check_normalize_op(migraphx::op::slice{{3}, {0}, {-1}}, migraphx::op::slice{{3}, {0}, {3}}, {{migraphx::shape::float_type, {1, 2, 3, 4}}});
    check_normalize_op(migraphx::op::slice{{3}, {-2}, {-1}}, migraphx::op::slice{{3}, {2}, {3}}, {{migraphx::shape::float_type, {1, 2, 3, 4}}});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
