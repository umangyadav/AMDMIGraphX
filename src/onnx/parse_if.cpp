#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_where : op_parser<parse_where>
{
    std::vector<op_desc> operators() const { return {{"If"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        migraphx::argument cond_arg = args.front()->eval();
        check_arg_empty(cond_arg, "PARSE_IF: Only support constant input condition");
        std::vector<bool> vec_conds;
        cond_arg.visit([&](auto s) { vec_conds.assign(s.begin(), s.end()); });
        if(vec_conds.size() != 1)
        {
            MIGRAPHX_THROW("PARSE_IF: condition input can have only one element!");
        }

        bool cond = vec_conds[0];
        // then branch
        if (cond)
        {
            onnx::GraphProto&& then_graph = info.attributes.at("then_branch").g();
            std::string graph_name = info.name + "_if";
            module* mdl = prog.create_module(graph_name);
            parse_graph(mdl, then_graph);
            
            // inputs of the return instruction are that of the output of the
            // if instruction
            instruction_ref ret_ins = std::prev(mdl->end());
            return ret_ins->inputs();
        }
        // else branch
        else
        {
            onnx::GraphProto&& else_graph = info.attributes.at("else_branch").g();
            std::string graph_name = info.name + "_else";
            module* mdl = prog.create_module(graph_name);
            parse_graph(mdl, then_graph);

            // inputs of the return instruction are that of the output of the
            // if instruction
            instruction_ref ret_ins = std::prev(mdl->end());
            return ret_ins->inputs();
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
