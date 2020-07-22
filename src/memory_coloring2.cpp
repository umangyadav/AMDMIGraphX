#include <migraphx/memory_coloring2.hpp>
#include <migraphx/program.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using instruction_set     = std::unordered_set<instruction_ref>;
using instruction_set_map = std::unordered_map<instruction_ref, instruction_set>;

// This will do liveness analysis on the program, and it will call the
// function `f` with the instruction and the set of the other instructions
// that are live
template <class F>
void liveness(const program& p, F f)
{
    instruction_set live_set;
    auto rp = reverse(p);
    for(auto rins : iterator_for(rp))
    {
        // The base iterator is one ahead, so we need to use the previous iterator
        auto ins = std::prev(rins.base());
        // Add live variables
        for(auto input : ins->inputs())
        {
            auto i = instruction::get_output_alias(input);
            live_set.insert(i);
        }
        // Remove last usage
        auto it = live_set.find(ins);
        if(it != live_set.end())
        {
            f(ins, live_set);
            live_set.erase(it);
        }
    }
}

// This will build the conflict table or interference graph. This is
// essentially a map from one instruction to a set of instruction that are
// used together. Each instruction will be the allocation instruction.
instruction_set_map build_conflict_table(const program& p, std::string allocation_op)
{
    instruction_set_map conflict_table;
    liveness(p, [&](auto, auto live_set) {
        for(auto i : live_set)
        {
            // Skip variables that aren't allocations
            if(i->name() != allocation_op)
                continue;
            conflict_table[i];
            for(auto j : live_set)
            {
                // Skip variables that aren't allocations
                if(j->name() != allocation_op)
                    continue;
                if(i == j)
                    continue;
                // Add all variables that are used together to the
                // conflict_table
                conflict_table[i].insert(j);
                conflict_table[j].insert(i);
            }
        }
    });
    assert(std::all_of(conflict_table.begin(), conflict_table.end(), [](auto&& pp) {
        return pp.second.count(pp.first) == 0;
    }));
    return conflict_table;
}

// A class to manage allocation colors
struct allocation_color
{
    std::unordered_map<instruction_ref, int> ins2color;
    std::map<int, instruction_set> color2ins;

    std::size_t colors() const
    {
        // return color2ins.size();
        if(color2ins.empty())
            return 0;
        else
            return std::prev(color2ins.end())->first + 1;
    }

    std::size_t instructions(int color) const
    {
        auto it = color2ins.find(color);
        if(it == color2ins.end())
            return 0;
        else
            return it->second.size();
    }

    // Add a color for an instruction. Each color must be a positive integer.
    void add_color(instruction_ref ins, int color)
    {
        assert(color >= 0);
        this->remove(ins);
        ins2color[ins] = color;
        color2ins[color].insert(ins);
    }

    // Get the color for an instruction, if the instruction doesn't have a
    // color it will return a negative number.
    int get_color(instruction_ref ins) const
    {
        auto it = ins2color.find(ins);
        if(it == ins2color.end())
            return -1;
        return it->second;
    }

    // Remove color for an instruction
    void remove(instruction_ref ins)
    {
        auto it = ins2color.find(ins);
        if(it != ins2color.end())
        {
            color2ins[it->second].erase(ins);
            if(color2ins[it->second].empty())
                color2ins.erase(it->second);
            ins2color.erase(it);
        }
    }

    // Get the max amount of memory for a color
    std::size_t max_bytes(int color) const
    {
        auto&& is = color2ins.at(color);
        auto it   = std::max_element(is.begin(), is.end(), [](auto x, auto y) {
            return x->get_shape().bytes() < y->get_shape().bytes();
        });
        if(it == is.end())
            return 0;
        else
            return (*it)->get_shape().bytes();
    }

    // Insert next available color in the set
    static int next_color(std::set<int>& colors)
    {
        auto start = colors.find(0);
        if(start == colors.end())
        {
            colors.insert(0);
            return 0;
        }
        auto it =
            std::adjacent_find(start, colors.end(), [](int x, int y) { return (x + 1) != y; });
        auto last = (it == colors.end()) ? std::prev(it) : it;
        // Compute the next color available
        auto n = *last + 1;
        assert(colors.count(n) == 0);
        colors.insert(n);
        return n;
    }

    // Build the allocation_color class from the conflict_table
    static allocation_color build(const instruction_set_map& conflict_table)
    {
        allocation_color ac{};
        std::vector<instruction_ref> conflict_queue;
        // Add all allocations to the conflict_queue
        std::transform(conflict_table.begin(),
                       conflict_table.end(),
                       std::back_inserter(conflict_queue),
                       [](auto&& pp) { return pp.first; });

        // Sort the conflict queue so we process the allocation with the least
        // number of adjacent allocations first
        std::sort(conflict_queue.begin(), conflict_queue.end(), [&](auto x, auto y) {
            return std::make_tuple(conflict_table.at(x).size(), x->get_shape().bytes()) <
                   std::make_tuple(conflict_table.at(y).size(), y->get_shape().bytes());
        });
        // Process the conflict_queue, we refer to the current allocation as
        // the parent and the adjacent allocations as children
        for(auto parent : conflict_queue)
        {
            // Sort children by size
            std::vector<instruction_ref> children(conflict_table.at(parent).begin(),
                                                  conflict_table.at(parent).end());
            std::sort(children.begin(), children.end(), [](auto x, auto y) {
                return x->get_shape().bytes() < y->get_shape().bytes();
            });
            // This set is to track the colors already processed
            std::set<int> colors;
            // Add all colors for the children to the colors already processed
            std::transform(children.begin(),
                           children.end(),
                           std::inserter(colors, colors.begin()),
                           [&](auto child) { return ac.get_color(child); });
            // Get the color for the parent
            auto parent_color = ac.get_color(parent);
            // Color the parent if hasn't been colored or the color is already used by the children
            if(parent_color < 0 or colors.count(parent_color) > 0)
            {
                // Get next available color
                parent_color = next_color(colors);
                ac.add_color(parent, parent_color);
            }
            else
            {
                colors.insert(parent_color);
            }
            for(auto child : children)
            {
                assert(child != parent);
                auto color = ac.get_color(child);
                if(color < 0)
                {
                    // Get next available color
                    color = next_color(colors);
                    ac.add_color(child, color);
                }
            }
        }
        // Reduce the number of colors
        for(auto parent : conflict_queue)
        {
            auto children = conflict_table.at(parent);
            // This set is to track the colors already processed
            std::set<int> colors;
            // Add all colors for the children to the colors already processed
            std::transform(children.begin(),
                           children.end(),
                           std::inserter(colors, colors.begin()),
                           [&](auto child) { return ac.get_color(child); });
            // Get the color for the parent
            auto parent_color = ac.get_color(parent);
            colors.insert(parent_color);
            assert(parent_color != -1);

            std::vector<int> next_colors;
            auto c = next_color(colors);
            while(c < ac.colors())
            {
                if(ac.instructions(c) > 0)
                    next_colors.push_back(c);
                c = next_color(colors);
            }

            std::sort(next_colors.begin(), next_colors.end(), [&](int x, int y) {
                return ac.max_bytes(x) < ac.max_bytes(y);
            });

            for(auto color : next_colors)
            {
                auto bytes = ac.max_bytes(color);
                if(bytes >= parent->get_shape().bytes() or ac.instructions(parent_color) == 1 or
                   ac.instructions(color) == 1)
                {
                    ac.add_color(parent, color);
                    break;
                }
            }
        }
        return ac;
    }
};

void memory_coloring2::apply(program& p) const
{
    auto conflict_table = build_conflict_table(p, allocation_op);
    auto ac             = allocation_color::build(conflict_table);

    // All allocations should have a color
    assert(std::all_of(conflict_table.begin(), conflict_table.end(), [&](auto&& pp) {
        return ac.get_color(pp.first) >= 0;
    }));

    // Adjacent allocations should not share the same color
    assert(std::none_of(conflict_table.begin(), conflict_table.end(), [&](auto&& pp) {
        auto c = ac.get_color(pp.first);
        return std::any_of(
            pp.second.begin(), pp.second.end(), [&](auto ins) { return ac.get_color(ins) == c; });
    }));

    const std::size_t alignment = 32;
    // Total memory
    std::size_t n = 0;
    // Compute offsets
    std::map<int, int> color2offset;
    for(auto&& pp : ac.color2ins)
    {
        auto color = pp.first;
        color2offset.emplace(color, n);
        std::size_t size    = ac.max_bytes(color);
        std::size_t padding = (alignment - (size % alignment)) % alignment;
        n += size + padding;
    }

    // Replace allocations
    auto mem = p.add_parameter("scratch", shape{shape::int8_type, {n}});
    for(auto&& pp : ac.color2ins)
    {
        auto color         = pp.first;
        auto&& allocations = pp.second;
        auto offset        = color2offset.at(color);
        for(auto ins : allocations)
        {
            assert(ins->name() == allocation_op);
            auto s = ins->get_shape();
            p.replace_instruction(ins, op::load{s, std::size_t(offset)}, mem);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
