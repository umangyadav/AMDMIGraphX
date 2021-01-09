#include <migraphx/memory_coloring.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
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

// This will do liveness analysis on the module, and it will call the
// function `f` with the instruction and the set of the other instructions
// that are live
template <class F>
void liveness(const module& m, F f)
{
    instruction_set live_set;
    auto rp = reverse(m);
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
instruction_set_map build_conflict_table(const module& m, std::string allocation_op)
{
    instruction_set_map conflict_table;
    liveness(m, [&](auto, auto live_set) {
        for(auto i : live_set)
        {
            // Skip variables that aren't allocations
            if(i->name() != allocation_op)
                continue;
            // Skip zero allocations
            if (i->get_shape().bytes() == 0)
                continue;
            conflict_table[i];
            for(auto j : live_set)
            {
                // Skip variables that aren't allocations
                if(j->name() != allocation_op)
                    continue;
                // Skip zero allocations
                if (j->get_shape().bytes() == 0)
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

// Check if intervals overlap
bool is_overlap(std::pair<std::size_t, std::size_t> x, std::pair<std::size_t, std::size_t> y)
{
    return std::max(x.first, y.first) < std::min(x.second, y.second);
}

const std::size_t alignment = 32;

struct allocation_segment
{
    using segment = std::pair<std::size_t, std::size_t>;
    std::unordered_map<instruction_ref, segment> ins2segment;

    const segment* add_segment(instruction_ref ins, segment s)
    {
        // this->remove(ins);
        return &(ins2segment[ins] = s);
    }

    const segment* get_segment(instruction_ref ins) const
    {
        auto it = ins2segment.find(ins);
        if(it == ins2segment.end())
            return nullptr;
        return &it->second;
    }

    // Remove segment for an instruction
    void remove(instruction_ref ins)
    {
        auto it = ins2segment.find(ins);
        if(it != ins2segment.end())
        {
            ins2segment.erase(it);
        }
    }

    static bool overlaps(const std::set<segment>& segments, const segment& s)
    {
        auto it = std::find_if(
            segments.begin(), segments.end(), [&](auto&& t) { return is_overlap(s, t); });
        return it != segments.end();
    }

    static segment next_segment(std::set<segment>& segments, instruction_ref ins)
    {
        assert(ins->get_shape().bytes() > 0);
        auto n = 1 + (ins->get_shape().bytes() - 1) / alignment;
        assert(n > 0);
        auto start = 0;
        // Insert at end if it can fit at the begining
        if(segments.empty() or segments.begin()->first <= n)
        {
            auto it =
                std::adjacent_find(segments.begin(), segments.end(), [&](segment x, segment y) {
                    if(x.second >= y.first)
                        return false;
                    auto k = y.first - x.second;
                    return (k >= n);
                });
            if(it == segments.end() and not segments.empty())
                it--;
            if(it != segments.end())
                start = it->second;
        }
        auto s = segment{start, start + n};
        assert(not overlaps(segments, s));
        segments.insert(s);
        return s;
    }

    // Build the allocation_color class from the conflict_table
    static allocation_segment build(const instruction_set_map& conflict_table)
    {
        allocation_segment as{};
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
            // This set is to track the segments already processed
            std::set<segment> segments;
            // Add all segemnts for the children to the segments already processed
            transform_if(children.begin(),
                         children.end(),
                         std::inserter(segments, segments.begin()),
                         [&](auto child) { return as.get_segment(child); },
                         [&](auto child) { return *as.get_segment(child); });

            // Get the segement for the parent
            auto* parent_segment = as.get_segment(parent);
            // Add segment for the parent if there is none or segment overlaps with the children
            if(parent_segment == nullptr or overlaps(segments, *parent_segment))
                as.add_segment(parent, next_segment(segments, parent));
            else
                segments.insert(*parent_segment);

            for(auto child : children)
            {
                assert(child != parent);
                if(not as.get_segment(child))
                    as.add_segment(child, next_segment(segments, child));
            }
        }
        return as;
    }
};

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

void memory_coloring::apply(module& m) const
{
    auto conflict_table = build_conflict_table(m, allocation_op);
    auto as             = allocation_segment::build(conflict_table);

    // All allocations should have a segment
    assert(std::all_of(conflict_table.begin(), conflict_table.end(), [&](auto&& pp) {
        return as.get_segment(pp.first);
    }));

    // Adjacent allocations should not have overlapping segments
    assert(std::none_of(conflict_table.begin(), conflict_table.end(), [&](auto&& pp) {
        auto* x = as.get_segment(pp.first);
        return std::any_of(pp.second.begin(), pp.second.end(), [&](auto ins) {
            auto* y = as.get_segment(ins);
            assert(x and y);
            return is_overlap(*x, *y);
        });
    }));

    // Total memory
    std::size_t n = 0;
    for(auto&& pp : as.ins2segment)
    {
        auto seg = pp.second;
        n        = std::max(n, seg.second * alignment);
    }

    // Replace allocations
    auto mem = m.add_parameter("scratch", shape{shape::int8_type, {n}});
    for(auto&& pp : as.ins2segment)
    {
        auto ins = pp.first;
        auto seg = pp.second;
        assert(ins->name() == allocation_op);
        auto s             = ins->get_shape();
        std::size_t offset = seg.first * 32;
        m.replace_instruction(ins, op::load{s, offset}, mem);
    }

    // Replace zero allocation
    for(auto ins: iterator_for(m))
    {
        if (ins->name() != allocation_op)
            continue;
        assert(ins->get_shape().bytes() == 0);
        m.replace_instruction(ins, op::load{ins->get_shape(), 0}, mem);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
