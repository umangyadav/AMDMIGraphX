#ifndef MIGRAPHX_GUARD_RTGLIB_ALGORITHM_HPP
#define MIGRAPHX_GUARD_RTGLIB_ALGORITHM_HPP

#include <algorithm>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Iterator, class Output, class Predicate>
void group_by(Iterator start, Iterator last, Output out, Predicate pred)
{
    while(start != last)
    {
        auto it = std::partition(start, last, [&](auto&& x) { return pred(x, *start); });
        out(start, it);
        start = it;
    }
}

template <class Iterator, class Output, class Predicate>
void group_unique(Iterator start, Iterator last, Output out, Predicate pred)
{
    while(start != last)
    {
        auto it = std::find_if(start, last, [&](auto&& x) { return not pred(*start, x); });
        out(start, it);
        start = it;
    }
}

template <class Iterator, class OutputIterator, class Predicate, class F>
OutputIterator transform_if(Iterator start, Iterator last, OutputIterator out, Predicate pred, F f)
{
    while(start != last)
    {
        if(pred(*start))
            *out++ = f(*start);
        ++start;
    }
    return out;
}

template<class Iterator>
void kth_permutation(std::size_t k, Iterator start, Iterator last)
{
    std::size_t size = std::distance(start, last);
    for(int j = 1; j < size; ++j) 
    {
        auto x = start + (k % (j + 1));
        auto y = start + j;
        std::iter_swap(x, y); 
        k = k / (j + 1);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
