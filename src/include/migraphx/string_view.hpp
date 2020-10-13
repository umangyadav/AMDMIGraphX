#ifndef MIGRAPHX_GUARD_RTGLIB_STRING_VIEW_HPP
#define MIGRAPHX_GUARD_RTGLIB_STRING_VIEW_HPP

#include <migraphx/config.hpp>

#if defined(__has_include)
#if __has_include(<string_view>) && __cplusplus >= 201703L
#define MIGRAPHX_HAS_STRING_VIEW 1
#else
#define MIGRAPHX_HAS_STRING_VIEW 0
#endif
#if __has_include(<experimental/string_view>) && __cplusplus >= 201103L
#define MIGRAPHX_HAS_STRING_VIEW_TS 1
#else
#define MIGRAPHX_HAS_STRING_VIEW_TS 0
#endif
#else
#define MIGRAPHX_HAS_STRING_VIEW 0
#define MIGRAPHX_HAS_STRING_VIEW_TS 0
#endif

#if MIGRAPHX_HAS_STRING_VIEW
#include <string_view>
#elif MIGRAPHX_HAS_STRING_VIEW_TS
#include <experimental/string_view>
#else
#error "No string_view include available"
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#if MIGRAPHX_HAS_STRING_VIEW
using string_view = ::std::string_view;
#elif MIGRAPHX_HAS_STRING_VIEW_TS
using string_view = ::std::experimental::string_view;
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
