#pragma once

#include <string_view>

namespace phc {

// Bytes of web/live/index.html, embedded at build time. See
// cpp/cmake/embed_html.cmake for the generator.
extern const std::string_view kEmbeddedIndexHtml;

}  // namespace phc
