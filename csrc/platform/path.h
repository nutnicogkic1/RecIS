/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once
#include <torch/extension.h>

#include <string>

#include "c10/util/string_view.h"
namespace recis {
namespace io {
namespace internal {
std::string JoinPathImpl(std::initializer_list<torch::string_view> paths);
}

template <typename... T>
std::string JoinPath(const T &...args) {
  return internal::JoinPathImpl({args...});
}

// Return true if path is absolute.
bool IsAbsolutePath(torch::string_view);

// Returns the part of the path before the final "/".  If there is a single
// leading "/" in the path, the result will be the leading "/".  If there is
// no "/" in the path, the result is the empty prefix of the input.
torch::string_view Dirname(torch::string_view path);

// Returns the part of the path after the final "/".  If there is no
// "/" in the path, the result is the same as the input.
torch::string_view Basename(torch::string_view path);

// Returns the part of the basename of path after the final ".".  If
// there is no "." in the basename, the result is empty.
torch::string_view Extension(torch::string_view path);

// Collapse duplicate "/"s, resolve ".." and "." path elements, remove
// trailing "/".
//
// NOTE: This respects relative vs. absolute paths, but does not
// invoke any system calls (getcwd(2)) in order to resolve relative
// paths with respect to the actual working directory.  That is, this is purely
// string manipulation, completely independent of process state.
std::string CleanPath(torch::string_view path);

// Populates the scheme, host, and path from a URI. scheme, host, and path are
// guaranteed by this function to point into the contents of uri, even if
// empty.
//
// Corner cases:
// - If the URI is invalid, scheme and host are set to empty strings and the
//   passed string is assumed to be a path
// - If the URI omits the path (e.g. file://host), then the path is left empty.
void ParseURI(torch::string_view uri, torch::string_view *scheme,
              torch::string_view *host, torch::string_view *path);

// Creates a URI from a scheme, host, and path. If the scheme is empty, we just
// return the path.
std::string CreateURI(torch::string_view scheme, torch::string_view host,
                      torch::string_view path);

// Creates a temporary file name with an extension.
std::string GetTempFilename(const std::string &extension);

}  // namespace io
}  // namespace recis