#include <vector>

#include "platform/status.h"
#pragma once
namespace recis {
class FileSystem;
class Env;

namespace internal {

// Given a pattern, stores in 'results' the set of paths (in the given file
// system) that match that pattern.
//
// This helper may be used by implementations of FileSystem::GetMatchingPaths()
// in order to provide parallel scanning of subdirectories (except on iOS).
//
// Arguments:
//   fs: may not be null and will be used to identify directories and list
//       their contents.
//   env: may not be null and will be used to check if a match has been found.
//   pattern: see FileSystem::GetMatchingPaths() for details.
//   results: will be cleared and may not be null.
//
// Returns an error status if any call to 'fs' failed.
Status GetMatchingPaths(FileSystem *fs, Env *env, const std::string &pattern,
                        std::vector<std::string> *results);

}  // namespace internal
}  // namespace recis