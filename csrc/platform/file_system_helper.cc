#include "platform/file_system_helper.h"

#include "platform/env.h"
#include "platform/filesystem.h"
#include "platform/path.h"
#include "platform/status.h"
#include "utils/str_util.h"
namespace recis {
namespace internal {
namespace {
void ForEach(int first, int last, const std::function<void(int)> &f) {
  for (int i = first; i < last; i++) {
    f(i);
  }
}
}  // namespace
Status GetMatchingPaths(FileSystem *fs, Env *env, const std::string &pattern,
                        std::vector<std::string> *results) {
  results->clear();
  // Find the fixed prefix by looking for the first wildcard.
  std::string fixed_prefix = pattern.substr(0, pattern.find_first_of("*?[\\"));
  std::string eval_pattern = pattern;
  std::vector<std::string> all_files;
  std::string dir(io::Dirname(fixed_prefix));
  // If dir is empty then we need to fix up fixed_prefix and eval_pattern to
  // include . as the top level directory.
  if (dir.empty()) {
    dir = ".";
    fixed_prefix = io::JoinPath(dir, fixed_prefix);
    eval_pattern = io::JoinPath(dir, pattern);
  }

  // Setup a BFS to explore everything under dir.
  std::deque<std::string> dir_q;
  dir_q.push_back(dir);
  Status ret;  // Status to return.
  // children_dir_status holds is_dir status for children. It can have three
  // possible values: OK for true; FAILED_PRECONDITION for false; CANCELLED
  // if we don't calculate IsDirectory (we might do that because there isn't
  // any point in exploring that child path).
  std::vector<Status> children_dir_status;
  while (!dir_q.empty()) {
    std::string current_dir = dir_q.front();
    dir_q.pop_front();
    std::vector<std::string> children;
    Status s = fs->GetChildren(current_dir, &children);
    ret.Update(s);
    if (children.empty()) continue;
    // This IsDirectory call can be expensive for some FS. Parallelizing it.
    children_dir_status.resize(children.size());
    ForEach(0, children.size(),
            [fs, &current_dir, &children, &fixed_prefix,
             &children_dir_status](int i) {
              const std::string child_path =
                  io::JoinPath(current_dir, children[i]);
              // In case the child_path doesn't start with the fixed_prefix then
              // we don't need to explore this path.
              if (!util::string::StartsWith(child_path, fixed_prefix)) {
                children_dir_status[i] =
                    Status(Code::CANCELLED, "Operation not needed");
              } else {
                children_dir_status[i] = fs->IsDirectory(child_path);
              }
            });
    for (int i = 0; i < children.size(); ++i) {
      const std::string child_path = io::JoinPath(current_dir, children[i]);
      // If the IsDirectory call was cancelled we bail.
      if (children_dir_status[i].code() == Code::CANCELLED) {
        continue;
      }
      // If the child is a directory add it to the queue.
      if (children_dir_status[i].ok()) {
        dir_q.push_back(child_path);
      }
      all_files.push_back(child_path);
    }
  }

  // Match all obtained files to the input pattern.
  for (const auto &f : all_files) {
    if (env->MatchPath(f, eval_pattern)) {
      results->push_back(f);
    }
  }
  return ret;
}
}  // namespace internal
}  // namespace recis