#include "platform/path.h"

#include "utils/scanner.h"
#include "utils/str_util.h"
namespace recis {
namespace io {
namespace internal {

std::string JoinPathImpl(std::initializer_list<torch::string_view> paths) {
  std::string result;

  for (torch::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = std::string(path);
      continue;
    }

    if (result[result.size() - 1] == '/') {
      if (IsAbsolutePath(path)) {
        util::string::StringAppend(result, path.substr(1));
      } else {
        util::string::StringAppend(result, path);
      }
    } else {
      if (IsAbsolutePath(path)) {
        util::string::StringAppend(result, path);
      } else {
        util::string::StringAppend(result, "/");
        util::string::StringAppend(result, path);
      }
    }
  }

  return result;
}

// Return the parts of the URI, split on the final "/" in the path. If there is
// no "/" in the path, the first part of the output is the scheme and host, and
// the second is the path. If the only "/" in the path is the first character,
// it is included in the first part of the output.
std::pair<torch::string_view, torch::string_view> SplitPath(
    torch::string_view uri) {
  torch::string_view scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);

  auto pos = path.rfind('/');
#ifdef PLATFORM_WINDOWS
  if (pos == torch::string_view::npos) pos = path.rfind('\\');
#endif
  // Handle the case with no '/' in 'path'.
  if (pos == torch::string_view::npos)
    return std::make_pair(
        torch::string_view(uri.begin(), host.end() - uri.begin()), path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(
        torch::string_view(uri.begin(), path.begin() + 1 - uri.begin()),
        torch::string_view(path.data() + 1, path.size() - 1));

  return std::make_pair(
      torch::string_view(uri.begin(), path.begin() + pos - uri.begin()),
      torch::string_view(path.data() + pos + 1, path.size() - (pos + 1)));
}

// Return the parts of the basename of path, split on the final ".".
// If there is no "." in the basename or "." is the final character in the
// basename, the second value will be empty.
std::pair<torch::string_view, torch::string_view> SplitBasename(
    torch::string_view path) {
  path = Basename(path);

  auto pos = path.rfind('.');
  if (pos == torch::string_view::npos)
    return std::make_pair(path,
                          torch::string_view(path.data() + path.size(), 0));
  return std::make_pair(
      torch::string_view(path.data(), pos),
      torch::string_view(path.data() + pos + 1, path.size() - (pos + 1)));
}
}  // namespace internal

bool IsAbsolutePath(torch::string_view path) {
  return !path.empty() && path[0] == '/';
}

torch::string_view Dirname(torch::string_view path) {
  return internal::SplitPath(path).first;
}

torch::string_view Basename(torch::string_view path) {
  return internal::SplitPath(path).second;
}

torch::string_view Extension(torch::string_view path) {
  return internal::SplitBasename(path).second;
}

std::string CleanPath(torch::string_view unclean_path) {
  std::string path(unclean_path);
  const char *src = path.c_str();
  std::string::iterator dst = path.begin();

  // Check for absolute path and determine initial backtrack limit.
  const bool is_absolute_path = *src == '/';
  if (is_absolute_path) {
    *dst++ = *src++;
    while (*src == '/') ++src;
  }
  std::string::const_iterator backtrack_limit = dst;

  // Process all parts
  while (*src) {
    bool parsed = false;

    if (src[0] == '.') {
      //  1dot ".<whateverisnext>", check for END or SEP.
      if (src[1] == '/' || !src[1]) {
        if (*++src) {
          ++src;
        }
        parsed = true;
      } else if (src[1] == '.' && (src[2] == '/' || !src[2])) {
        // 2dot END or SEP (".." | "../<whateverisnext>").
        src += 2;
        if (dst != backtrack_limit) {
          // We can backtrack the previous part
          for (--dst; dst != backtrack_limit && dst[-1] != '/'; --dst) {
            // Empty.
          }
        } else if (!is_absolute_path) {
          // Failed to backtrack and we can't skip it either. Rewind and copy.
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) {
            *dst++ = *src;
          }
          // We can never backtrack over a copied "../" part so set new limit.
          backtrack_limit = dst;
        }
        if (*src) {
          ++src;
        }
        parsed = true;
      }
    }

    // If not parsed, copy entire part until the next SEP or EOS.
    if (!parsed) {
      while (*src && *src != '/') {
        *dst++ = *src++;
      }
      if (*src) {
        *dst++ = *src++;
      }
    }

    // Skip consecutive SEP occurrences
    while (*src == '/') {
      ++src;
    }
  }

  // Calculate and check the length of the cleaned path.
  std::string::difference_type path_length = dst - path.begin();
  if (path_length != 0) {
    // Remove trailing '/' except if it is root path ("/" ==> path_length := 1)
    if (path_length > 1 && path[path_length - 1] == '/') {
      --path_length;
    }
    path.resize(path_length);
  } else {
    // The cleaned path is empty; assign "." as per the spec.
    path.assign(1, '.');
  }
  return path;
}

void ParseURI(torch::string_view remaining, torch::string_view *scheme,
              torch::string_view *host, torch::string_view *path) {
  // 0. Parse scheme
  // Make sure scheme matches [a-zA-Z][0-9a-zA-Z.]*
  // TODO(keveman): Allow "+" and "-" in the scheme.
  // Keep URI pattern in tensorboard/backend/server.py updated accordingly
  if (!util::string::Scanner(remaining)
           .One(util::string::Scanner::LETTER)
           .Many(util::string::Scanner::LETTER_DIGIT_DOT)
           .StopCapture()
           .OneLiteral("://")
           .GetResult(&remaining, scheme)) {
    // If there's no scheme, assume the entire string is a path.
    *scheme = torch::string_view(remaining.begin(), 0);
    *host = torch::string_view(remaining.begin(), 0);
    *path = remaining;
    return;
  }

  // 1. Parse host
  if (!util::string::Scanner(remaining).ScanUntil('/').GetResult(&remaining,
                                                                 host)) {
    // No path, so the rest of the URI is the host.
    *host = remaining;
    *path = torch::string_view(remaining.end(), 0);
    return;
  }

  // 2. The rest is the path
  *path = remaining;
}

std::string CreateURI(torch::string_view scheme, torch::string_view host,
                      torch::string_view path) {
  if (scheme.empty()) {
    return std::string(path);
  }
  return torch::str(scheme, "://", host, path);
}
}  // namespace io
}  // namespace recis