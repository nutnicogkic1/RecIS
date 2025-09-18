"""Filesystem abstraction layer for handling different storage backends.

This module provides a unified interface for accessing different filesystem types
including local filesystem and distributed filesystem (DFS). It uses fsspec
as the underlying abstraction layer and automatically selects the appropriate
filesystem based on the path protocol.
"""

from pathlib import Path

import fsspec
from fsspec import AbstractFileSystem

from recis.info import is_internal_enabled


# Global filesystem mapper for different protocols
_file_system_mapper = {
    "file": fsspec.filesystem("file"),
}

# Add DFS support if internal features are enabled
if is_internal_enabled():
    _file_system_mapper["dfs"] = fsspec.filesystem("dfs")


def get_file_system(path) -> AbstractFileSystem:
    """Get the appropriate filesystem handler for the given path.

    This function automatically detects the filesystem type based on the path
    protocol and returns the corresponding fsspec filesystem instance.

    Args:
        path (Union[str, Path]): The path to determine filesystem for.
                                Can be a local path or a path with protocol prefix.

    Returns:
        AbstractFileSystem: The fsspec filesystem instance for the given path.

    Raises:
        NotImplementedError: If the protocol in the path is not supported.

    Examples:
        >>> # Local filesystem
        >>> fs = get_file_system("/local/path/to/file")
        >>> # DFS filesystem (if internal features enabled)
        >>> fs = get_file_system("dfs://cluster/path/to/file")
        >>> # Using Path object
        >>> from pathlib import Path
        >>> fs = get_file_system(Path("/local/path"))
    """
    global _file_system_mapper
    if isinstance(path, Path):
        path = str(path)
    if "://" in path:
        protocol = path.split("://")[0]
        if protocol in _file_system_mapper:
            return _file_system_mapper.get(protocol)
        else:
            raise NotImplementedError(
                f"fsspec has not implement {protocol} with {path}"
            )
    else:
        return _file_system_mapper.get("file")
