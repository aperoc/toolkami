import re
import pathlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
import logging

from mcp.server.fastmcp import FastMCP
from . import MCPModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FilesystemModule:

    def __init__(self):
        self._mcp = FastMCP("filesystem")
        self._mcp_server = self._mcp._mcp_server
        print(f"Debug: FilesystemModule initialized")

    def register_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register command line arguments for the filesystem module."""
        parser.add_argument(
            '--directories', 
            nargs='*',
            help='Allowed directories to access (for filesystem tools)'
        )

    def initialize(self, args: argparse.Namespace = None) -> None:
        """Initialize the filesystem module with command line arguments.
        
        Args:
            args: Command line arguments namespace. Uses:
                - directories: List of allowed directory paths
        """
        print(f"Debug: Initializing FilesystemModule with args: {args}")
        self.allowed_directories = args.directories
        
    def register_tools(self, mcp: FastMCP) -> None:
        """Register the filesystem tools with the given MCP instance."""
        @mcp.tool()
        async def read_file(path: str) -> str:
            """Read the contents of a file."""
            return pathlib.Path(path).read_text()

        @mcp.tool()
        async def write_file(path: str, content: str) -> Dict[str, Any]:
            """Write content to a file.
            
            Returns:
                Dict containing:
                    - success: bool indicating if write was successful
                    - path: str containing the file path that was written to
            """

            # Only allow writing to allowed directories
            if not any(path.startswith(allowed_dir) for allowed_dir in self.allowed_directories):
                raise ValueError(f"Writing to {path} is not allowed. Allowed directories: {self.allowed_directories}")

            pathlib.Path(path).write_text(content)
            return {"success": True, "path": path}

        @mcp.tool()
        async def diff_fenced_edit_file(original_text: str, diff_text: str) -> str:
            """Edit files using a diff-fenced format and return the edited content."""
            # Extract edit blocks from the diff-fenced format
            pattern = r'```diff\n(.*?)\n<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE\n```'
            edit_blocks = re.findall(pattern, diff_text, re.DOTALL)
            
            if not edit_blocks:
                # Try without the code fences for direct input
                pattern = r'(.*?)\n<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE'
                edit_blocks = re.findall(pattern, diff_text, re.DOTALL)
                
            if not edit_blocks:
                return original_text
            
            # Apply each edit block
            result = original_text
            for block in edit_blocks:
                if len(block) == 3:
                    filename, search_text, replace_text = block
                    # Strip any leading/trailing whitespace
                    search_text = search_text.strip()
                    replace_text = replace_text.strip()
                    
                    # Replace the search text with the replace text
                    if search_text in result:
                        result = result.replace(search_text, replace_text)
                    else:
                        # If search text not found and it's empty, append the replace text
                        if not search_text.strip():
                            result += '\n' + replace_text
            
            return result
            
        @mcp.tool()
        async def list_directory(path: str, recursive: bool = False, sort_order: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
            """List contents of a directory.
            
            Args:
                path: The directory path to list
                recursive: If True, recursively list all subdirectories and their contents
                sort_order: Optional sort order for filenames ('asc' or 'desc'). If None, no sorting is applied
                limit: Optional maximum number of entries to return. If None, all entries are returned
            """
            entries = []
            base_path = pathlib.Path(path)
            
            if recursive:
                # Use rglob to recursively list all files and directories
                for entry in base_path.rglob("*"):
                    # Skip the base directory itself
                    if entry == base_path:
                        continue
                    entries.append({
                        "path": str(entry),
                        "type": "file" if entry.is_file() else "directory",
                    })
            else:
                # Original behavior for non-recursive listing
                for entry in base_path.iterdir():
                    entries.append({
                        "path": str(entry),
                        "type": "file" if entry.is_file() else "directory",
                    })
            
            # Apply sorting if requested
            if sort_order:
                if sort_order not in ['asc', 'desc']:
                    raise ValueError("sort_order must be either 'asc' or 'desc'")
                # Sort by filename (last part of the path)
                entries.sort(key=lambda x: pathlib.Path(x["path"]).name, reverse=(sort_order == 'desc'))
            
            # Apply limit if requested
            if limit is not None:
                entries = entries[:limit]
            
            return entries


# Create a singleton instance of the module
filesystem_module = FilesystemModule() 