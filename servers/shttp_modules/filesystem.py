# server.py
from mcp.server.fastmcp import FastMCP

import re
import uuid
import shutil
import pathlib
from typing import Dict, Any, List, Optional

# Create an MCP server
mcp = FastMCP("Filesystem", stateless_http=True)

@mcp.tool()
async def read_file(path: str) -> str:
    """Read the contents of a file."""
    return pathlib.Path(path).read_text()

@mcp.tool()
async def diff_fenced_edit_file(diff_text: str) -> Dict[str, Any]:
    """Edit files using a diff-fenced format and return the status.

Basic Format Structure:
```diff
/filename.py
<<<<<<< SEARCH  
// original text that should be found and replaced  
=======  
// new text that will replace the original content  
>>>>>>> REPLACE  
```
"""
    pattern = r'```diff\n(.*?)\n<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE\n```'
    edit_blocks = re.findall(pattern, diff_text, re.DOTALL)
    
    blocks_edited = 0
    for block in edit_blocks:
        if len(block) == 3:
            file_path, search_text, replace_text = block

            search_text = search_text.strip()
            replace_text = replace_text.strip()

            content = pathlib.Path(file_path).read_text()
            original_content = content
            
            # Replace the search text with the replace text
            if search_text in content:
                content = content.replace(search_text, replace_text)
            else:
                # If search text not found and it's empty, append the replace text
                if not search_text.strip():
                    content += '\n' + replace_text

            # Only increment blocks_edited if content actually changed
            if content != original_content:
                pathlib.Path(file_path).write_text(content)
                blocks_edited += 1

    if blocks_edited == len(edit_blocks):
        return {"success": True, "blocks_edited": blocks_edited}
    
    return {"success": False, "blocks_edited": blocks_edited}