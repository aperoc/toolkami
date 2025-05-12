#!/usr/bin/env -S PYTHONPATH=. uv run --script
# /// script
# dependencies = [ "mcp[cli]", "openai", "httpx", "anyio", "prompt_toolkit", "jsonpickle"]
# ///

import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack

from dotenv import load_dotenv

load_dotenv()

async def main():
    async with AsyncExitStack() as exit_stack:
        try:
            # Create and enter SSE client context
            sse_streams = await exit_stack.enter_async_context(sse_client(url="http://localhost:8081/sse"))
            
            # Create and enter MCP session context
            mcp_session = await exit_stack.enter_async_context(ClientSession(*sse_streams))
            
            # Initialize the session
            await mcp_session.initialize()
            
            # List available tools
            mcp_tools = await mcp_session.list_tools()
            print(mcp_tools)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())
