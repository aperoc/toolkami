#!/usr/bin/env -S PYTHONPATH=. uv run --script
# /// script
# dependencies = [ "mcp[cli]", "openai", "httpx", "anyio", "prompt_toolkit", "jsonpickle"]
# ///

import json
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack

from openai import OpenAI

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

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            k: v 
                            for k, v in tool.inputSchema.items()
                            if k not in ["additionalProperties", "$schema", "title"]
                        }
                    }
                }
                for tool in mcp_tools.tools
            ]

            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "user", "content": "list files"}
                ],
                tools=tools,
            )

            print(response)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())
