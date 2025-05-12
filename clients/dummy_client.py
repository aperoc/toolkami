#!/usr/bin/env -S PYTHONPATH=. uv run --script
# /// script
# dependencies = [ "mcp[cli]", "openai", "httpx", "anyio", "prompt_toolkit", "jsonpickle"]
# ///

import os
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack

from openai import OpenAI

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
print(f"Loading .env file from: {env_path}")
load_dotenv(env_path, override=True)  # Added override=True to ensure values are set

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

            # client = OpenAI()
            # Gemini
            # client = OpenAI(
            #     api_key=os.getenv("GEMINI_API_KEY"),
            #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            # )
            # Vertex

            base_url = f"https://{os.environ['GOOGLE_VERTEX_LOCATION']}-aiplatform.googleapis.com/v1beta1/projects/{os.environ['GOOGLE_VERTEX_PROJECT']}/locations/{os.environ['GOOGLE_VERTEX_LOCATION']}/endpoints/openapi"
            client = OpenAI(
                base_url=base_url
            )

            response = client.chat.completions.create(
                # model="gpt-4.1",
                # model="gemini-2.0-flash",
                model=f"google/{os.environ['GEMINI_MODEL']}",
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
