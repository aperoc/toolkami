#!/usr/bin/env -S PYTHONPATH=. uv run --script
# /// script
# dependencies = [ "mcp[cli]", "google-genai", "httpx", "anyio", "prompt_toolkit", "jsonpickle"]
# ///

import asyncio
import logging
import os
import sys
import functools
from typing import Optional, Callable, Awaitable, TypeVar, List, Dict, Any, Tuple
from contextlib import AsyncExitStack
import argparse

import httpx
import anyio

# Import prompt_toolkit
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from google import genai
from google.genai import types # Keep this

from mcp import ClientSession
from mcp.client.sse import sse_client

from agent import Agent
from dotenv import load_dotenv

load_dotenv()

# Define prompt_toolkit styles dictionary
PROMPT_STYLE_DICT = {
    "prompt": "fg:yellow",
    "output.model": "fg:green",
    "output.tool": "fg:blue",
    "output.error": "fg:red",
    "output.warning": "fg:yellow",
    "output.debug": "fg:gray",
}

# Create Style object from the dictionary
PROMPT_STYLE_OBJ = Style.from_dict(PROMPT_STYLE_DICT)

# Custom logging handler integrating with prompt_toolkit
class PromptToolkitLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            log_entry = self.format(record)
            style_class = "output.debug"  # Default

            # Check if it's a captured warning and matches the specific uv cache path
            if record.name == 'py.warnings' and record.levelno == logging.WARNING and '/root/.cache/uv/' in record.getMessage():
                style_class = "output.warning"
            elif record.levelno >= logging.ERROR:
                style_class = "output.error"
            elif record.levelno >= logging.WARNING:
                 # Use the standard warning style (yellow) for other warnings
                 style_class = "output.warning"
            elif record.levelno >= logging.INFO:
                # Use a less prominent style for INFO
                style_class = "output.debug"

            # Ensure we only print if there's actual content
            if log_entry.strip():
                print_pt(log_entry.strip(), style_class=style_class)
        except Exception:
            self.handleError(record)

# Helper function to set up logging
def setup_logging(debug: bool = False):
    # Capture warnings issued by the warnings module
    logging.captureWarnings(True)
    
    root_logger = logging.getLogger()
    # Remove default handlers like StreamHandler to avoid duplicate output
    # or output going to the original stderr
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our custom handler
    pt_handler = PromptToolkitLogHandler()
    # Basic formatter, showing level, logger name, and message
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    pt_handler.setFormatter(formatter)
    root_logger.addHandler(pt_handler)
    # Set logging level based on debug flag
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # Specifically set httpx/anyio levels if they are too noisy later
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    # logging.getLogger("anyio").setLevel(logging.WARNING)

# Helper to print formatted text using prompt_toolkit
def print_pt(text: str, style_class: str = ""):
    if style_class:
        print_formatted_text(FormattedText([(f"class:{style_class}", text)]), style=PROMPT_STYLE_OBJ)
    else:
        # Print with default style if no class specified
        print_formatted_text(text)

# Decorator for retryable async functions
def retryable(max_retries=5, delay=1, connection_errors=(httpx.ReadError, httpx.WriteError, 
                                                     httpx.RemoteProtocolError, httpx.ConnectError, 
                                                     anyio.ClosedResourceError, ConnectionError)):
    """
    Decorator for making async functions automatically retry on connection errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        connection_errors: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function that will retry on connection errors
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            operation_name = func.__name__
            retries = 0
            
            while True:
                try:
                    return await func(self, *args, **kwargs)
                except connection_errors as e:
                    retries += 1
                    if retries >= max_retries:
                        error_message = f"{operation_name} failed after {retries} attempts: {e}"
                        print_pt(error_message, "output.error")
                        return error_message
                    
                    print_pt(f"Connection error during {operation_name}: {e}. Attempting reconnect... ({retries}/{max_retries})", "output.error")
                    if hasattr(self, 'connect') and await self.connect():
                        print_pt(f"Reconnected. Retrying {operation_name}...", "output.debug")
                    else:
                        error_message = f"Reconnect failed for {operation_name}"
                        print_pt(error_message, "output.error")
                        return error_message
                        
                    await asyncio.sleep(delay)
                except Exception as error:
                    error_message = f"Error processing {operation_name}: {error}"
                    print_pt(error_message, "output.error")
                    return error_message
                    
        return wrapper
    return decorator

def truncate_text_both_ends(text: str, max_length: int = 250):
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length//2] + "..." + text[-max_length//2:]

class MCPClient:
    def __init__(self, server_url: str):
        self.exit_stack = AsyncExitStack() # Use one stack for the lifetime
        self._stop_event = asyncio.Event() # Event to signal shutdown

        self.server_url = server_url # Store the server URL for reconnection
        self._sse_stream_context = None
        self.sse_stream = None

        self._mcp_session_context = None
        self.mcp_session: Optional[ClientSession] = None

        self.prompt_session = PromptSession(history=None)

        if os.getenv("GEMINI_API_KEY"):
            self.provider = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )
        else:
            self.provider = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_VERTEX_PROJECT"),
                location=os.getenv("GOOGLE_VERTEX_LOCATION"),
            )

        self.agent = Agent()

    async def _connect_internal(self):
        """Internal logic to establish a connection."""
        await self.cleanup()
        self.exit_stack = AsyncExitStack()

        self._sse_stream_context = sse_client(url=self.server_url)
        self.sse_stream = await self.exit_stack.enter_async_context(self._sse_stream_context)

        self._mcp_session_context = ClientSession(*self.sse_stream)
        self.mcp_session: ClientSession = await self.exit_stack.enter_async_context(self._mcp_session_context)

        await self.mcp_session.initialize()
        print_pt(f"[DEBUG] Initialized SSE and MCP sessions...", "output.debug")

    @retryable(max_retries=5, delay=1)
    async def connect(self):
        """Attempts to connect to the server with retries."""
        try:
            await self._connect_internal()
            print_pt(f"[DEBUG] Successfully connected to server.", "output.debug")
            return True
        except Exception as e:
            print_pt(f"Connection error: {e}", "output.error")
            raise

    async def cleanup(self):
        """Properly clean up the session, streams, and background task."""
        print_pt(f"[DEBUG] Initiating client cleanup...", "output.debug")
        self._stop_event.set()
        await self.exit_stack.aclose()
        self._mcp_session_context = None
        self.mcp_session = None
        self._sse_stream_context = None
        self.sse_stream = None
        print_pt(f"[DEBUG] Client cleanup complete.", "output.debug")

    async def inlined_process_query_recursive(self, query: str):
        if query == "":
            print("No query provided.")
            return

        mcp_tools = await self.mcp_session.list_tools()

        tools =[
            types.Tool(
                function_declarations=[{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        k: v 
                        for k, v in tool.inputSchema.items()
                        if k not in ["additionalProperties", "$schema"]
                    },
                }]
            )
            for tool in mcp_tools.tools
        ]

        self.agent.add_content(
            types.Content(
                role="user", 
                parts=[types.Part(text=query)]
            )
        )

        while True:
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                print_pt(f"[DEBUG] Gemini Content History: {self.agent.content_history}", "output.debug")

            try:
                response = self.provider.models.generate_content(
                    model=os.getenv("MAIN_MODEL"),
                    contents=self.agent.content_history,
                    config=types.GenerateContentConfig(
                        temperature=0.1, # Slightly increase temperature to encourage more relevant responses while maintaing some consistency
                        tools=tools,
                        system_instruction=[self.agent.system_instruction],
                    ),
                )
            except Exception as e:
                print_pt(f"[ERROR] Error generating content: {e}", "output.error")
                print_pt(str(self.agent.content_history), "output.error")

                if e.error.code == 400:
                    raise Exception("Token limit exceeded. Forgetting history?")

                raise

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                print_pt(f"[DEBUG] Gemini response: {response}", "output.debug")

            print_pt(f"[WARNING] Token usage: {response.usage_metadata.total_token_count} / 1,047,576", "output.warning")

            candidate = response.candidates[0]

            if not candidate.content:
                print_pt("[ERROR] No content received from Gemini API", "output.error")
                print_pt(str(response), "output.error")

            if not candidate.content.parts:
                print_pt("[ERROR] No parts received from Gemini API", "output.error")
                print_pt(str(response), "output.error")
                print_pt(str(self.agent.content_history), "output.warning")
                # Malformed function call, continue to the next part
                continue

            for part in candidate.content.parts:
                if part.function_call:
                    # Please ensure that the number of function response parts is equal to the number of function call parts of the function call turn.
                    self.agent.add_content(
                        types.Content(
                            role="model",
                            parts=[part]
                        )
                    )

                    function_call = part.function_call
                    print_pt(f"[TOOL] Function call: {function_call.name}, args: {truncate_text_both_ends(str(function_call.args))}", "output.tool")

                    tool_result = await self.mcp_session.call_tool(
                        function_call.name,
                        arguments=dict(function_call.args),
                    )
                    print_pt(f"[TOOL] Tool result: {truncate_text_both_ends(str(tool_result))}", "output.tool")

                    function_response_part = types.Part.from_function_response(
                        name = function_call.name,
                        response = { "result": tool_result },
                    )
                    self.agent.add_content(
                        types.Content(
                            role="user",
                            parts=[function_response_part]
                        )
                    )

                    if function_call.name == "forget":
                        self.agent.forget_history()
                        # TODO: hack for pro-active tool calling
                        self.agent.add_content(
                            types.Content(
                                role="user",
                                parts=[types.Part(text="Begin the task.")]
                            )
                        )
                        continue

                    if function_call.name == "ask":
                        # get user input
                        print(f"Model (clarification): {function_call.args['question']}")
                        self.agent.save_history()

                        answer = await self.prompt_session.prompt_async(
                            FormattedText([("class:prompt", "User (clarification): ")]),
                            style=PROMPT_STYLE_OBJ
                        )

                        self.agent.add_content(
                            types.Content(
                                role="user", 
                                parts=[types.Part(text=answer)]
                            )
                        )

                else:

                    # Skip empty parts
                    if len(part.text.strip()) == 0:
                        continue

                    self.agent.add_content(
                        types.Content(
                            role="model",
                            parts=[part]
                        )
                    )
                    print(f"Model:\n{part.text.strip()}")

                self.agent.save_history()

                # TODO: hack for pro-active tool calling
                self.agent.add_content(
                    types.Content(
                        role="user",
                        # parts=[types.Part(text="Continue with the next needful action or if it starts to get repetitive, use the 'think' tool to think or use the 'ask' tool to ask the user for input.")]
                        parts=[types.Part(text="Continue with the next needful action or if it starts to get repetitive, use the 'think' tool to figure out next action or how to make it better.")]
                    )
                )


    async def chat_loop(self):
        """Run an interactive chat loop using prompt_toolkit"""
        prompt_session = PromptSession(history=None)
        print_pt(f"MCP Client Started! (Using prompt_toolkit)")
        print_pt(f"Type your queries or 'quit' to exit.")

        # await self.inlined_process_query_recursive("Re-confirm allowed directories with me and do nothing else.")

        while True:
            try:
                # Use prompt_async with the Style object
                query = await self.prompt_session.prompt_async(
                    FormattedText([("class:prompt", "User: ")]),
                    style=PROMPT_STYLE_OBJ
                )
                query = query.strip()

                if query.lower() == 'quit':
                    break
                
                await self.inlined_process_query_recursive(query)

            except anyio.ClosedResourceError:
                print_pt(f"Connection closed. Attempting to reconnect...", "output.debug")
                await self.connect()

                await self.inlined_process_query_recursive(query)

            except (EOFError, KeyboardInterrupt):
                print_pt(f"Exiting client...", "output.debug")
                break

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MCP Client')
    parser.add_argument('server_url', help='URL of SSE MCP server (i.e. http://localhost:8080/sse)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging as the very first thing
    setup_logging(debug=args.debug)

    server_url = args.server_url
    logging.info(f"MCP Client attempting to connect to: {server_url}")
    client = MCPClient(server_url=server_url)
    try:
        if not await client.connect():
            logging.error(f"Initial connection failed. Exiting.")
            # print_pt already happens within connect on failure
            sys.exit(1)
        
        # Use patch_stdout context manager
        with patch_stdout():
            await client.chat_loop()

    finally:
        logging.info("Initiating final client cleanup.")
        await client.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        # Catch any other unexpected exceptions escaping main
        print_pt(f"\nUnhandled exception occurred: {e}", "output.error")
        import traceback
        print_pt(traceback.format_exc(), "output.error") # Print stack trace too
