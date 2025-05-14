from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig
from pathlib import Path
import os

from typing import Protocol, Dict, Any, Optional
import argparse
from mcp.server.fastmcp import FastMCP
from . import MCPModule

os.environ["ANONYMIZED_TELEMETRY"] = "false"

class BrowserModule:
    def __init__(self):
        self._mcp = FastMCP("browser")
        self._mcp_server = self._mcp._mcp_server

        if os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("BROWSE_MODEL"),
                temperature=0,
                max_retries=3,
            )
        elif os.getenv("OPENAI_API_KEY"):
            self.llm=ChatOpenAI(
                model=os.getenv("BROWSE_MODEL"),
            )
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.llm=ChatAnthropic(
                model=os.getenv("BROWSE_MODEL"),
                temperature=0.0,
                timeout=100, # Increase for complex tasks
            )
        else:
            self.llm = ChatVertexAI(
                model=os.getenv("BROWSE_MODEL"),
                temperature=0,
                max_retries=3,
            )

    async def initialize_playwright(self):
        """Initialize Playwright browser connection."""
        pass

    def initialize(self, args: argparse.Namespace = None) -> None:
        """Initialize the adhoc module with command line arguments.
        
        Args:
            args: Command line arguments namespace. Currently unused.
        """
        pass

    def register_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register command line arguments for this module.
        
        Args:
            parser: The argument parser to add arguments to.
        """
        pass

    def register_tools(self, mcp: FastMCP) -> None:
        """Register the adhoc tools with the given MCP instance."""

        @mcp.tool()
        async def browse(task: str) -> str:
            """Browse the web with an agent and return the result.

            Args:
                task: The task to complete.
            """

            browser_config = BrowserConfig()
            browser = Browser(config=browser_config)
            context_config = BrowserContextConfig()
            browser_context = await browser.new_context(config=context_config)

            agent = Agent(
                task=task,
                llm=self.llm,
                browser_context=browser_context,
                message_context=""
            )
            history = await agent.run()

            result = {
                'history_final_result': history.final_result(),
            }

            await browser_context.close()

            return str(result)

# Create a singleton instance of the module
browser_module = BrowserModule() 

