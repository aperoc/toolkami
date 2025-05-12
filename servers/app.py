import uvicorn
import argparse
import importlib
import pkgutil
import os
import json
from typing import List, Type, Dict, Any
from sse_server import create_starlette_app
from dotenv import load_dotenv

# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
print(f"Loading .env file from: {env_path}")
load_dotenv(env_path, override=True)  # Added override=True to ensure values are set
# print(f"Current working directory: {os.getcwd()}")
# print(f"All environment variables: {dict(os.environ)}")

# Environment variable for storing config in reload mode
CONFIG_ENV = 'MCP_SERVER_CONFIG'

def load_modules() -> List[Type['MCPModule']]:
    """Load all MCP modules from the modules directory."""
    modules = []
    from modules import MCPModule
    modules_package = importlib.import_module('modules')
    
    for module_info in pkgutil.iter_modules(modules_package.__path__):
        if module_info.name == '__init__':
            continue
            
        module = importlib.import_module(f'modules.{module_info.name}')
        if hasattr(module, f'{module_info.name}_module'):
            module_instance = getattr(module, f'{module_info.name}_module')
            if isinstance(module_instance, MCPModule):
                modules.append(module_instance)
                print(f"Loaded module: {module_info.name}")
    
    return modules

def get_module_config(module_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get configuration for a specific module based on global config."""
    module_config = config.get(module_name, {})
    print(f"Debug: Getting config for {module_name}: {module_config}")
    return module_config

def create_app(args: argparse.Namespace = None) -> Any:
    """Create the FastAPI application with configured modules.
    
    Args:
        args: Command line arguments namespace. Each module handles its own args.
    """
    # In reload mode, get args from environment variable
    if args is None and CONFIG_ENV in os.environ:
        # Convert the stored dict back to a Namespace object
        stored_args = json.loads(os.environ[CONFIG_ENV])
        args = argparse.Namespace(**stored_args)
        print(f"Debug: Loaded args from environment: {args}")
    
    print(f"Debug: Creating app with args: {args}")
    
    # Create a new FastMCP instance for the combined server
    from mcp.server.fastmcp import FastMCP
    combined_mcp = FastMCP("combined")
    
    # Load and initialize all modules
    modules = load_modules()
    for module in modules:
        # Get module name from the instance
        module_name = module.__class__.__name__.lower().replace('module', '')
        print(f"Initializing {module_name} with args: {args}")
        module.initialize(args)
        module.register_tools(combined_mcp)
    
    return create_starlette_app(
        combined_mcp._mcp_server,
        debug=True,
        health_check=True,
        cors_origins=None
    )

if __name__ == "__main__":
    # Create the base argument parser
    parser = argparse.ArgumentParser(description='Run combined MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to listen on')
    parser.add_argument('--reload', action='store_true', help='Enable hot reloading')
    
    # Load modules and let them register their arguments
    modules = load_modules()
    for module in modules:
        module.register_arguments(parser)
    
    # Parse arguments
    args = parser.parse_args()

    print(f"Debug: Received arguments: {args}")
    print(f"Debug: Directories argument type: {type(args.directories)}")
    print(f"Debug: Directories argument value: {args.directories}")

    print(f"Starting combined MCP server on http://{args.host}:{args.port}")
    if args.reload:
        print("Hot reloading enabled (with forced shutdown).")
        # Store args in environment for reload mode
        os.environ[CONFIG_ENV] = json.dumps(vars(args))
        print(f"Debug: Stored args in environment: {vars(args)}")
        
        uvicorn.run(
            "app:create_app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=["modules"],
            factory=True,
            timeout_graceful_shutdown=0
        )
    else:
        app = create_app(args)
        uvicorn.run(app, host=args.host, port=args.port) 