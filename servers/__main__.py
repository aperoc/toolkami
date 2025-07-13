# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2,<3",
#     "mcp>=1.2.0,<2",
# ]
# ///

from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount

from shttp_modules.filesystem import mcp as filesystem_mcp

def combine_lifespans(*lifespans):
    @asynccontextmanager
    async def combined_lifespan(app):
        async with AsyncExitStack() as stack:
            for l in lifespans:
                ctx = l(app)
                await stack.enter_async_context(ctx)
            yield
    return combined_lifespan


main_app = Starlette(
    routes=[
        Mount("/filesystem/", app=filesystem_mcp.streamable_http_app()),
    ],
    lifespan=combine_lifespans(
        lambda _: filesystem_mcp.session_manager.run(),
    ),
)

if __name__ == "__main__":
    uvicorn.run(
        "__main__:main_app",
        host="127.0.0.1",
        port=8002,
        reload=True,
        reload_dirs=["."]
    )