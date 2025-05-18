# Toolkami

**Seven tools is all you need**. A minimal AI agent that just works, using only seven tools. Comes with hands-free `Turbo mode` and `Hot-reloading` for self-modification.

![7 tools: Read, Write Diff, Browse, Command, Ask, Think](images/7-tools.png)

Watch it in action:
![Agent demo](images/agent-demo.gif)

**Go Turbo**: The standard pace is for chumps. Have it go full autonomous by disabling `ask`.
```python
# @mcp.tool()
async def ask(question: str) -> str:
```

## Use Cases

### Google's AlphaEvolve: ToolKami style

A minimal implementation of AlphaEvolve using this framework along with diff-fenced editing.
- [Detailed Writeup](https://toolkami.com/alphaevolve-toolkami-style/)
- [Code](https://github.com/aperoc/toolkami/pull/5)

![AlphaEvolve's Architecture](https://lh3.googleusercontent.com/0arf1iMoZrNmKp9wHT5nU5Qp1D834jAUD2mlSA2k8dG3lzW81deaxqBXVuYOLlUiu-R1Luz4Kr2j8wosjdRlJeGZK_pRwiedtQR5qtIneDETuljkpMg=w1232-rw)
(Credits to https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)



## Quickstart

Devcontainer (`.devcontainer`) included and useable, otherwise proceed with manual install.

1. Provide credentials in `servers/.env` (used for browsing LLM) and `clients/.env` (used by agent) file accordingly.

2. Install UV:
```bash
# OSX/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Start the MCP server:
```bash
# --reload enables Hot-reloading
# --directories allow folders to be written to
cd servers && PYTHONPATH=. uv run app.py --reload --directories /workspaces/toolkami/projects

# For Browser Use (on linux)
# sudo apt-get update && sudo apt-get install -y  libglib2.0-0t64 libnss3 libnspr4 libdbus-1-3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libxkbcommon0 libatspi2.0-0t64 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64
# cd servers && uv run sync && uv run patchright install chromium
```

4. Start the MCP client (`self-executable UV script`):
```bash
# For native Gemini client
./clients/gemini_client.py http://localhost:8081/sse # --debug

# For OpenAI-compatible clients (including Anthropic)
./clients/openai_client.py http://localhost:8081/sse # --debug
```

## Troubleshooting
* Delete `content_history.json` to clear message history.

## Roadmap
* [x] OpenAI compatible APIs (including Anthropic)
* [ ] System prompt guidelines with single file project templates
* [ ] OpenAI editing format

## Limitations
- This is a customisable sharp tool for now. Guardrails will only be implemented over time.

## Social
- [Blog](https://toolkami.com/blog/)
- [Twitter](https://x.com/tool_kami)
- [toolkami@aperoc.com](mailto:toolkami@aperoc.com)