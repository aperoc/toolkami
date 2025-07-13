# Toolkami (https://toolkami.com)

[![The best DX for structured AI Workflows](images/hero.gif)](mailto:toolkami@aperoc.com)

**Seven tools is all you need**. A minimal AI agent framework that just works, using only 7 tools. Comes with hands-free `Turbo mode` and `Hot-reloading` for self-modification.

Following the UNIX philosophy - build upon a collection of minimal, composable tools that scales with LLM's capabilities.
<a href="https://toolkami.com">
  <img src="images/7-tools.png" alt="7 tools: Ask, Browse, File, Shell + 3 Task-specific tools" style="border-radius:6px;">
</a>

**Go Turbo**: The standard pace is for chumps. Have it go full autonomous by commenting out the `ask` tool.
```python
# @mcp.tool()
async def ask(question: str) -> str:
```

<span style="color: red;">^^^</span> **Interleave** non-deterministic AI behaviour with normal code execution. First, define your AI workflow in a declarative way in YAML.
```yml
name: create_tests
model: gpt-4.1

tools:
  - Kami::Tools::Grep
  - Kami::Tools::EditFile

steps:
  - git_diff: $(git diff)
  - Generate a list of tests to write based on the changes.
  - run_linter: $(rubocop)
```

<span style="color: red;">^^^</span> And use our **node-based editor** for non-technical members to customize and test prompts.
![node based editor](images/node-editor.png)

Features marked <span style="color: red;">^^^</span> are in private beta, please [contact us](mailto:toolkami@aperoc.com) for access.

**Watch it in action:**
![Demo of agent browsing the web](images/agent-demo.gif)

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

## (In-progress) Beta version
Migrating MCP from SSE to SHTTP

1. Start the server
```bash
uv run server/__main__.py
```

## Troubleshooting
* Delete `content_history.json` to clear message history.

## Roadmap
* [x] OpenAI compatible APIs (including Anthropic)
* [ ] **(in-progress)** Migration from SSE to SHTTP
* [ ] System prompt guidelines with single file project templates
* [ ] OpenAI editing format

## Limitations
- This is a customisable sharp tool for now. Guardrails will only be implemented over time.

## Use Cases

### Google's AlphaEvolve: ToolKami style

A minimal implementation of AlphaEvolve using this framework along with diff-fenced editing.
- [Detailed Writeup](https://toolkami.com/alphaevolve-toolkami-style/)
- [Code](https://github.com/aperoc/toolkami/pull/5)

![AlphaEvolve's Architecture](https://lh3.googleusercontent.com/0arf1iMoZrNmKp9wHT5nU5Qp1D834jAUD2mlSA2k8dG3lzW81deaxqBXVuYOLlUiu-R1Luz4Kr2j8wosjdRlJeGZK_pRwiedtQR5qtIneDETuljkpMg=w1232-rw)
(Credits to https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

## Social
- [Website](https://toolkami.com)
- [Blog](https://blog.toolkami.com/blog/)
- [Twitter](https://x.com/tool_kami)
- [toolkami@aperoc.com](mailto:toolkami@aperoc.com)