# ToolKami - Simple Agents Made Easy

ToolKami is an open sourced ["simple" framework with conceptually clear, independant parts](https://www.youtube.com/watch?v=SxdOUGdseq4) that allows you to build and work seamlessly with AI agents. It comes with a **Command Line Interface** and curated **Tools**.

[![Toolkami framework](images/framework.png)](https://toolkami.com)

## Command Line Interface (CLI)

`toolkami` CLI is a modified version of <img src="https://img.logo.dev/shopify.com?token=pk_EgJ9qk0vTlaOIkXy9RR0sg" alt="Shopify" height="24px" style="vertical-align: text-bottom;"></img>'s CEO Tobias [try implementation](https://github.com/tobi/try). It extends the implementation with sandboxing capabilities and designed with [functional core, imperative shell](https://www.destroyallsoftware.com/talks/boundaries) in mind.

### Usage

Commands:

* `toolkami init [PATH]`: Generate shell function
* `toolkami cd [QUERY]`: Interactive selector
* `toolkami wt [NAME]`: Create worktree from current repo
  * `merge`: Merge worktree changes back to parent repo
  * `drop`: Delete worktree and branch
* `toolkami sb`: Run Docker sandbox from .toolkami/docker-compose.yml
  * `build`: Re-build service image 
  * `exec [CMD...]`: Exec into the sandbox container (defaults to interactive `bash`)

It is designed to support multiple, concurrent agent workflows:

![Toolkami CLI](images/cli.png)

### Installation
```bash
curl -sL https://raw.githubusercontent.com/aperoc/toolkami/refs/heads/main/toolkami.rb > ~/.local/toolkami.rb

# Make "try" executable so it can be run directly
chmod +x ~/.local/toolkami.rb

# Add to your shell (bash/zsh)
echo >> ~/.zshrc # add new line
echo 'eval "$(ruby ~/.local/toolkami.rb init)"' >> ~/.zshrc
```

## Framework

ToolKami's framework let **deterministic tools** and **dynamic agents** to work together seamlessly. It is designed on the premise of simplicity, composability and extensibility that scales nicely with LLM's increasing capability.

All the MCP servers can be [distributed as a single file binary, thanks to UV script](https://blog.toolkami.com/mcp-server-in-a-file/).

I have elaborated on [default File and Shell tool in this blog post, along with what can be improved](https://blog.toolkami.com/openai-codex-tools/).

![Toolkami Tools](images/tools.png)

### Browse (WIP)

`Browse` tool allows agent to access up-to-date documentations, autonomously evaluate client-side rendering and Javascript errors when development web application.

![Demo of agent browsing the web](images/agent-demo.gif)

You can even feed it to a Vision Language Model (VLM) to get a structured output if a sound or animation is played back correctly.

![Browser VLM](images/vlm.png)

### Installation
```bash
# Install UV

## OSX/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
## Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Start the MCP server
`./servers/__main__.py`
```

## Use Cases

### Google's AlphaEvolve: ToolKami style

A minimal implementation of AlphaEvolve using this framework with [detailed writeup](https://toolkami.com/alphaevolve-toolkami-style/) and [code](https://github.com/aperoc/toolkami/pull/5).

![AlphaEvolve's Architecture](images/alphaevolve.png)
(Credits to https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

## Social
- [Website](https://toolkami.com)
- [Blog](https://blog.toolkami.com/blog/)
- [Twitter](https://x.com/tool_kami)
- [toolkami@aperoc.com](mailto:toolkami@aperoc.com)

## Vision

[![The best DX for structured AI Workflows](images/hero.gif)](mailto:toolkami@aperoc.com)

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

<span style="color: red;">^^^</span> Then use **node-based editor** for non-technical members to customize and test prompts.
![node based editor](images/node-editor.png)

Features marked <span style="color: red;">^^^</span> are in private beta, please [contact us](mailto:toolkami@aperoc.com) for access.
