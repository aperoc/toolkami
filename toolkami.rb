#!/usr/bin/env ruby

require 'io/console'
require 'fileutils'
require 'yaml'
require 'shellwords'

# Double-buffered UI module - adapted from try.rb
module UI
  TOKEN_MAP = {
    '{text}' => "\e[39m",
    '{dim}' => "\e[90m",
    '{highlight}' => "\e[1;33m",
    '{reset}' => "\e[0m",
  }.freeze

  @@buffer = []
  @@last_buffer = []
  @@current_line = ""

  def self.print(text, io: STDERR)
    return if text.nil?
    @@current_line += text
  end

  def self.puts(text = "", io: STDERR)
    @@current_line += text
    @@buffer << @@current_line
    @@current_line = ""
  end

  def self.flush(io: STDERR)
    # Finalize current line into buffer
    unless @@current_line.empty?
      @@buffer << @@current_line
      @@current_line = ""
    end

    # In non-TTY contexts, print plain text without control codes
    unless io.tty?
      plain = @@buffer.join("\n").gsub(/\{.*?\}/, '')
      io.print(plain)
      io.print("\n") unless plain.end_with?("\n")
      @@last_buffer = []
      @@buffer.clear
      @@current_line = ""
      io.flush
      return
    end

    # Position cursor at home for TTY
    io.print("\e[H")

    max_lines = [@@buffer.length, @@last_buffer.length].max
    reset = TOKEN_MAP['{reset}']

    # Double buffering: only redraw changed lines
    (0...max_lines).each do |i|
      current_line = @@buffer[i] || ""
      last_line = @@last_buffer[i] || ""

      if current_line != last_line
        # Move to line and clear it, then write new content
        io.print("\e[#{i + 1};1H\e[2K")
        if !current_line.empty?
          processed_line = expand_tokens(current_line)
          io.print(processed_line)
          io.print(reset)
        end
      end
    end

    # Store current buffer as last buffer for next comparison
    @@last_buffer = @@buffer.dup
    @@buffer.clear
    @@current_line = ""

    io.flush
  end

  def self.cls(io: STDERR)
    @@current_line = ""
    @@buffer.clear
    @@last_buffer.clear
    io.print("\e[2J\e[H")  # Clear screen and go home
  end

  def self.expand_tokens(str)
    str.gsub(/\{.*?\}/) { |match| TOKEN_MAP.fetch(match, '') }
  end

  def self.read_key
    input = STDIN.getc
    if input == "\e"
      input << STDIN.read_nonblock(3) rescue ""
      input << STDIN.read_nonblock(2) rescue ""
    end
    input
  end
end

class ToolkamiSelector
  DEFAULT_PATH = ENV['TOOLKAMI_PATH'] || File.expand_path("~/kamis")

  def initialize(search_term = "", base_path: DEFAULT_PATH)
    @search_term = search_term.strip
    @base_path = base_path
    @cursor = 0

    FileUtils.mkdir_p(@base_path) unless Dir.exist?(@base_path)
  end

  def self.is_git_uri?(arg)
    return false unless arg
    arg.match?(%r{^(https?://|git@)}) || arg.include?('github.com') || arg.include?('gitlab.com') || arg.end_with?('.git')
  end

  def self.parse_git_uri(uri)
    # Remove .git suffix if present
    uri = uri.sub(/\.git$/, '')

    # Handle different git URI formats
    if uri.match(%r{^https?://github\.com/([^/]+)/([^/]+)})
      # https://github.com/user/repo
      user, repo = $1, $2
      return { user: user, repo: repo, host: 'github.com' }
    elsif uri.match(%r{^git@github\.com:([^/]+)/([^/]+)})
      # git@github.com:user/repo
      user, repo = $1, $2
      return { user: user, repo: repo, host: 'github.com' }
    elsif uri.match(%r{^https?://([^/]+)/([^/]+)/([^/]+)})
      # https://gitlab.com/user/repo or other git hosts
      host, user, repo = $1, $2, $3
      return { user: user, repo: repo, host: host }
    elsif uri.match(%r{^git@([^:]+):([^/]+)/([^/]+)})
      # git@host:user/repo
      host, user, repo = $1, $2, $3
      return { user: user, repo: repo, host: host }
    else
      return nil
    end
  end

  def self.generate_clone_directory_name(git_uri, custom_name = nil)
    return custom_name if custom_name && !custom_name.empty?

    parsed = parse_git_uri(git_uri)
    return nil unless parsed

    date_prefix = Time.now.strftime("%Y-%m-%d")
    "#{date_prefix}-#{parsed[:user]}-#{parsed[:repo]}"
  end

  def run
    return nil unless STDIN.tty? && STDERR.tty?

    UI.cls  # Clear screen once at start
    STDERR.raw do
      STDERR.print("\e[?25l")  # Hide cursor
      loop do
        items = get_items
        render(items)

        key = UI.read_key
        case key
        when "\r"  # Enter
          result = items[@cursor]
          STDERR.print("\e[?25h")  # Show cursor
          UI.cls
          return result ? result[:path] : create_new
        when "\e[A", "\x10"  # Up, Ctrl-P
          @cursor = [@cursor - 1, 0].max
        when "\e[B", "\x0E"  # Down, Ctrl-N
          @cursor = [@cursor + 1, items.length].min
        when "\x7F"  # Backspace
          @search_term = @search_term[0...-1]
          @cursor = 0
        when "\e", "\x03"  # ESC, Ctrl-C
          STDERR.print("\e[?25h")
          UI.cls
          return nil
        when /^[a-zA-Z0-9\-_. ]$/
          @search_term += key
          @cursor = 0
        end
      end
    end
  ensure
    STDERR.print("\e[?25h")
    UI.cls
  end

  private

  def get_items
    dirs = Dir.entries(@base_path)
      .reject { |e| e.start_with?('.') }
      .select { |e| File.directory?(File.join(@base_path, e)) }
      .map { |name| { name: name, path: File.join(@base_path, name) } }

    return dirs if @search_term.empty?

    # Simple fuzzy filter
    query = @search_term.downcase
    dirs.select { |d| fuzzy_match?(d[:name].downcase, query) }
  end

  def fuzzy_match?(text, query)
    qi = 0
    text.each_char do |c|
      qi += 1 if qi < query.length && c == query[qi]
    end
    qi == query.length
  end

  def render(items)
    UI.puts "{highlight}Toolkami Selector{reset}"
    UI.puts "{dim}#{'─' * 40}{reset}"
    UI.puts "Search: #{@search_term}"
    UI.puts "{dim}#{'─' * 40}{reset}"

    items.each_with_index do |item, idx|
      prefix = idx == @cursor ? "→ " : "  "
      UI.puts "#{prefix}#{item[:name]}"
    end

    # "Create new" option
    prefix = @cursor == items.length ? "→ " : "  "
    UI.puts ""
    UI.puts "#{prefix}+ Create new: #{@search_term.empty? ? '(enter name)' : @search_term}"

    UI.flush  # Flush double buffer
  end

  def create_new
    name = @search_term.empty? ? nil : @search_term

    unless name
      STDERR.print("\e[2J\e[H\e[?25h")
      UI.puts "Enter name:"
      STDERR.cooked { name = STDIN.gets.chomp }
      return nil if name.empty?
    end

    # Check if it's a git URI
    if ToolkamiSelector.is_git_uri?(name)
      dir_name = ToolkamiSelector.generate_clone_directory_name(name)
      return { type: :clone, uri: name, path: File.join(@base_path, dir_name) } if dir_name
    end

    # Add date prefix for new directories
    date_prefix = Time.now.strftime("%Y-%m-%d")
    sanitized_name = name.gsub(/\s+/, '-')
    File.join(@base_path, "#{date_prefix}-#{sanitized_name}")
  end
end

class ConfigSelector
  def initialize(base_path:)
    @base_path = base_path
    @config_dir = File.join(base_path, '.configs')
    @cursor = 0
  end

  def run
    return nil unless STDIN.tty? && STDERR.tty?

    # Return nil early if no configs directory exists
    return nil unless Dir.exist?(@config_dir)

    configs = get_configs
    return nil if configs.empty?

    UI.cls  # Clear screen once at start
    STDERR.raw do
      STDERR.print("\e[?25l")  # Hide cursor
      loop do
        render(configs)

        key = UI.read_key
        case key
        when "\r"  # Enter
          STDERR.print("\e[?25h")
          UI.cls
          return @cursor == 0 ? nil : configs[@cursor - 1]
        when "\e[A", "\x10"  # Up, Ctrl-P
          @cursor = [@cursor - 1, 0].max
        when "\e[B", "\x0E"  # Down, Ctrl-N
          @cursor = [@cursor + 1, configs.length].min
        when "\e", "\x03"  # ESC, Ctrl-C
          STDERR.print("\e[?25h")
          UI.cls
          return nil
        end
      end
    end
  ensure
    STDERR.print("\e[?25h")
    UI.cls
  end

  private

  def get_configs
    return [] unless Dir.exist?(@config_dir)

    Dir.entries(@config_dir)
      .reject { |e| e.start_with?('.') }
      .select { |name|
        path = File.join(@config_dir, name)
        File.directory?(path) && File.exist?(File.join(path, 'Dockerfile'))
      }
      .sort
  end

  def render(configs)
    UI.puts "{highlight}Select Config{reset}"
    UI.puts "{dim}#{'─' * 40}{reset}"

    # "Skip" option at index 0
    prefix = @cursor == 0 ? "→ " : "  "
    UI.puts "#{prefix}{dim}Skip (no config){reset}"
    UI.puts ""

    # Config options start at index 1
    configs.each_with_index do |name, idx|
      prefix = @cursor == (idx + 1) ? "→ " : "  "
      UI.puts "#{prefix}#{name}"
    end

    UI.puts "{dim}#{'─' * 40}{reset}"
    UI.puts "{dim}↑↓: Navigate  Enter: Select  ESC: Cancel{reset}"

    UI.flush  # Flush double buffer
  end
end

# CLI Entry Point
if __FILE__ == $0
  def print_help
    puts <<~HELP
      Toolkami - Minimal directory selector

      Usage:
        toolkami.rb init [PATH]    # Generate shell function
        toolkami.rb cd [QUERY]     # Interactive selector
        toolkami.rb worktree       # Create worktree from current repo
        toolkami.rb merge          # Merge worktree changes back to parent repo
        toolkami.rb drop           # Delete worktree and branch
        toolkami.rb sb             # Run Docker sandbox from .toolkami/docker-compose.yml
        toolkami.rb sb exec [CMD...]  # Exec into running sandbox container (default: bash)
        toolkami.rb sb build       # Rebuild sandbox image (compose build --no-cache)

      Config Selection:
        Place configs in $TOOLKAMI_PATH/.configs/
        Each config dir should contain:
          - Dockerfile (required)
          - docker-compose.yml (required)
          - config.toml, settings.json, etc. (optional)
          - Other files (copied to .toolkami/ in worktree)

      Environment:
        TOOLKAMI_PATH - Root directory (default: ~/kamis)
    HELP
  end

  def sanitize_service_name(name)
    name.downcase.gsub(/[^a-z0-9_.-]/, '-').gsub(/-+/, '-').sub(/^-+/, '').sub(/-+$/, '')
  end

  def determine_service_name(compose_path)
    data = YAML.safe_load(File.read(compose_path))
    if data.is_a?(Hash) && data['services'].is_a?(Hash) && !data['services'].empty?
      data['services'].keys.first.to_s
    else
      sanitize_service_name(File.basename(Dir.pwd))
    end
  rescue
    sanitize_service_name(File.basename(Dir.pwd))
  end

  def ensure_docker_available!
    return if system('command -v docker >/dev/null 2>&1')

    STDERR.puts "Error: Docker is not installed or not in PATH"
    STDERR.puts "Install Docker from https://docs.docker.com/get-docker/"
    exit 1
  end

  command = ARGV.shift

  case command
  when 'init'
    path = ARGV.shift || ToolkamiSelector::DEFAULT_PATH
    script_path = File.expand_path($0)

    # Create directory structure
    FileUtils.mkdir_p(path)
    configs_dir = File.join(path, '.configs')
    FileUtils.mkdir_p(configs_dir)

    # Create example codex config if it doesn't exist
    codex_config_dir = File.join(configs_dir, 'codex')
    unless Dir.exist?(codex_config_dir)
      FileUtils.mkdir_p(codex_config_dir)

      # Create example Dockerfile
      dockerfile_path = File.join(codex_config_dir, 'Dockerfile')
      File.write(dockerfile_path, <<~DOCKERFILE)
        FROM node:22
        WORKDIR /workspace

        RUN npm install -g @openai/codex

        CMD ["/bin/bash"]
      DOCKERFILE

      # Create example config.toml
      config_toml_path = File.join(codex_config_dir, 'config.toml')
      File.write(config_toml_path, <<~TOML)
        model = "gpt-5-codex"
        approval_policy = "never"
        sandbox_mode = "danger-full-access"
        sandbox_workspace_write.network_access = true

        [shell_environment_policy]
        ignore_default_excludes = true

        [tools]
        web_search = true

        [projects."/workspace"]
        trust_level = "trusted"
      TOML

      # Create example docker-compose.yml
      compose_path = File.join(codex_config_dir, 'docker-compose.yml')
      File.write(compose_path, <<~COMPOSE)
        services:
          toolkami:
            build:
              context: .
              dockerfile: Dockerfile
            volumes:
              - ..:/workspace
              - ./config.toml:/root/.codex/config.toml:ro
            network_mode: host
            working_dir: /workspace
            stdin_open: true
            tty: true
            command: ["/bin/bash"]
      COMPOSE

      STDERR.puts "✓ Created example config at #{codex_config_dir}"
    end

    # Simple bash/zsh wrapper
    puts <<~SHELL
      toolkami() {
        script_path='#{script_path}'
        case "$1" in
          worktree|init|merge|drop|sb)
            cmd=$(/usr/bin/env ruby "$script_path" "$@" 2>/dev/tty)
            ;;
          *)
            cmd=$(/usr/bin/env ruby "$script_path" cd "$@" 2>/dev/tty)
            ;;
        esac
        [ $? -eq 0 ] && [ -n "$cmd" ] && eval "$cmd"
      }
    SHELL

  when 'cd', nil
    search = ARGV.join(' ')
    selector = ToolkamiSelector.new(search)
    result = selector.run

    if result
      # Handle different result types
      if result.is_a?(Hash) && result[:type] == :clone
        # Git clone workflow
        quoted_path = "'" + result[:path].gsub("'", %q('"'"')) + "'"
        quoted_uri = "'" + result[:uri].gsub("'", %q('"'"')) + "'"
        puts "mkdir -p #{quoted_path} && git clone #{quoted_uri} #{quoted_path} && cd #{quoted_path}"
      else
        # Regular directory creation
        quoted = "'" + result.gsub("'", %q('"'"')) + "'"
        puts "mkdir -p #{quoted} && cd #{quoted}"
      end
    end

  when 'worktree'
    # Get custom name from args or use repo name
    custom_name = ARGV.join(' ')

    # Get current directory base name
    base = if custom_name && !custom_name.strip.empty?
      custom_name.gsub(/\s+/, '-')
    else
      File.basename(Dir.pwd)
    end

    # Add date prefix
    date_prefix = Time.now.strftime("%Y-%m-%d")
    dir_name = "#{date_prefix}-#{base}"
    full_path = File.join(ToolkamiSelector::DEFAULT_PATH, dir_name)

    # Check if we're in a git repo
    if File.directory?(File.join(Dir.pwd, '.git'))
      # Check for config selection
      config_selector = ConfigSelector.new(base_path: ToolkamiSelector::DEFAULT_PATH)
      selected_config = config_selector.run

      # Build shell command parts
      quoted_path = "'" + full_path.gsub("'", %q('"'"')) + "'"
      branch_name = "feature/#{dir_name}"
      commands = [
        "mkdir -p #{quoted_path}",
        "git worktree add -b #{branch_name} #{quoted_path}"
      ]

      # Add config copy if selected
      if selected_config
        # Add confirmation message to shell output
        commands << "echo '✓ Config: #{selected_config} → .toolkami/' >&2"

        config_src = File.join(ToolkamiSelector::DEFAULT_PATH, '.configs', selected_config)
        quoted_src = "'" + config_src.gsub("'", %q('"'"')) + "'"
        toolkami_dir = File.join(full_path, '.toolkami')
        quoted_toolkami = "'" + toolkami_dir.gsub("'", %q('"'"')) + "'"

        commands << "mkdir -p #{quoted_toolkami}"
        commands << "cp -r #{quoted_src}/. #{quoted_toolkami}/"

        # After copying, rename the first docker compose service to the directory name (sanitized)
        compose_dest = File.join(full_path, '.toolkami', 'docker-compose.yml')
        quoted_compose = "'" + compose_dest.gsub("'", %q('"'"')) + "'"
        service_name = dir_name.downcase.gsub(/[^a-z0-9_.-]/, '-').gsub(/-+/, '-').sub(/^-+/, '').sub(/-+$/, '')
        quoted_service = "'" + service_name.gsub("'", %q('"'"')) + "'"
        ruby_edit = %q{p=ARGV[0]; s=ARGV[1]; l=File.readlines(p); i=l.index{|x| x =~ /^\s*services\s*:/}; if i; j=i+1; while j<l.length && l[j] =~ /^\s*(?:#.*)?$/; j+=1; end; if j<l.length && l[j] =~ /^(\s+)([A-Za-z0-9_.-]+):/; ind=$1; l[j]=l[j].sub(/^(\s+)[A-Za-z0-9_.-]+:/, ind + s + ":"); File.write(p, l.join); end; end}
        commands << "[ -f #{quoted_compose} ] && ruby -e '#{ruby_edit}' #{quoted_compose} #{quoted_service} || true"
      end

      commands << "cd #{quoted_path}"
      puts commands.join(" && ")
    else
      STDERR.puts "Error: Not in a git repository"
      exit 1
    end

  when 'merge'
    # Check if we're in a git worktree
    git_common_dir = `git rev-parse --git-common-dir 2>/dev/null`.strip
    if $?.exitstatus != 0
      STDERR.puts "Error: Not in a git repository"
      exit 1
    end

    git_dir = `git rev-parse --git-dir 2>/dev/null`.strip
    if git_common_dir == git_dir || git_common_dir == '.git'
      STDERR.puts "Error: Not in a worktree (you're in the main repository)"
      exit 1
    end

    # Check for uncommitted changes
    status_output = `git status --porcelain 2>/dev/null`.strip
    if !status_output.empty?
      STDERR.puts "Error: You have uncommitted changes. Please commit or stash them first."
      STDERR.puts ""
      STDERR.puts "Uncommitted changes:"
      STDERR.puts status_output
      exit 1
    end

    # Get current commit SHA
    commit_sha = `git rev-parse HEAD 2>/dev/null`.strip
    if commit_sha.empty?
      STDERR.puts "Error: Could not determine current commit"
      exit 1
    end

    # Get parent repo location
    # git_common_dir returns something like /path/to/repo/.git
    # We want /path/to/repo
    parent_repo = File.dirname(git_common_dir)

    # Get current worktree path for cleanup
    worktree_path = Dir.pwd
    quoted_worktree = "'" + worktree_path.gsub("'", %q('"'"')) + "'"
    quoted_parent = "'" + parent_repo.gsub("'", %q('"'"')) + "'"

    # Get branch name for merge
    branch_name = `git rev-parse --abbrev-ref HEAD 2>/dev/null`.strip

    # Emit shell commands: cd to parent and squash-merge the branch, then return to worktree
    if branch_name.empty? || branch_name == "HEAD"
      # Fallback to cherry-pick for detached HEAD
      puts "cd #{quoted_parent} && git cherry-pick #{commit_sha} && cd #{quoted_worktree}"
    else
      puts "cd #{quoted_parent} && git merge --squash #{branch_name} && git commit -m 'squash: merge #{branch_name}' && cd #{quoted_worktree}"
    end

  when 'drop'
    # Check if we're in a git worktree
    git_common_dir = `git rev-parse --git-common-dir 2>/dev/null`.strip
    if $?.exitstatus != 0
      STDERR.puts "Error: Not in a git repository"
      exit 1
    end

    git_dir = `git rev-parse --git-dir 2>/dev/null`.strip
    if git_common_dir == git_dir || git_common_dir == '.git'
      STDERR.puts "Error: Not in a worktree (you're in the main repository)"
      exit 1
    end

    # Get parent repo location
    parent_repo = File.dirname(git_common_dir)

    # Get current worktree path for cleanup
    worktree_path = Dir.pwd
    quoted_worktree = "'" + worktree_path.gsub("'", %q('"'"')) + "'"
    quoted_parent = "'" + parent_repo.gsub("'", %q('"'"')) + "'"

    # Emit shell commands: cd to parent, remove worktree and delete branch
    branch_name = `git rev-parse --abbrev-ref HEAD 2>/dev/null`.strip
    if branch_name.empty? || branch_name == "HEAD"
      # Just remove worktree for detached HEAD
      puts "cd #{quoted_parent} && git worktree remove --force #{quoted_worktree}"
    else
      # Remove worktree and delete branch
      puts "cd #{quoted_parent} && git worktree remove --force #{quoted_worktree} && git branch -D #{branch_name}"
    end

  when 'sb'
    # Check for .toolkami/docker-compose.yml
    compose_path = File.join(Dir.pwd, '.toolkami', 'docker-compose.yml')
    unless File.exist?(compose_path)
      STDERR.puts "Error: No .toolkami/docker-compose.yml found in current directory"
      STDERR.puts "Run 'toolkami worktree' from a git repo and select a config to create a sandbox."
      exit 1
    end

    # Check if Docker is available
    ensure_docker_available!

    # Quote current directory for shell command
    quoted_pwd = "'" + Dir.pwd.gsub("'", %q('"'"')) + "'"

    # Optional subcommand: build, exec
    subcommand = ARGV.shift

    # Determine service name from compose file, fallback to sanitized directory name
    service_name = determine_service_name(compose_path)

    case subcommand
    when 'build'
      puts "cd #{quoted_pwd} && docker compose -f .toolkami/docker-compose.yml build --no-cache #{service_name}"
    when 'exec'
      exec_args = ARGV.empty? ? ['bash'] : ARGV
      exec_command = exec_args.map { |arg| Shellwords.escape(arg) }.join(' ')
      puts "cd #{quoted_pwd} && docker compose -f .toolkami/docker-compose.yml exec -it #{service_name} #{exec_command}".strip
    else
      puts "cd #{quoted_pwd} && docker compose -f .toolkami/docker-compose.yml run --rm #{service_name}"
    end

  when '--help', '-h'
    print_help

  else
    STDERR.puts "Unknown command: #{command}"
    print_help
    exit 1
  end
end
