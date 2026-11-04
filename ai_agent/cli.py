"""
Command-line interface and entry point for Python AI Agent.
"""

import argparse
import sys
# Ensure UTF-8 output encoding across Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path

from rich.markup import escape

from .config import (
    BASE_URL,
    DEFAULT_API_TIMEOUT,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_SHELL_TIMEOUT,
    WORKSPACE_DIR,
    is_openrouter_url,
)
from .core.agent import Agent
from .core.client import create_client, fetch_account_usage, fetch_models
from .core.keys import KeyManager
from .tools.base import ToolRegistry
from .tools.filesystem import EditTool, ListDirTool, ReadTool, WriteTool
from .tools.shell import BashTool
from .ui.console import RichAgentConsole
from .ui.tui import run_tui


def run_interactive_repl(agent: Agent, rich_console: RichAgentConsole) -> None:
    """Run multi-turn conversational REPL in the terminal."""
    rich_console.console.print("\n[bold cyan]Python AI Agent -- Interactive REPL[/bold cyan]")
    rich_console.console.print(
        "[dim]Type your prompt and press Enter. Ctrl+C during a task cancels it. Commands: [bold]/reset[/bold] (clear context), "
        "[bold]/usage[/bold] (view token counts), [bold]/exit[/bold] or [bold]Ctrl+C[/bold] to quit.[/dim]\n"
    )

    while True:
        try:
            prompt = rich_console.console.input("[bold yellow]User > [/bold yellow]").strip()
            if not prompt:
                continue

            if prompt.lower() in ("/exit", "exit", "quit", ":q"):
                rich_console.console.print("[dim]Goodbye![/dim]")
                break

            if prompt.lower() in ("/reset", "reset"):
                agent.memory.reset()
                rich_console.console.print("[cyan]Conversation context has been reset.[/cyan]\n")
                continue

            if prompt.lower() in ("/usage", "usage"):
                is_custom = not agent.is_local and not is_openrouter_url(agent.base_url)
                rich_console.print_usage_summary(
                    agent.memory.total_prompt_tokens,
                    agent.memory.total_completion_tokens,
                    account_usage=None if (agent.is_local or is_custom) else fetch_account_usage(base_url=agent.base_url),
                    is_local=agent.is_local,
                    is_custom_api=is_custom,
                )
                continue

            try:
                agent.run(prompt)
            except KeyboardInterrupt:
                rich_console.console.print("\n[yellow]Task cancelled. Type a new prompt, or 'Continue' to resume.[/yellow]\n")
                continue
            is_custom = not agent.is_local and not is_openrouter_url(agent.base_url)
            rich_console.print_usage_summary(
                agent.memory.total_prompt_tokens,
                agent.memory.total_completion_tokens,
                account_usage=None if (agent.is_local or is_custom) else fetch_account_usage(base_url=agent.base_url),
                is_local=agent.is_local,
                is_custom_api=is_custom,
            )
            print()

        except (KeyboardInterrupt, EOFError):
            rich_console.console.print("\n[dim]Session terminated.[/dim]")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous AI Agent CLI powered by OpenRouter & Local LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--prompt", required=False, help="Single task prompt for the agent to execute")
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Model identifier (defaults to $env:OPENROUTER_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Ask for approval before every Bash command (file tools stay sandboxed either way)",
    )
    parser.add_argument(
        "--repl", "--cli",
        action="store_true",
        help="Launch interactive text console REPL instead of graphical TUI",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the visual Textual Terminal User Interface (TUI) dashboard (default)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Connect to local Ollama server at {DEFAULT_LOCAL_BASE_URL} (no API key required)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom OpenAI-compatible base URL (e.g. for LM Studio, vLLM, or remote servers)",
    )
    parser.add_argument(
        "-l", "--list-models",
        action="store_true",
        help="List available models from the endpoint and exit",
    )
    parser.add_argument(
        "-w", "--workspace",
        default=str(WORKSPACE_DIR),
        help="Path to workspace directory for sandboxed file operations",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum reasoning steps before halting to prevent infinite loops",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_SHELL_TIMEOUT,
        help="Timeout in seconds for shell command executions",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=DEFAULT_API_TIMEOUT,
        help="Timeout in seconds for API LLM calls to prevent hanging (default: 30s)",
    )
    parser.add_argument(
        "-u", "--usage",
        action="store_true",
        help="Check your OpenRouter account limits, remaining daily requests, and credits, then exit",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Run without Rich UI formatting (plain stdout only)",
    )

    args = parser.parse_args()

    # Handle usage query
    if args.usage:
        if not is_openrouter_url(args.base_url):
            print("Account usage endpoint is only available for OpenRouter.", file=sys.stderr)
            sys.exit(1)
        usage = fetch_account_usage(base_url=args.base_url)
        if not usage:
            print("Unable to fetch account usage. Ensure OPENROUTER_API_KEY is configured.", file=sys.stderr)
            sys.exit(1)

        from rich.table import Table
        rich_console = RichAgentConsole(quiet=False)
        table = Table(title="OpenRouter Account Usage & Limits", border_style="dim")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="bold green")

        table.add_row("Key Label", str(usage.get("label", "N/A")))
        table.add_row("Account Tier", "Free Tier" if usage.get("is_free_tier") else "Paid / Custom")

        daily = usage.get("free_model_daily_requests")
        if daily:
            table.add_row("Daily Free Requests Remaining", f"{daily.get('remaining')} of {daily.get('limit')}")
            table.add_row("Daily Free Requests Used Today", str(daily.get('used')))

        if usage.get("limit_remaining") is not None:
            table.add_row("Credit Limit Remaining", f"${usage.get('limit_remaining'):.4f}")

        table.add_row("Total Usage", f"${usage.get('usage', 0):.4f}")
        rich_console.console.print(table)
        return

    # Determine model and base URL
    base_url = args.base_url
    if args.local and not base_url:
        base_url = DEFAULT_LOCAL_BASE_URL

    model = args.model
    if not model:
        model = DEFAULT_LOCAL_MODEL if args.local else DEFAULT_MODEL

    workspace_path = Path(args.workspace).resolve()

    # Handle model listing early if requested
    if args.list_models:
        try:
            client = create_client(
                base_url=base_url,
                is_local=args.local,
                timeout=args.api_timeout,
            )
            print(f"Fetching models from {base_url or BASE_URL}...", file=sys.stderr)
            models = fetch_models(client, only_free=not args.local)
            print(f"\nAvailable Models ({len(models)} found):\n")
            for idx, m_id in enumerate(models, start=1):
                print(f"   {idx:2d}. {m_id}")
            print("\nRun with any model:")
            print(f'   py app/main.py -m "{models[0]}" -p "your task prompt"\n')
            return
        except Exception as e:
            print(f"Configuration Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Launch Textual TUI by default when no single-shot prompt and no --repl requested
    if not args.prompt and not args.repl:
        try:
            run_tui(
                model=model,
                workspace_dir=workspace_path,
                is_local=args.local,
                base_url=base_url,
                max_steps=args.max_steps,
                shell_timeout=args.timeout,
                api_timeout=args.api_timeout,
                confirm_bash=args.confirm,
            )
            return
        except Exception as e:
            print(f"Error launching TUI: {e}", file=sys.stderr)
            sys.exit(1)

    # Initialize client for CLI / REPL; KeyManager also finds Gemini/Groq/DeepSeek keys
    key_manager = None if args.local else KeyManager(base_url=base_url)
    try:
        client = create_client(
            api_key=key_manager.get_current_key() if key_manager else None,
            base_url=base_url,
            is_local=args.local,
            timeout=args.api_timeout,
        )
    except Exception as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)

    rich_console = RichAgentConsole(quiet=args.quiet)

    def approve_command(command: str) -> bool:
        reply = rich_console.console.input(f"[bold yellow]Run Bash command?[/bold yellow] {escape(command)} [y/N] ")
        return reply.strip().lower() in ("y", "yes")

    # Initialize tools & agent
    tools = ToolRegistry([
        ReadTool(workspace_path),
        WriteTool(workspace_path),
        EditTool(workspace_path),
        ListDirTool(workspace_path),
        BashTool(workspace_path, timeout=args.timeout, approve=approve_command if args.confirm else None),
    ])

    agent = Agent(
        client=client,
        model=model,
        tools=tools,
        max_steps=args.max_steps,
        event_handler=rich_console.handle_event,
        key_manager=key_manager,
        is_local=args.local,
        base_url=base_url,
        client_factory=lambda key: create_client(
            api_key=key, base_url=base_url, is_local=args.local, timeout=args.api_timeout
        ),
    )

    if args.prompt:
        # Single-shot CLI execution
        try:
            answer = agent.run(args.prompt)
        except KeyboardInterrupt:
            print("\nCancelled.", file=sys.stderr)
            sys.exit(130)
        if args.quiet:
            print(answer)
        else:
            is_custom = not agent.is_local and not is_openrouter_url(agent.base_url)
            rich_console.print_usage_summary(
                agent.memory.total_prompt_tokens,
                agent.memory.total_completion_tokens,
                account_usage=None if (agent.is_local or is_custom) else fetch_account_usage(base_url=agent.base_url),
                is_local=agent.is_local,
                is_custom_api=is_custom,
            )
    else:
        # Interactive REPL mode
        run_interactive_repl(agent, rich_console)


if __name__ == "__main__":
    main()
