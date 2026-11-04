"""
Rich terminal formatting, panels, syntax highlighting, and event handlers.
"""

from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..core.agent import AgentEvent, AgentEventType


class RichAgentConsole:
    """Renders agent events to terminal using Rich."""

    def __init__(self, console: Optional[Console] = None, quiet: bool = False):
        self.console = console or Console(safe_box=True)
        self.quiet = quiet

    def handle_event(self, event: AgentEvent) -> None:
        if self.quiet:
            return

        if event.event_type == AgentEventType.STEP:
            self.console.print(f"\n[bold cyan]--- Step {event.step} ---[/bold cyan]")

        elif event.event_type == AgentEventType.THINKING:
            self.console.print(f"[dim italic]Thinking: {escape(str(event.data))}[/dim italic]")

        elif event.event_type == AgentEventType.TOOL_CALL:
            data = event.data
            fn_name = data.get("name", "Tool")
            fn_args = data.get("arguments", "{}")
            self.console.print(
                Panel(
                    Syntax(fn_args, "json", theme="monokai", word_wrap=True),
                    title=f"[bold green][Tool Call] {fn_name}[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                )
            )

        elif event.event_type == AgentEventType.TOOL_RESULT:
            data = event.data
            fn_name = data.get("name", "Tool")
            result = data.get("result", "")
            # Truncate preview if very long for console rendering
            preview = result if len(result) < 1500 else result[:1500] + "\n... [Output preview clipped in console]"
            self.console.print(
                Panel(
                    Text(preview),
                    title=f"[bold blue][Tool Result] {fn_name}[/bold blue]",
                    border_style="blue",
                    padding=(0, 1),
                )
            )

        elif event.event_type == AgentEventType.ANSWER:
            self.console.print(
                Panel(
                    Markdown(event.data),
                    title="[bold magenta]Assistant Response[/bold magenta]",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

        elif event.event_type == AgentEventType.ERROR:
            err_data = event.data
            if isinstance(err_data, dict):
                err_text = err_data.get("error", "Unknown error")
                suggestions = err_data.get("suggestions", [])
                sugg_text = ""
                if suggestions:
                    sugg_text = "\n\n[yellow]Suggested models to try:[/yellow]\n" + "\n".join(
                        f"  * {s}" for s in suggestions
                    )
                self.console.print(
                    Panel(
                        f"[bold red]{escape(err_text)}[/bold red]{sugg_text}",
                        title="[bold red]Execution Error[/bold red]",
                        border_style="red",
                    )
                )
            else:
                self.console.print(f"[bold red]Error: {escape(str(err_data))}[/bold red]")

    def print_usage_summary(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        account_usage: Optional[dict] = None,
        is_local: bool = False,
        is_custom_api: bool = False,
    ) -> None:
        """Render a clean token telemetry table."""
        if self.quiet:
            return
        table = Table(title="Telemetry & Usage", border_style="dim")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green", justify="right")
        table.add_row("Prompt Tokens", f"{prompt_tokens:,}")
        table.add_row("Completion Tokens", f"{completion_tokens:,}")
        table.add_row("Total Session Tokens", f"{(prompt_tokens + completion_tokens):,}")

        if is_local:
            table.add_row("Requests", "Unlimited (Local)")
        elif is_custom_api:
            table.add_row("Provider", "Custom API")
        elif account_usage:
            daily = account_usage.get("free_model_daily_requests")
            if daily:
                table.add_row(
                    "Daily Free Requests",
                    f"{daily.get('remaining')}/{daily.get('limit')} left ({daily.get('used')} used)",
                )
            elif account_usage.get("limit_remaining") is not None:
                table.add_row("Credits Remaining", f"${account_usage.get('limit_remaining'):.4f}")

        self.console.print(table)
