"""
CLI entry point for the conversational agent system.

Provides an interactive command-line interface for chatting with the AI agent.
Designed to be modular for future web interface adaptation.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from config import settings, LLMProvider, RAGMode
from conversation.manager import ConversationManager


console = Console()

help_text = """
[bold]Available Commands:[/bold]

  [cyan]/quit, /exit[/cyan] - Exit the application
  [cyan]/new[/cyan]         - Start a new conversation
  [cyan]/history[/cyan]     - Show conversation history
  [cyan]/help[/cyan]        - Show this help message

[bold]Environment Variables:[/bold]

  [cyan]LLM_PROVIDER[/cyan]         - openai or anthropic (default: openai)
  [cyan]OPENAI_API_KEY[/cyan]       - Your OpenAI API key
  [cyan]ANTHROPIC_API_KEY[/cyan]    - Your Anthropic API key
  [cyan]RAG_MODE[/cyan]             - none, semantic, or graph (default: none)
  [cyan]MAX_CONVERSATION_TURNS[/cyan] - Max history turns (default: 20)
  [cyan]MAX_PROMPT_TOKENS[/cyan]    - Max tokens in prompt (default: 4000)
"""

def print_welcome():
    """Print welcome message and configuration info."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]AI Conversational Agent[/bold blue]\n"
        "[dim]Long-term Memory System Demo[/dim]",
        border_style="blue"
    ))
    console.print()
    
    # Show configuration
    config_info = [
        f"[cyan]Provider:[/cyan] {settings.llm_provider.value}",
        f"[cyan]Model:[/cyan] {settings.get_model()}",
        f"[cyan]RAG Mode:[/cyan] {settings.rag_mode.value}",
        f"[cyan]Max Turns:[/cyan] {settings.max_conversation_turns}",
        f"[cyan]Max Tokens:[/cyan] {settings.max_prompt_tokens}",
    ]
    console.print(Panel(
        "\n".join(config_info),
        title="Configuration",
        border_style="dim"
    ))
    console.print()
    console.print("[dim]Commands: /quit, /exit, /history, /new, /help[/dim]")
    console.print()

def print_help():
    """Print help message."""
    console.print(Panel(help_text.strip(), title="Help", border_style="green"))

def print_history(manager: ConversationManager):
    """Print conversation history."""
    history = manager.get_conversation_history()
    
    if not history:
        console.print("[dim]No conversation history yet.[/dim]")
        return
    
    console.print(Panel.fit("[bold]Conversation History[/bold]", border_style="blue"))
    
    for turn in history:
        if turn.role == "user":
            console.print(f"\n[bold green]You:[/bold green] {turn.content}")
        else:
            console.print(f"\n[bold blue]Assistant:[/bold blue] {turn.content}")

def run_chat_loop(manager: ConversationManager, username: str):
    """
    Run the main chat loop.
    
    Args:
        manager: ConversationManager instance
        username: Current user's username
    """
    # Start a new conversation
    manager.start_conversation(username)
    console.print(f"[green]Started new conversation for user: {username}[/green]")
    console.print()
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold green]You[/bold green]")
            
            if not user_input.strip():
                continue
            
            # Handle commands
            command = user_input.strip().lower()
            
            if command in ["/quit", "/exit"]:
                console.print("[yellow]Conversation ended[/yellow]")
                break
            
            if command == "/help":
                print_help()
                continue
            
            if command == "/history":
                print_history(manager)
                continue
            
            if command == "/new":
                manager.start_conversation(username)
                console.print("[green]Started new conversation.[/green]")
                console.print()
                continue
            
            # Send message and get response
            with console.status("[bold blue]Thinking...[/bold blue]"):
                response = manager.send_message(user_input)
            
            # Display response
            console.print()
            console.print("[bold blue]Assistant:[/bold blue]")
            # Render as markdown for better formatting
            console.print(Markdown(response))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Try again or type /quit to exit.[/dim]")

def validate_config():
    """Validate configuration before starting."""
    errors = []
    
    if settings.llm_provider == LLMProvider.OPENAI and not settings.openai_api_key:
        errors.append("OPENAI_API_KEY is required when using OpenAI provider")
    
    if settings.llm_provider == LLMProvider.ANTHROPIC and not settings.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY is required when using Anthropic provider")
    
    if errors:
        console.print("[bold red]Configuration Errors:[/bold red]")
        for error in errors:
            console.print(f"  [red]• {error}[/red]")
        console.print()
        console.print("[dim]Set the required environment variables or create a .env file.[/dim]")
        return False
    
    return True

def main():
    """Main entry point."""
    print_welcome()
    
    # Validate configuration
    if not validate_config():
        sys.exit(1)
    
    # Get username
    username = Prompt.ask("[cyan]Enter your username[/cyan]", default="user")
    
    # Initialize conversation manager
    try:
        manager = ConversationManager(settings=settings)
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    console.print()
    
    # Run chat loop
    run_chat_loop(manager, username)


if __name__ == "__main__":
    main()
