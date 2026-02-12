"""
CLI entry point for the conversational agent system.

Provides an interactive command-line interface for chatting with the AI agent.
Designed to be modular for future web interface adaptation.
"""

import argparse
import sys
import os
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config import settings, log_settings, LLMProvider, RAGMode
from conversation.manager import ConversationManager
from utils.quiet import suppress_verbose_extraction_output
from utils.print_artifacts import (
    print_history, 
    print_memories, 
    print_active_memories, 
    print_created_edges, 
    print_graph, 
    print_rag_queries, 
    print_user_stats,
)

console = Console()

help_text = """
[bold]Available Commands:[/bold]

  [cyan]/quit, /exit[/cyan] - Exit the application (extracts memories from conversation)
  [cyan]/new[/cyan]         - Start a new conversation (extracts memories from current)
  [cyan]/history[/cyan]     - Show conversation history
  [cyan]/memories[/cyan]    - Show extracted memories from past conversations
  [cyan]/graph[/cyan]       - Visualize the memory graph connections
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
        f"[cyan]Max Memories:[/cyan] {settings.max_memories}",
        f"[cyan]Max Prompt Tokens:[/cyan] {settings.max_prompt_tokens}",
        f"[cyan]Max Output Tokens:[/cyan] {settings.max_response_tokens}",
    ]
    console.print(Panel(
        "\n".join(config_info),
        title="Configuration",
        border_style="dim"
    ))
    console.print()
    console.print("[dim]Commands: /quit, /exit, /history, /memories, /graph, /new, /help[/dim]")
    console.print()

def print_help():
    """Print help message."""
    console.print(Panel(help_text.strip(), title="Help", border_style="green"))

def run_chat_loop(manager: ConversationManager, username: str):
    """
    Run the main chat loop.
    
    Args:
        manager: ConversationManager instance
        username: Current user's username
    """
    # Start a new conversation
    manager.start_conversation(username)
    print_user_stats(manager, username)
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
                # End conversation and extract memories
                with suppress_verbose_extraction_output():
                    with console.status("[bold magenta]Extracting memories from conversation...[/bold magenta]"):
                        try:
                            extracted = manager.end_conversation()
                            if extracted:
                                console.print(f"[magenta]Extracted {len(extracted)} memories from this conversation.[/magenta]")
                            else:
                                console.print("[dim]No significant memories to extract.[/dim]")
                            print_created_edges(manager)
                        except ValueError:
                            # No active conversation
                            pass
                console.print("[yellow]Conversation ended[/yellow]")
                break
            
            if command == "/help":
                print_help()
                continue
            
            if command == "/history":
                print_history(manager)
                continue
            
            if command == "/memories":
                print_memories(manager)
                continue
            
            if command == "/graph":
                print_graph(manager)
                continue
            
            if command == "/new":
                # End current conversation with memory extraction
                with suppress_verbose_extraction_output():
                    with console.status("[bold magenta]Extracting memories from conversation...[/bold magenta]"):
                        try:
                            extracted = manager.end_conversation()
                            if extracted:
                                console.print(f"[magenta]Extracted {len(extracted)} memories from previous conversation.[/magenta]")
                            print_created_edges(manager)
                        except ValueError:
                            pass
                # Start new conversation
                manager.start_conversation(username)
                console.print("[green]Started new conversation.[/green]")
                console.print()
                continue
            
            # Timestamp the user message
            user_ts = datetime.now().strftime("%H:%M:%S")
            console.print(f"[dim]  {user_ts}[/dim]")
            
            
            # Send message and get response
            with suppress_verbose_extraction_output():
                with console.status("[bold blue]Thinking...[/bold blue]"):
                    response = manager.send_message(user_input)
            
            # Display response with timestamp
            assistant_ts = datetime.now().strftime("%H:%M:%S")
            console.print()
            
            # Display RAG queries after response
            print_rag_queries(manager)
            
            # Show memories used in this response (subtle)
            print_active_memories(manager)

            console.print()
            console.print(f"[bold blue]Assistant:[/bold blue] {response}")  # TODO: Render as markdown for better formatting
            console.print(f"[dim]  {assistant_ts}[/dim]")
            
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
    parser = argparse.ArgumentParser(
        description="AI Conversational Agent with long-term memory."
    )
    parser.add_argument(
        "--rag-mode",
        choices=["none", "semantic", "graph"],
        default=None,
        help="Override RAG_MODE from .env (none, semantic, or graph)."
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Username to use (skips interactive prompt)."
    )
    args = parser.parse_args()
    
    # Override settings if CLI args provided
    if args.rag_mode:
        settings.rag_mode = RAGMode(args.rag_mode)
    
    log_settings(settings)
    print_welcome()
    
    # Validate configuration
    if not validate_config():
        sys.exit(1)
    
    # Get username
    if args.username:
        username = args.username
    else:
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
