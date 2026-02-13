"""
Seed the database with conversations using an LLM as the simulated user.

Uses the same ConversationManager as the main app. Generates a persona,
varies topics by memory type (semantic, episodic, procedural, affective),
runs multiple sessions, and naturally tests the assistant's memory by having
the simulated user assume the assistant remembers previous context.

Run from project root:
  python scripts/seed_conversations.py
  python scripts/seed_conversations.py --sessions 4 --turns-per-session 6 --seed 42
"""

import argparse
import logging
import random
import sys
import os
from datetime import datetime
from typing import List, Optional

# Add src to path for imports (run from project root)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_path = os.path.join(_project_root, "src")
sys.path.insert(0, _src_path)

from utils.print_artifacts import print_active_memories, print_memories, print_rag_queries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config import settings, log_settings, LLMProvider, RAGMode
from utils.quiet import suppress_verbose_extraction_output
from utils.print_artifacts import print_created_edges
from conversation.manager import ConversationManager
from database.models import Memory, Conversation, Turn
from agents.base import Message
from agents.openai_agent import OpenAIAgent
from agents.anthropic_agent import AnthropicAgent

console = Console()

PERSONA_PROMPT_TEMPLATE = """
You are generating a brief persona for a simulated user in a chat.

Write 2-4 short sentences that define this person. Include:
{NAME_INSTRUCTION}
- One specific past event or experience you had (episodic).
- One way you usually do something—a habit or workflow (procedural).
- One clear preference, like, or dislike (affective).
- Mention that you enjoy talking about a variety of topics, not just those related to your profession.

Be specific and varied. Write in first person as if the person is describing themselves. Output only the persona text, no labels or bullet points."""

TOPIC_PROMPTS = {
    "semantic": [
        "Talk about your job or professional background.",
        "Mention your skills, expertise, or what you've been learning.",
        "Share something about your identity or background (where you're from, role in life).",
    ],
    "episodic": [
        "Mention a specific recent event or experience (e.g. last week, last month).",
        "Refer to something that happened at a particular time or place (e.g. a trip, a conference).",
        "Bring up a past experience that influenced you.",
    ],
    "procedural": [
        "Describe how you usually approach a task or project.",
        "Share your preferred workflow or method for doing something.",
        "Explain a process or habit you follow regularly.",
    ],
    "affective": [
        "Share a strong preference or dislike about something.",
        "Mention something you care about or value.",
        "Say what you enjoy or find frustrating in a certain area.",
    ],
}

USER_MESSAGE_SYSTEM_PROMPT_TEMPLATE = """
You are playing the role of a user in a chat with an AI assistant called Maya.
Stay in character, however, your messages must be short as if you are communicating over SMS text messages.

IMPORTANT INSTRUCTIONS:
- The assistant is supposed to build long-term memory about you across conversations.
- From time to time, naturally test whether the assistant remembers things you've mentioned before.
- Don't explicitly say 'Remember when...' or 'As I mentioned...'. Instead, continue the conversation naturally as if the assistant should remember.
- For example: if you mentioned a project, ask 'How do you think I should proceed with it?' without re-explaining what 'it' is.
- Or: if you mentioned a preference, make a request that assumes they remember that preference.
- The assistant should recall context WITHOUT you having to repeat yourself.

Your persona:
{PERSONA}

{PREVIOUS_TURNS}

{PREVIOUS_MEMORIES}

{MEMORY_TEST_INSTRUCTION}

For this turn: {TOPIC_INSTRUCTION}

Reply with a single short message (1-2 sentences) as this user. No meta-commentary.
"""


def get_persona_prompt(username: str) -> str:
    """Generate the persona prompt, optionally incorporating the username."""
    if username and username != "simulated_user":
        # Use the provided username as the persona's name
        name_instruction = f"- Say your name is {username}, and mention your role or job (semantic fact)."
    else:
        # Let the LLM choose a name
        name_instruction = "- A name or how they prefer to be called, and their role or job (semantic fact)."
    
    return PERSONA_PROMPT_TEMPLATE.format(NAME_INSTRUCTION=name_instruction)


def show_persona_template():
    """Display the persona prompt template with placeholders."""
    console.print("\n" + "="*60)
    console.print(Text("PERSONA GENERATION TEMPLATE", style="bold yellow"))
    console.print("="*60)
    console.print(Text("Placeholders: {NAME_INSTRUCTION}", style="dim cyan"))
    console.print("-"*60)
    console.print(PERSONA_PROMPT_TEMPLATE)
    console.print("="*60 + "\n")


def create_user_agent():
    """Create an LLM agent for the simulated user (same config as main app)."""
    if settings.llm_provider == LLMProvider.OPENAI:
        return OpenAIAgent(
            api_key=settings.get_api_key(),
            model=settings.openai_model,
        )
    elif settings.llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicAgent(
            api_key=settings.get_api_key(),
            model=settings.anthropic_model,
        )
    else:
        raise ValueError(f"Unknown provider: {settings.llm_provider}")


def generate_persona(user_agent, username: str) -> str:
    """Generate a brief persona with one LLM call."""
    persona_prompt = get_persona_prompt(username)
    messages = [Message(role="user", content=persona_prompt)]
    response = user_agent.generate(
        messages=messages,
        max_tokens=300,
        temperature=0.8,
    )
    return response.content.strip()


def confirm_or_replace_persona(persona: str) -> str:
    """Show persona to the user; return confirmed or replacement text."""
    console.print()
    console.print(Text("Generated persona:", style="bold green"))
    console.print(Text("-" * 40, style="dim"))
    console.print(Text(persona, style="green"))
    console.print(Text("-" * 40, style="dim"))
    while True:
        choice = input("Accept this persona? [y]es / [n]o to type replacement: ").strip().lower()
        if choice in ("y", "yes", ""):
            return persona
        if choice in ("n", "no"):
            console.print(Text("Enter the replacement persona (2-4 sentences). End with a blank line:", style="yellow"))
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                if line == "":
                    if lines:
                        break
                    continue
                lines.append(line)
            replacement = "\n".join(lines).strip()
            if replacement:
                return replacement
            console.print(Text("Empty input. Try again or accept with [y].", style="red"))
        else:
            console.print(Text("Please answer [y]es or [n]o.", style="red"))


def pick_topic_for_turn(session_index: int, turn_index: int, seed: Optional[int]) -> tuple:
    """Pick memory type and topic prompt for this turn. Returns (memory_type, prompt)."""
    types = list(TOPIC_PROMPTS.keys())
    if seed is not None:
        rng = random.Random(seed + session_index * 1000 + turn_index)
    else:
        rng = random
    memory_type = rng.choice(types)
    prompt = rng.choice(TOPIC_PROMPTS[memory_type])
    return memory_type, prompt


def decide_recall_turn(session_index: int, turn_index: int, turns_per_session: int, seed: Optional[int]) -> bool:
    """Decide if this turn should test the assistant's memory (relies on previous context)."""
    # Enable memory-testing in any session if there's existing data
    # The caller will check if there's existing context
    if seed is not None:
        rng = random.Random(seed + session_index * 2000 + turn_index)
    else:
        rng = random
    # 1-2 memory-testing turns per session
    n_recall = rng.randint(1, min(2, turns_per_session))
    return turn_index < n_recall


def load_existing_user_data(repository, username: str, max_turns: int):
    """Load existing conversations, memories, and recent turns for username. Returns (user_id or None, memories, turns, conversation_count)."""
    user = repository.get_user_by_username(username)
    if not user:
        return None, [], [], 0
    memories = repository.get_user_memories(user.id, limit=50)
    # Get last N turns (most recent conversations)
    turns = repository.get_user_turns(user_id=user.id, limit=max_turns)
    with repository.get_session() as session:
        conversation_count = session.query(Conversation).filter(
            Conversation.user_id == user.id
        ).count()
    return user.id, memories, turns, conversation_count


def format_turns_for_prompt(turns: List[Turn], max_display: int = 20) -> str:
    """Format recent conversation turns for the user-LLM system prompt."""
    if not turns:
        return ""
    lines = ["Recent conversation history:"]
    # Turns are ordered by turn_number; show most recent first for context
    recent_turns = turns[-max_display:] if len(turns) > max_display else turns
    for t in recent_turns:
        role_display = "You" if t.role == "user" else "Assistant"
        lines.append(f"  [{role_display}]: {t.content}")
    if len(turns) > max_display:
        lines.insert(1, f"  (showing last {max_display} of {len(turns)} turns)")
    return "\n".join(lines)


def format_memories_for_prompt(memories: List[Memory]) -> str:
    """Format extracted memories for the user-LLM system prompt."""
    if not memories:
        return ""
    lines = ["Previously in conversation, the following was established (you can reference these):"]
    for m in memories:
        lines.append(f"- [{m.memory_type}] {m.statement}")
    return "\n".join(lines)


def build_user_message_prompt(
    persona: str,
    topic_instruction: str,
    is_memory_test: bool,
    previous_memories_text: str,
    previous_turns_text: str,
    last_assistant_message: Optional[str],
) -> List[Message]:
    """Build messages for the user-LLM to generate the next user message."""
    
    # Build previous turns section
    turns_section = ""
    if previous_turns_text:
        turns_section = previous_turns_text
    
    # Build previous memories section
    memories_section = ""
    if previous_memories_text:
        memories_section = previous_memories_text
    
    # Build memory test instruction
    memory_test_instruction = ""
    if is_memory_test:
        memory_test_instruction = """FOR THIS TURN SPECIFICALLY: Assume the assistant remembers something from your conversation history.
Continue or reference a topic naturally, as if they should already know the context."""
    
    # Fill in the template
    system_content = USER_MESSAGE_SYSTEM_PROMPT_TEMPLATE.format(
        PERSONA=persona,
        PREVIOUS_TURNS=turns_section,
        PREVIOUS_MEMORIES=memories_section,
        MEMORY_TEST_INSTRUCTION=memory_test_instruction,
        TOPIC_INSTRUCTION=topic_instruction
    )
    
    messages = [Message(role="system", content=system_content)]

    if last_assistant_message:
        messages.append(Message(role="user", content=f"Assistant said:\n{last_assistant_message}\n\nYour reply:"))
    else:
        messages.append(Message(role="user", content="Start the conversation. Your first message:"))

    return messages


def show_user_message_template():
    """Display the user message system prompt template with placeholders."""
    console.print("\n" + "="*60)
    console.print(Text("USER MESSAGE SYSTEM PROMPT TEMPLATE", style="bold yellow"))
    console.print("="*60)
    console.print(Text("Placeholders: {PERSONA}, {PREVIOUS_TURNS}, {PREVIOUS_MEMORIES}, {MEMORY_TEST_INSTRUCTION}, {TOPIC_INSTRUCTION}", style="dim cyan"))
    console.print("-"*60)
    console.print(USER_MESSAGE_SYSTEM_PROMPT_TEMPLATE)
    console.print("="*60 + "\n")


def generate_user_message(user_agent, messages: List[Message]) -> str:
    """Get one user message from the user LLM."""
    response = user_agent.generate(
        messages=messages,
        max_tokens=200,
        temperature=0.7,
    )
    return response.content.strip()


def validate_config() -> bool:
    """Validate configuration (same as main.py)."""
    if settings.llm_provider == LLMProvider.OPENAI and not settings.openai_api_key:
        return False
    if settings.llm_provider == LLMProvider.ANTHROPIC and not settings.anthropic_api_key:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Seed the database with LLM-simulated user conversations for memory extraction testing.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=4,
        help="Number of conversation sessions to run (default: 4).",
    )
    parser.add_argument(
        "--turns-per-session",
        type=int,
        default=6,
        help="Number of user-assistant turns per session (default: 6).",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="simulated_user",
        help="Username for the simulated user (default: simulated_user).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for topic/memory-test selection (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and log what would be done, without calling the assistant or DB.",
    )
    parser.add_argument(
        "--show-templates",
        action="store_true",
        help="Display the full prompt templates with placeholders and exit.",
    )
    parser.add_argument(
        "--rag-mode",
        choices=["none", "semantic", "graph"],
        default=None,
        help="Override RAG_MODE from .env (none, semantic, or graph).",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="Override LLM_PROVIDER from .env.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model (e.g. gpt-4o, claude-3-5-sonnet-20241022).",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Override database URL (e.g. sqlite:///conversations.db).",
    )
    args = parser.parse_args()

    # Override settings if CLI args provided
    if args.rag_mode:
        settings.rag_mode = RAGMode(args.rag_mode)
    if args.provider:
        settings.llm_provider = LLMProvider(args.provider)
    if args.model:
        if settings.llm_provider == LLMProvider.OPENAI:
            settings.openai_model = args.model
        else:
            settings.anthropic_model = args.model
    if args.database:
        settings.database_url = args.database

    log_settings(settings)

    if args.show_templates:
        show_persona_template()
        show_user_message_template()
        return

    if args.seed is not None:
        random.seed(args.seed)

    console.print(Text("Seed conversations script", style="bold blue"))
    config_lines = [
        f"[cyan]provider:[/cyan] {settings.llm_provider.value}",
        f"[cyan]model:[/cyan] {settings.get_model()}",
        f"[cyan]rag_mode:[/cyan] {settings.rag_mode.value}",
        f"[cyan]database:[/cyan] {settings.database_url}",
    ]
    console.print(Panel("\n".join(config_lines), title="Settings", border_style="dim"))
    if not validate_config():
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY (and use matching LLM_PROVIDER).", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("Dry run: no LLM or DB calls.")
        persona_placeholder = "(Persona would be generated here.)"
        print("Persona: %s" % persona_placeholder)
        for session in range(args.sessions):
            print("Session %d" % (session + 1))
            for turn in range(args.turns_per_session):
                mem_type, prompt = pick_topic_for_turn(session, turn, args.seed)
                memory_test = decide_recall_turn(session, turn, args.turns_per_session, args.seed)
                print("  turn %d [%s] memory-test=%s: %s" % (
                    turn + 1, mem_type, memory_test, prompt[:50] + "..." if len(prompt) > 50 else prompt
                ))
            print("  -> would call end_conversation() and extract memories")
        return

    manager = ConversationManager(settings=settings)
    user_agent = create_user_agent()
    repo = manager.repository

    # By default: pull existing conversations/memories for this username
    _, existing_memories, existing_turns, existing_conv_count = load_existing_user_data(
        repo, args.username, max_turns=settings.max_conversation_turns
    )
    if existing_conv_count or existing_memories or existing_turns:
        console.print(
            Text(
                f"User '{args.username}' has {existing_conv_count} existing conversation(s), "
                f"{len(existing_memories)} memory(ies), and {len(existing_turns)} recent turn(s). "
                "They will be used as context from the first turn.",
                style="dim cyan"
            )
        )
    else:
        console.print(Text(f"No existing data for user '{args.username}'. Starting fresh.", style="dim"))
    console.print()

    persona_prompt = get_persona_prompt(args.username)
    console.print(Text("Persona generation prompt (what we send to the LLM):", style="bold"))
    console.print(Text("-" * 40, style="dim"))
    console.print(persona_prompt)
    console.print(Text("-" * 40, style="dim"))
    console.print(Text("Generating persona...", style="yellow"))
    persona = generate_persona(user_agent, args.username)
    
    # Auto-accept persona if seed is provided (for automated/reproducible runs)
    if args.seed is not None:
        console.print()
        console.print(Text("Generated persona (auto-accepted with --seed):", style="bold green"))
        console.print(Text("-" * 40, style="dim"))
        console.print(Text(persona, style="green"))
        console.print(Text("-" * 40, style="dim"))
    else:
        persona = confirm_or_replace_persona(persona)
    console.print()

    for session_index in range(args.sessions):
        console.print(Text(f"Session {session_index + 1}/{args.sessions}", style="bold yellow"))
        manager.start_conversation(args.username)

        # Pull all memories and turns for this user (existing + extracted/created in earlier sessions this run)
        memories_so_far = manager.get_memories()
        turns_so_far = manager.repository.get_user_turns(
            user_id=manager.current_user.id, limit=100
        )
        
        previous_memories_text = format_memories_for_prompt(memories_so_far[:20])
        if len(memories_so_far) > 20:
            previous_memories_text += "\n(… and %d more.)" % (len(memories_so_far) - 20)
        
        previous_turns_text = format_turns_for_prompt(turns_so_far, max_display=20)

        last_assistant_message = None

        for turn_index in range(args.turns_per_session):
            memory_type, topic_instruction = pick_topic_for_turn(
                session_index, turn_index, args.seed
            )
            # Only enable memory-testing if there's context (memories or turns)
            has_context = len(memories_so_far) > 0 or len(turns_so_far) > 0
            is_memory_test = has_context and decide_recall_turn(
                session_index, turn_index, args.turns_per_session, args.seed
            )

            messages = build_user_message_prompt(
                persona=persona,
                topic_instruction=topic_instruction,
                is_memory_test=is_memory_test,
                previous_memories_text=previous_memories_text,
                previous_turns_text=previous_turns_text,
                last_assistant_message=last_assistant_message,
            )
            turn_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with suppress_verbose_extraction_output():
                user_message = generate_user_message(user_agent, messages)
                assistant_response = manager.send_message(user_message)
            last_assistant_message = assistant_response

            mem_test_flag = "yes" if is_memory_test else "no"
            panel_title = f"turn {turn_index + 1} [{memory_type}] memory-test={mem_test_flag}  rag={settings.rag_mode.value}  {turn_ts}"
            user_line = "[bold green]User:[/bold green] " + " ".join(user_message.split())
            assistant_line = "[bold steel_blue1]Assistant:[/bold steel_blue1] " + " ".join(assistant_response.split())
            console.print(Panel(f"{user_line}\n{assistant_line}", title=panel_title, title_align="left", border_style="dim"))
            print_rag_queries(manager)
            print_active_memories(manager)
            console.print()  # Blank line between turns

        with suppress_verbose_extraction_output():
            extracted = manager.end_conversation()
        console.print(Text(f"  → extracted {len(extracted)} memories", style="magenta bold"))
        print_memories(manager)
        print_created_edges(manager)
        console.print()

    console.print(Text(f"Done. User '{args.username}' has {args.sessions} sessions of conversations and extracted memories.", style="bold green"))


if __name__ == "__main__":
    main()
