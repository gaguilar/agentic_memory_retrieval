import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from rich.panel import Panel
from conversation.manager import ConversationManager
from rich.console import Console
from rich.text import Text

console = Console()

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

def print_memories(manager: ConversationManager):
    """Print extracted memories from past conversations."""
    memories = manager.get_memories()
    
    if not memories:
        console.print("[dim]No extracted memories yet. Complete a conversation to extract memories.[/dim]")
        return
    
    console.print(Panel.fit("[bold]Extracted Memories[/bold]", border_style="magenta"))
    
    for memory in memories:
        type_colors = {
            "semantic": "cyan",
            "episodic": "yellow",
            "procedural": "green",
            "affective": "magenta"
        }
        color = type_colors.get(memory.memory_type, "white")
        timestamp = memory.created_at.strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"\n[{color}][{memory.memory_type.upper()}][/{color}] {memory.statement}")
        console.print(f"  [dim]Extracted: {timestamp}[/dim]")

def print_active_memories(manager: ConversationManager):
    """Print the memories that were used in the last response, subtly."""
    memories = manager.last_used_memories
    if not memories:
        return
    
    type_icons = {
        "semantic": "S",
        "episodic": "E",
        "procedural": "P",
        "affective": "A",
    }
    
    lines = []
    for m in memories:
        icon = type_icons.get(m.memory_type, "?")
        tags = m.get_tags()
        tag_str = f" \[{', '.join(tags)}]" if tags else ""
        created = m.created_at.strftime("%Y-%m-%d %H:%M:%S")
        
        # Build updated timestamp suffix if it differs from created
        updated_str = ""
        if m.updated_at and m.updated_at != m.created_at:
            delta = abs((m.updated_at - m.created_at).total_seconds())
            if delta > 1:
                updated = m.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                updated_str = f"  (updated {updated})"
        
        lines.append(f"  \[{created}] \[{icon}] {m.statement}{tag_str}{updated_str}")
    
    text = "\n".join(lines)
    console.print(f"[dim italic]memories ({len(memories)}):[/dim italic]")
    console.print(f"[dim]{text}[/dim]")

def print_created_edges(manager: ConversationManager):
    """Print memory graph edges created during session end."""
    edges = manager.last_created_edges
    if not edges:
        return

    lines = []
    for e in edges:
        lines.append(f"[bold cyan]Edge:[/bold cyan] {e['label']}")
        lines.append(f"[bold cyan]Source: {e['source_id']}[/bold cyan] ({e.get['source_type']}): {e['source_statement']}")
        lines.append(f"[bold cyan]Target: {e['target_id']}[/bold cyan] ({e.get['target_type']}): {e['target_statement']}")
        lines.append(f"[bold cyan]Reason:[/bold cyan] {e['reasoning']}")
        lines.append("")

    content = "\n".join(lines).rstrip()
    panel_title = f"Memory Graph Edges ({len(edges)})"
    console.print()
    console.print(Panel(content, title=panel_title, title_align="left", border_style="dim"))

def print_graph(manager: ConversationManager):
    """Visualize the user's memory graph as a textual canvas in the CLI."""
    user = manager.current_user
    if not user:
        console.print("[dim]No active user session.[/dim]")
        return

    # Fetch all active memories for this user
    memories = manager.repository.get_user_active_memories(user.id, limit=None)
    if not memories:
        console.print("[dim]No memories to visualize.[/dim]")
        return

    mem_map = {m.id: m for m in memories}
    memory_ids = list(mem_map.keys())
    edges = manager.repository.get_edges_for_memories(memory_ids)

    # Filter edges to only those whose both endpoints are active
    edges = [e for e in edges if e.source_memory_id in mem_map and e.target_memory_id in mem_map]

    if not edges:
        console.print(f"[dim]{len(memories)} memories exist but none are connected by edges.[/dim]")
        console.print("[dim]Edges are created when RAG_MODE=graph is enabled and conversations end.[/dim]")
        return

    # Identify connected vs isolated memories
    connected_ids = set()
    for e in edges:
        connected_ids.add(e.source_memory_id)
        connected_ids.add(e.target_memory_id)
    isolated = [m for m in memories if m.id not in connected_ids]

    # Build networkx graph
    G = nx.Graph()
    for nid in connected_ids:
        G.add_node(nid)
    for e in edges:
        G.add_edge(e.source_memory_id, e.target_memory_id, label=e.label)

    if not G.nodes():
        console.print("[dim]No valid edges to display.[/dim]")
        return

    # Compute force-directed layout
    if len(G.nodes()) == 1:
        pos = {list(G.nodes())[0]: (0.0, 0.0)}
    elif len(G.nodes()) <= 3:
        pos = nx.spring_layout(G, seed=42, k=3.0, iterations=100)
    else:
        pos = nx.spring_layout(
            G, seed=42,
            k=2.0 / max(1, len(G.nodes()) ** 0.3),
            iterations=100,
        )

    # Style config
    type_styles = {
        "semantic": "bold cyan",
        "episodic": "bold yellow",
        "procedural": "bold green",
        "affective": "bold magenta",
    }
    type_icons = {
        "semantic": "S", "episodic": "E",
        "procedural": "P", "affective": "A",
    }

    # Canvas dimensions — nodes are just their ID number, so labels are compact
    term_width = console.width or 120
    canvas_w = min(term_width - 6, 120)
    canvas_h = max(15, min(45, len(connected_ids) * 3))
    margin_x = 5
    margin_y = 1

    # Scale layout positions to canvas coordinates
    if len(pos) == 1:
        node = list(pos.keys())[0]
        scaled = {node: (canvas_w // 2, canvas_h // 2)}
    else:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = (max_x - min_x) or 1.0
        range_y = (max_y - min_y) or 1.0

        scaled = {}
        for node, (x, y) in pos.items():
            sx = margin_x + int((x - min_x) / range_x * (canvas_w - 2 * margin_x))
            sy = margin_y + int((1.0 - (y - min_y) / range_y) * (canvas_h - 2 * margin_y))
            scaled[node] = (
                max(margin_x, min(canvas_w - margin_x - 1, sx)),
                max(margin_y, min(canvas_h - margin_y - 1, sy)),
            )

    # Build compact node labels (just the ID number) and bounding boxes
    node_labels = {}   # nid -> label string
    node_bboxes = {}   # nid -> (x_start, x_end, y)
    for nid, (cx, cy) in scaled.items():
        label = str(nid)
        x_start = max(0, cx - len(label) // 2)
        if x_start + len(label) > canvas_w:
            x_start = canvas_w - len(label)
        node_labels[nid] = label
        node_bboxes[nid] = (x_start, x_start + len(label), cy)

    # Resolve overlapping labels (greedy vertical shift)
    sorted_nodes = sorted(scaled.keys(), key=lambda n: (scaled[n][1], scaled[n][0]))
    for idx in range(len(sorted_nodes)):
        nid = sorted_nodes[idx]
        cx, cy = scaled[nid]
        xs_n, xe_n, _ = node_bboxes[nid]
        attempt_y = cy
        for _ in range(8):
            overlap = False
            for jdx in range(idx):
                oid = sorted_nodes[jdx]
                xs_o, xe_o, oy = node_bboxes[oid]
                if abs(attempt_y - oy) < 2 and not (xs_n >= xe_o + 1 or xe_n <= xs_o - 1):
                    overlap = True
                    break
            if not overlap:
                break
            attempt_y += 2
            if attempt_y >= canvas_h - margin_y:
                attempt_y = max(margin_y, cy - 2)
                break
        scaled[nid] = (cx, attempt_y)
        node_bboxes[nid] = (xs_n, xe_n, attempt_y)

    # Create canvas: each cell = (char, style)
    canvas = [[(" ", "")] * canvas_w for _ in range(canvas_h)]

    def in_any_node_bbox(x, y):
        for xs, xe, ny in node_bboxes.values():
            if y == ny and xs <= x < xe:
                return True
        return False

    # Draw edges (Bresenham's line algorithm)
    def draw_line(x1, y1, x2, y2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx_step = 1 if x1 < x2 else -1
        sy_step = 1 if y1 < y2 else -1
        err = dx - dy
        x, y = x1, y1
        while True:
            if 0 <= x < canvas_w and 0 <= y < canvas_h:
                if not in_any_node_bbox(x, y) and canvas[y][x][0] == " ":
                    canvas[y][x] = ("·", "dim")
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx_step
            if e2 < dx:
                err += dx
                y += sy_step

    for e in edges:
        src, tgt = e.source_memory_id, e.target_memory_id
        if src in scaled and tgt in scaled:
            draw_line(scaled[src][0], scaled[src][1], scaled[tgt][0], scaled[tgt][1])

    # Draw nodes (overlay edges) — colored by memory type
    for nid, (cx, cy) in scaled.items():
        label = node_labels[nid]
        style = type_styles.get(mem_map[nid].memory_type, "bold white")
        x_start = node_bboxes[nid][0]
        for i, ch in enumerate(label):
            px = x_start + i
            if 0 <= px < canvas_w and 0 <= cy < canvas_h:
                canvas[cy][px] = (ch, style)

    # Build Rich Text from the canvas
    output = Text()
    for y in range(canvas_h):
        for x in range(canvas_w):
            ch, style = canvas[y][x]
            output.append(ch, style=style or None)
        if y < canvas_h - 1:
            output.append("\n")

    console.print()
    console.print(Panel(
        output,
        title=f"Memory Graph ({len(connected_ids)} nodes, {len(edges)} edges)",
        border_style="cyan",
        subtitle="[dim]S=Semantic  E=Episodic  P=Procedural  A=Affective[/dim]",
    ))

    # Node legend: map ID to type + statement
    console.print()
    console.print("[bold]Nodes:[/bold]")
    type_colors = {
        "semantic": "cyan", "episodic": "yellow",
        "procedural": "green", "affective": "magenta",
    }
    for nid in sorted(connected_ids):
        mem = mem_map[nid]
        color = type_colors.get(mem.memory_type, "white")
        icon = type_icons.get(mem.memory_type, "?")
        console.print(f"  [{color}]{nid}[/{color}]  [{color}][{icon}][/{color}] {mem.statement}")

    # Edge details
    console.print()
    console.print("[bold]Edges:[/bold]")
    for i, e in enumerate(edges):
        src = mem_map.get(e.source_memory_id)
        tgt = mem_map.get(e.target_memory_id)
        if src and tgt:
            src_color = type_colors.get(src.memory_type, "white")
            tgt_color = type_colors.get(tgt.memory_type, "white")
            prefix = "└─" if i == len(edges) - 1 else "├─"
            console.print(
                f"  {prefix} [{src_color}]{e.source_memory_id}[/{src_color}] "
                f"─[cyan]{e.label}[/cyan]→ "
                f"[{tgt_color}]{e.target_memory_id}[/{tgt_color}]"
            )

    if isolated:
        console.print()
        console.print(f"[dim]{len(isolated)} memories not connected by edges (not shown)[/dim]")

def print_rag_queries(manager: ConversationManager):
    """Print RAG queries from the last inference, after the response."""
    if manager.last_semantic_query:
        console.print("[dim]Semantic retrieval query:[/dim]")
        console.print(f"   [cyan]{manager.last_semantic_query}[/cyan]")
    if manager.last_graph_queries:
        console.print("[dim italic]Graph retrieval queries:[/dim italic]")
        for i, q in enumerate(manager.last_graph_queries, 1):
            query = q.get("query", "")
            seeds = q.get("seed_types", [])
            hops = q.get("hop_types", [])
            seeds_str = f"   [dim]{seeds}[/dim]" if seeds else ""
            hops_str = f" → [dim]{hops}[/dim]" if hops else ""
            console.print(f"   [dim]query {i}:[/dim] [dim]{query}[/dim]{seeds_str}{hops_str}")

def print_user_stats(manager: ConversationManager, username: str):
    """Print user statistics and config limits at conversation start."""
    user = manager.current_user
    if not user:
        return
    
    session_count = manager.repository.count_user_conversations(user.id)
    memory_count = manager.repository.count_user_active_memories(user.id)
    
    stats_lines = [
        f"[cyan]User:[/cyan] {username}",
        f"[cyan]Past sessions:[/cyan] {session_count}",
        f"[cyan]Active memories:[/cyan] {memory_count}",
        "",
        f"[dim]Memory limit:[/dim] {manager.settings.max_memories}",
        f"[dim]History turns limit:[/dim] {manager.settings.max_conversation_turns}",
        f"[dim]Max prompt tokens:[/dim] {manager.settings.max_prompt_tokens}",
    ]
    console.print(Panel(
        "\n".join(stats_lines),
        title="User Profile",
        border_style="green"
    ))
    console.print()
