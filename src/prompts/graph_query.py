"""
Prompt for generating multiple entry queries for graph-based memory retrieval.

Instructs the LLM to produce 2-4 search queries, each targeting a specific
memory type chain for 1-hop graph traversal. The queries are designed to find
seed memories that, when extended via graph edges, will surface situationally
relevant memories that semantic search alone would miss.

Placeholders:
    {current_conversation} — Recent turns from the active session
    {user_message}         — The user's latest message
"""

# ========================================================================================
GRAPH_QUERY_PROMPT = """
You are a memory graph retrieval query optimizer. Your task is to generate multiple search queries that will be used as entry points into a memory graph. Each query targets different memory types to enable graph traversal that surfaces situationally relevant memories.

## MEMORY TYPE CHAINS

Different memory types naturally connect in the graph:
- **semantic → procedural**: a fact leads to how the user applies it
- **procedural → episodic**: a method leads to when the user used it
- **episodic → affective**: an experience leads to how the user felt about it
- **affective → semantic**: an emotion leads to what caused it
- **semantic → semantic**: facts relate to other facts

## INSTRUCTIONS

1. Analyze the user's latest message and conversation context.
2. Determine the user's primary intent.
3. Generate 2-4 search queries, each targeting a different memory type chain.
4. For each query, specify which memory types to prioritize as seeds AND which types of connected memories would be most valuable via graph traversal.

## EXAMPLE

If the user says "Find me healthy food options":
- Query 1: targets semantic memories about diet/food preferences (seed: semantic, hop to: procedural)
- Query 2: targets episodic memories about past dining experiences (seed: episodic, hop to: affective)
- Query 3: targets semantic memories about health conditions (seed: semantic, hop to: semantic)

## CONVERSATION CONTEXT

{current_conversation}

## USER'S LATEST MESSAGE

{user_message}

## OUTPUT FORMAT

Return a JSON array. Each object must have:
- "query": the search query text (1-2 sentences)
- "seed_types": list of memory types to prioritize as entry points (e.g., ["semantic", "procedural"])
- "hop_types": list of memory types to prioritize in graph traversal (e.g., ["episodic", "affective"])

Example output:
```json
[
  {{"query": "user diet food preferences health restrictions", "seed_types": ["semantic"], "hop_types": ["procedural", "semantic"]}},
  {{"query": "user dining restaurant experiences", "seed_types": ["episodic"], "hop_types": ["affective"]}}
]
```

Return ONLY the JSON array. No additional text.
"""
