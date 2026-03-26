"""
LLM prompt templates for PageIndex tree building and retrieval.

All prompt strings used by ``indexer_pageindex.py`` and ``retriever_pageindex.py``
are centralised here as module-level constants.  No prompt text should appear
inside the indexer or retriever modules themselves.
"""

# ── Indexing prompts (used by PageIndexAdapter) ───────────────────────────────

TOPIC_SEGMENTATION_PROMPT = """\
# Expected JSON output: list of topic dicts
# Schema: [{{"start_segment": int, "end_segment": int, "topic_title": str}}, ...]
# Parsed by: indexer_pageindex.PageIndexAdapter._segment_topics()
You are analyzing a meeting/conversation transcript.  Identify 4-7 major topic
shift boundaries.  Each topic block must be a contiguous range of segments.

Segments:
{segments}

Return ONLY valid JSON.  No markdown fences, no explanation, no text before or
after the JSON.

Return a JSON array where each element has:
- "start_segment": int (0-based index of the first segment in this topic)
- "end_segment": int (0-based index of the last segment in this topic, inclusive)
- "topic_title": str (short descriptive title for the topic)

Every segment index from 0 to {max_segment} must be covered exactly once.
[{{"start_segment": 0, "end_segment": ..., "topic_title": "..."}}]
"""

TOPIC_SEGMENTATION_RETRY_PROMPT = """\
# Expected JSON output: same as TOPIC_SEGMENTATION_PROMPT (retry on parse failure)
# Schema: [{{"start_segment": int, "end_segment": int, "topic_title": str}}, ...]
# Parsed by: indexer_pageindex.PageIndexAdapter._segment_topics()
Your previous response was not valid JSON.  Please try again.

You are analyzing a meeting/conversation transcript.  Identify 4-7 major topic
shift boundaries.  Each topic block must be a contiguous range of segments.

Segments:
{segments}

CRITICAL: Return ONLY a valid JSON array.  Do NOT wrap it in markdown code
fences.  Do NOT include any explanation or text outside the JSON.

Every segment index from 0 to {max_segment} must be covered exactly once.
[{{"start_segment": 0, "end_segment": ..., "topic_title": "..."}}]
"""

SUBTOPIC_SEGMENTATION_PROMPT = """\
# Expected JSON output: list of subtopic dicts within a single topic block
# Schema: [{{"start_segment": int, "end_segment": int, "title": str, "speakers": [str, ...]}}, ...]
# Parsed by: indexer_pageindex.PageIndexAdapter._segment_subtopics()
You are analyzing a section of a meeting transcript (one topic block).
Split the segments below into 2-4 sub-clusters grouped by speaker turns or
sub-topic shifts.

Segments:
{segments}

Return ONLY valid JSON.  No markdown fences, no explanation, no text before or
after the JSON.

Return a JSON array where each element has:
- "start_segment": int (0-based index within the ORIGINAL transcript)
- "end_segment": int (inclusive)
- "title": str (e.g. "SPEAKER_XX: description of what they discuss")
- "speakers": list of speaker labels that appear in this sub-cluster

[{{"start_segment": ..., "end_segment": ..., "title": "...", "speakers": [...]}}]
"""

NODE_SUMMARY_PROMPT = """\
# Expected JSON output: single summary object for one tree node
# Schema: {{"summary": str, "decision": str|null, "action_items": [str], "owner": str|null, "keywords": [str]}}
# Parsed by: indexer_pageindex.PageIndexAdapter._summarize_node()
Summarize the following meeting transcript excerpt.

{segments_text}

Return ONLY valid JSON.  No markdown fences, no explanation, no text before or
after the JSON.

Return a JSON object with these fields:
- "summary": str (2-3 sentence summary of this section)
- "decision": str or null (any decision made, or null if none)
- "action_items": list of str (action items mentioned, empty list if none)
- "owner": str or null (person responsible for action items, or null)
- "keywords": list of str (3-6 key terms or phrases)

{{"summary": "...", "decision": null, "action_items": [], "owner": null, "keywords": [...]}}
"""

ROOT_SYNTHESIS_PROMPT = """\
# Expected JSON output: root node title and overall meeting summary
# Schema: {{"title": str, "summary": str}}
# Parsed by: indexer_pageindex.PageIndexAdapter._synthesize_root()
You are given the topic summaries of a meeting.  Produce a concise overall
title and a 2-3 sentence summary of the entire meeting.

Topic summaries:
{topic_summaries}

Return ONLY valid JSON.  No markdown fences, no explanation, no text before or
after the JSON.

{{"title": "...", "summary": "..."}}
"""

# ── Retrieval prompts (used by PageIndexRetriever) ────────────────────────────

TREE_SEARCH_PROMPT = """\
# Expected JSON output: selected node IDs to fetch for the user's query
# Schema: {{"thinking": str, "node_ids": [str, ...]}}  (node_ids like "T0", "T0-S1")
# Parsed by: retriever_pageindex.PageIndexRetriever.search()
You are given a user query and a tree structure of a meeting transcript.
Select 1-3 node IDs most likely to contain the answer.

Query: {query}

{chat_history_block}

Meeting tree (node_id — title — summary):
{tree_summary}

Return ONLY valid JSON.  No markdown fences, no explanation, no text before or
after the JSON.

{{"thinking": "brief reasoning", "node_ids": ["T0", "T0-S1"]}}
"""

# ── Vector retrieval level selection prompt ───────────────────────────────────

RETRIEVAL_LEVEL_SELECTION_PROMPT = """\
Choose the retrieval granularity for this question. Reply with exactly one word.

Rules:
- chunk    — specific facts, quotes, narrow topics, precise detail questions
- segment  — speaker-specific questions ("what did X say about…"), per-turn context
- document — broad summaries, overviews, "what was discussed", whole-meeting questions

Reply with only the single word (chunk, segment, or document). No punctuation, no explanation."""

SUFFICIENCY_CHECK_PROMPT = """\
# Expected JSON output: sufficiency verdict; may include additional node_ids to fetch
# Schema (sufficient): {{"sufficient": true, "reasoning": str}}
# Schema (needs more): {{"sufficient": false, "node_ids": [str, ...], "reasoning": str}}
# Parsed by: retriever_pageindex.PageIndexRetriever.search()
You retrieved the content below to answer the user's query.
Decide if the retrieved content is sufficient to answer the query fully.

Query: {query}

{chat_history_block}

Previously selected nodes: {previously_selected_nodes}

Retrieved content:
{retrieved_content}

Return ONLY valid JSON.  No markdown fences, no explanation, no text before or
after the JSON.

If sufficient, return:
{{"sufficient": true, "reasoning": "..."}}

If insufficient and you need more context, return:
{{"sufficient": false, "node_ids": ["T1-S0"], "reasoning": "need more context about ..."}}
"""
