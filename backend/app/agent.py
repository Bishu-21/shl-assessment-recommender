"""
Agent: orchestrates intent detection → retrieval → Gemini reasoning → response.

This module is the brain of the system. It:
1. Classifies intent deterministically + LLM fallback
2. Retrieves catalog items via hybrid BM25 + heuristic search
3. Calls Gemini for natural language generation (constrained to catalog)
4. Assembles schema-safe responses in Python (never trusts raw LLM JSON for URLs)
"""

from __future__ import annotations

import json
import os
import re
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from google import genai
from google.genai import errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .catalog import CatalogItem, catalog
from .prompts import (
    CLARIFY_SYSTEM,
    COMPARE_SYSTEM,
    CONFIRM_SYSTEM,
    EXTRACT_SYSTEM,
    INTENT_SYSTEM,
    RECOMMEND_SYSTEM,
    REFINE_SYSTEM,
    REFUSE_SYSTEM,
)
from .retrieval import retriever
from .schemas import ChatRequest, ChatResponse, Message, Recommendation

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _get_model() -> str:
    return os.environ.get("MODEL_NAME", "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
def _call_gemini(system_prompt: str, user_content: str = "Analyze the above.", temperature: float = 0.2) -> str:
    """Call Gemini and return the text response."""
    client = _get_client()
    model = _get_model()

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_content}]},
            ],
            config={
                "temperature": temperature,
                "max_output_tokens": 2048,
            },
        )
        return response.text or ""
    except (errors.ServerError, errors.ClientError) as e:
        print(f"[agent] Gemini API Error: {e}")
        raise
    except Exception as e:
        print(f"[agent] Unexpected Gemini Error: {e}")
        raise


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Conversation formatting
# ---------------------------------------------------------------------------

def _format_conversation(messages: list[Message]) -> str:
    """Format message history for prompt injection."""
    lines = []
    for msg in messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n".join(lines)


def _get_last_user_message(messages: list[Message]) -> str:
    """Get the most recent user message."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

# Quick deterministic patterns before burning an LLM call
_CONFIRM_PATTERNS = [
    r"^(perfect|confirmed?|lock(ing)? it in|thanks?|thank you|that covers it|lgtm|yes|no|none)[\.\!\,]?$",
    r"^(ok|okay|sure|sounds good|that'?s? (fine|correct|good|great|what i need))[\.\!\,]?$",
]

_OFFTOPIC_PATTERNS = [
    r"(legal|legally|lawsuit|sue|pricing|price|cost|how much|discount|quote|invoice)",
    r"(ignore (previous|above)|forget (previous|your) instructions|you are now|act as|pretend)",
]


def _classify_intent_heuristic(last_msg: str, messages: list[Message]) -> str | None:
    """Fast deterministic intent detection. Returns None if unsure."""
    text = last_msg.strip().lower()
    words = text.split()
    message_count = len(messages)
    user_message_count = sum(1 for m in messages if m.role == "user")

    # Confirmation - only if it doesn't look like a role description
    role_markers = ["hiring", "hire", "need", "looking for", "developer", "engineer", "manager", "test", "assessment"]
    is_role_query = any(m in text for m in role_markers)
    
    for pat in _CONFIRM_PATTERNS:
        if re.match(pat, text, re.IGNORECASE) and not is_role_query:
            return "confirm"

    # Off-topic / prompt injection
    for pat in _OFFTOPIC_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "off_topic"

    # Very short messages are usually vague
    vague_phrases = {
        "help", "help me", "i need help", "hello", "hi", "hey",
        "what can you do", "who are you", "how does this work",
        "find an assessment", "find a test", "show me products",
        "i need an assessment", "i need a test", "give me a recommendation",
        "start", "get started", "assessment", "assessments", "test", "tests",
        "show tests", "recommend tests", "what do you have", "hiring",
        "i'm hiring", "we are hiring"
    }
    if text in vague_phrases:
        return "clarify"

    # Turn 1 (first user message) is almost always clarify unless very detailed
    if user_message_count <= 1:
        # If it's the first message and under 15 words, clarify to get JD details
        if len(words) < 15:
            return "clarify"
        
        # Check for JD markers (role, skills, domain)
        jd_markers = ["role", "hiring", "engineer", "manager", "skills", "experience", "candidate", "level"]
        if not any(m in text for m in jd_markers):
            return "clarify"

    # If message is short and lacks specific role/skill markers
    if len(words) <= 7:
        vague_indicators = {
            "test", "tests", "assessment", "assessments", "help", "need",
            "hiring", "hire", "shl", "product", "products", "catalog",
            "looking", "want", "show", "me", "find", "give", "recommender",
            "please", "can", "you", "something", "any", "which", "available",
            "for", "the", "role", "position", "staff", "candidates"
        }
        if all(w in vague_indicators or len(w) <= 3 for w in words):
            return "clarify"

    return None


def _classify_intent_llm(messages: list[Message]) -> str:
    """Use Gemini to classify intent."""
    conversation = _format_conversation(messages)
    prompt = INTENT_SYSTEM
    response_text = _call_gemini(prompt, conversation)
    parsed = _parse_json_response(response_text)
    intent = parsed.get("intent", "recommend")

    valid_intents = {"clarify", "recommend", "refine", "compare", "confirm", "off_topic"}
    if intent not in valid_intents:
        intent = "recommend"

    return intent


def classify_intent(messages: list[Message]) -> str:
    """Classify the user's intent — heuristic first, LLM fallback."""
    last_msg = _get_last_user_message(messages)

    # Try heuristic first
    heuristic = _classify_intent_heuristic(last_msg, messages)
    if heuristic:
        return heuristic

    # Check if there are previous recommendations in the conversation
    has_prior_recs = any(
        msg.role == "assistant" and ("http" in msg.content or "shl.com" in msg.content)
        for msg in messages[:-1]
    )

    # If the last message references adding/removing/dropping/swapping
    refine_words = ["add", "remove", "drop", "swap", "replace", "include", "exclude", "keep only", "update"]
    if has_prior_recs and any(w in last_msg.lower() for w in refine_words):
        return "refine"

    # If asking about differences or comparisons
    compare_words = ["difference", "different", "compare", "vs", "versus", "which is better"]
    if any(w in last_msg.lower() for w in compare_words):
        return "compare"

    # LLM classification
    try:
        return _classify_intent_llm(messages)
    except Exception:
        traceback.print_exc()
        # Default: if short and first message, clarify; otherwise recommend
        if len(messages) == 1 and len(last_msg.split()) < 6:
            return "clarify"
        return "recommend"


# ---------------------------------------------------------------------------
# Constraint extraction
# ---------------------------------------------------------------------------

def _extract_constraints(messages: list[Message]) -> dict:
    """Use Gemini to extract structured hiring constraints."""
    conversation = _format_conversation(messages)
    try:
        response_text = _call_gemini(EXTRACT_SYSTEM, conversation)
        return _parse_json_response(response_text)
    except Exception:
        traceback.print_exc()
        return {}


# ---------------------------------------------------------------------------
# Catalog context builder
# ---------------------------------------------------------------------------

def _build_catalog_context(items: list[tuple[CatalogItem, float]]) -> str:
    """Format retrieved catalog items for prompt injection."""
    lines = []
    for item, score in items:
        langs = ", ".join(item.languages[:5])
        if len(item.languages) > 5:
            langs += f" (+{len(item.languages) - 5} more)"
        lines.append(
            f"- Name: {item.name}\n"
            f"  URL: {item.url}\n"
            f"  Type: {item.test_type} | Keys: {', '.join(item.keys)}\n"
            f"  Duration: {item.duration or '—'}\n"
            f"  Languages: {langs or '—'}\n"
            f"  Job Levels: {', '.join(item.job_levels)}\n"
            f"  Description: {item.description}\n"
            f"  Remote: {item.remote} | Adaptive: {item.adaptive}\n"
        )
    return "\n".join(lines)


def _build_search_query(messages: list[Message], constraints: dict) -> str:
    """Build a comprehensive search query from conversation + constraints."""
    parts = []

    # Last user message
    last_msg = _get_last_user_message(messages)
    parts.append(last_msg)

    # Key constraint fields
    if constraints.get("role"):
        parts.append(constraints["role"])
    if constraints.get("domain"):
        parts.append(constraints["domain"])
    if constraints.get("skills"):
        parts.extend(constraints["skills"])
    if constraints.get("job_level"):
        parts.append(constraints["job_level"])
    if constraints.get("assessment_types_wanted"):
        parts.extend(constraints["assessment_types_wanted"])
    if constraints.get("special_requirements"):
        parts.append(constraints["special_requirements"])

    # Also include earlier user messages for context
    for msg in messages[:-1]:
        if msg.role == "user":
            parts.append(msg.content)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Response builders for each intent
# ---------------------------------------------------------------------------

def _resolve_names_to_items(names: list[str]) -> list[CatalogItem]:
    """Resolve LLM-returned assessment names to actual catalog items."""
    resolved = []
    seen_urls = set()

    for name in names:
        item = catalog.get_by_name(name)
        if item and item.url not in seen_urls:
            resolved.append(item)
            seen_urls.add(item.url)
            continue

        # Fuzzy match: check if name is a substring
        name_lower = name.lower()
        for cat_item in catalog.items:
            if name_lower in cat_item.name.lower() or cat_item.name.lower() in name_lower:
                if cat_item.url not in seen_urls:
                    resolved.append(cat_item)
                    seen_urls.add(cat_item.url)
                    break

    return resolved


def _items_to_recommendations(items: list[CatalogItem]) -> list[Recommendation]:
    """Convert catalog items to API recommendation objects."""
    recs = []
    for item in items[:10]:  # Max 10
        recs.append(Recommendation(
            name=item.name,
            url=item.url,
            test_type=item.test_type,
            keys=", ".join(item.keys),
            duration=item.duration or "—",
            languages=_summarize_languages_short(item.languages),
        ))
    return recs


def _summarize_languages_short(langs: list[str], max_shown: int = 4) -> str:
    if not langs:
        return "—"
    if len(langs) <= max_shown:
        return ", ".join(langs)
    shown = ", ".join(langs[:max_shown])
    return f"{shown} _(+{len(langs) - max_shown} more)_"


def _extract_previous_recommendations(messages: list[Message]) -> list[str]:
    """Extract assessment names from the conversation history."""
    names = []
    seen_names = set()
    
    # Process all assistant messages to build the current aggregate shortlist
    for msg in messages:
        if msg.role == "assistant":
            content = msg.content
            # We look for exact matches in the text of assistant messages
            # To improve recall, we check against all catalog items
            for item in catalog.items:
                if item.name in content:
                    if item.name not in seen_names:
                        names.append(item.name)
                        seen_names.add(item.name)
    
    # Handle removal requests in the latest user message
    last_user_msg = _get_last_user_message(messages).lower()
    remove_indicators = ["remove", "drop", "exclude", "delete", "stop", "don't want", "take out", "swap out"]
    if any(ind in last_user_msg for ind in remove_indicators):
        final_names = []
        for name in names:
            if name.lower() not in last_user_msg:
                final_names.append(name)
        return final_names

    return names


# ---------------------------------------------------------------------------
# Main handler for each intent
# ---------------------------------------------------------------------------

async def _handle_clarify(messages: list[Message]) -> ChatResponse:
    """Handle vague queries by asking a clarifying question."""
    last_msg = _get_last_user_message(messages).lower()
    
    # Use word boundaries or whole-word matching for greetings to avoid substring hits (like "hi" in "hiring")
    words = set(last_msg.split())
    if words.intersection({"hello", "hi", "hey", "greetings"}):
        reply = "Hello! I'm your SHL assessment assistant. To help you find the right tests, could you tell me about the role you're hiring for and the key skills you want to measure?"
    elif words.intersection({"help", "who"}):
        reply = "I can help you browse the SHL catalog and recommend specific assessments for your hiring needs. What role or job family are you looking to assess?"
    else:
        # Fallback to Gemini for context-aware clarification
        conversation = _format_conversation(messages)
        prompt = CLARIFY_SYSTEM.format(conversation=conversation)

        try:
            response_text = _call_gemini(prompt)
            parsed = _parse_json_response(response_text)
            reply = parsed.get("reply", "Could you tell me more about the role you're hiring for and what type of assessment you need?")
        except Exception:
            traceback.print_exc()
            reply = "I'd love to help! Could you share more details about the role, job level, and what type of assessment you're looking for?"

    return ChatResponse(
        reply=reply,
        recommendations=[],
        end_of_conversation=False,
    )


async def _handle_recommend(messages: list[Message]) -> ChatResponse:
    """Handle recommendation requests with full retrieval + LLM pipeline."""
    # Extract constraints
    constraints = _extract_constraints(messages)

    # Build search query and retrieve
    query = _build_search_query(messages, constraints)
    retrieved = retriever.search(query, top_k=20)

    if not retrieved:
        return ChatResponse(
            reply="I couldn't find matching assessments in the SHL catalog for your requirements. Could you provide more details about the role and skills needed?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Build catalog context for LLM
    catalog_context = _build_catalog_context(retrieved)
    conversation = _format_conversation(messages)

    prompt = RECOMMEND_SYSTEM.format(
        catalog_context=catalog_context,
        conversation=conversation,
        requirements=json.dumps(constraints, indent=2),
    )

    try:
        response_text = _call_gemini(prompt)
        parsed = _parse_json_response(response_text)

        reply = parsed.get("reply", "Based on your requirements, here are my recommendations:")
        recommended_names = parsed.get("recommended_names", [])
        end_conv = parsed.get("end_of_conversation", False)

        # Resolve names to actual catalog items (safety: never trust LLM URLs)
        items = _resolve_names_to_items(recommended_names)

        # If LLM returned no valid names, pick top results from retrieval
        if not items and retrieved:
            items = [item for item, _ in retrieved[:5]]

        recommendations = _items_to_recommendations(items)

    except Exception:
        traceback.print_exc()
        # Fallback: return top retrieval results
        items = [item for item, _ in retrieved[:5]]
        recommendations = _items_to_recommendations(items)
        reply = "Based on your requirements, here are the most relevant SHL assessments:"
        end_conv = False

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_conv,
    )


async def _handle_refine(messages: list[Message]) -> ChatResponse:
    """Handle refinement of previous recommendations."""
    constraints = _extract_constraints(messages)
    prev_names = _extract_previous_recommendations(messages)

    # Retrieve additional items based on latest request
    last_msg = _get_last_user_message(messages)
    query = _build_search_query(messages, constraints)
    retrieved = retriever.search(query, top_k=20)
    catalog_context = _build_catalog_context(retrieved)
    conversation = _format_conversation(messages)

    prompt = REFINE_SYSTEM.format(
        catalog_context=catalog_context,
        previous_recommendations=json.dumps(prev_names),
        conversation=conversation,
        requirements=json.dumps(constraints, indent=2),
    )

    try:
        response_text = _call_gemini(prompt)
        parsed = _parse_json_response(response_text)

        reply = parsed.get("reply", "Updated the recommendation list as requested.")
        recommended_names = parsed.get("recommended_names", prev_names)
        end_conv = parsed.get("end_of_conversation", False)

        items = _resolve_names_to_items(recommended_names)
        recommendations = _items_to_recommendations(items)
    except Exception:
        traceback.print_exc()
        items = _resolve_names_to_items(prev_names)
        recommendations = _items_to_recommendations(items)
        reply = "I've noted your changes. Here's the updated list:"
        end_conv = False

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_conv,
    )


async def _handle_compare(messages: list[Message]) -> ChatResponse:
    """Handle comparison questions about assessments."""
    # Find mentioned assessments
    last_msg = _get_last_user_message(messages)
    prev_names = _extract_previous_recommendations(messages)

    # Search for any assessments mentioned
    query = _build_search_query(messages, {})
    retrieved = retriever.search(query, top_k=15)
    catalog_context = _build_catalog_context(retrieved)
    conversation = _format_conversation(messages)

    prompt = COMPARE_SYSTEM.format(
        catalog_context=catalog_context,
        conversation=conversation,
    )

    try:
        response_text = _call_gemini(prompt)
        parsed = _parse_json_response(response_text)
        reply = parsed.get("reply", "Let me explain the differences between these assessments.")
        end_conv = parsed.get("end_of_conversation", False)
    except Exception:
        traceback.print_exc()
        reply = "These assessments serve different purposes. Could you specify which ones you'd like me to compare?"
        end_conv = False

    return ChatResponse(
        reply=reply,
        recommendations=[],  # Comparisons don't return new recommendations
        end_of_conversation=end_conv,
    )


async def _handle_confirm(messages: list[Message]) -> ChatResponse:
    """Handle conversation conclusion."""
    last_msg = _get_last_user_message(messages).lower()
    user_turns = sum(1 for m in messages if m.role == "user")
    
    # If forced by turn limit
    if user_turns >= 8:
        reply = "We've reached the maximum turn limit for this session. I hope these recommendations help with your hiring process! Please reach out to SHL support if you need further assistance. Goodbye!"
    # Generic polite closings
    elif any(w in last_msg for w in ["thank", "thanks", "great", "perfect", "good", "bye", "exit"]):
        reply = "You're welcome! I'm glad I could help you find the right assessments. If you need anything else, just ask. Goodbye!"
    else:
        reply = "Excellent! I have noted your confirmation for these assessments. You can find more details at the links provided. Is there anything else I can help you with?"

    return ChatResponse(
        reply=reply,
        recommendations=[],
        end_of_conversation=True,
    )


async def _handle_off_topic(messages: list[Message]) -> ChatResponse:
    """Handle off-topic or prompt injection attempts."""
    # Reduced LLM call for common off-topic themes
    last_msg = _get_last_user_message(messages).lower()
    
    if any(w in last_msg for w in ["price", "cost", "how much", "quote", "discount"]):
        reply = "I specialize in helping you select the right SHL assessments based on roles and skills. For pricing information, please visit the SHL website or contact their sales team directly."
    elif any(w in last_msg for w in ["legal", "lawsuit", "sue", "compliance"]):
        reply = "I can provide information about SHL assessment capabilities, but I cannot provide legal or compliance advice. You should consult with your legal department regarding hiring regulations."
    else:
        # Fallback to Gemini for subtle cases
        try:
            conversation = _format_conversation(messages)
            prompt = REFUSE_SYSTEM.format(conversation=conversation)
            response_text = _call_gemini(prompt)
            parsed = _parse_json_response(response_text)
            reply = parsed.get("reply", "I'm sorry, but I can only assist with SHL assessment recommendations.")
        except Exception:
            reply = "I'm specifically designed to help with SHL assessment selection. I can't assist with that request."

    return ChatResponse(
        reply=reply,
        recommendations=[],
        end_of_conversation=False,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request through the full agent pipeline.

    1. Classify intent
    2. Route to appropriate handler
    3. Return schema-safe response
    """
    messages = request.messages

    # Classify intent
    intent = classify_intent(messages)
    
    # Enforce Turn Cap (Hard Eval requirement: max 8 turns)
    user_turns = sum(1 for m in messages if m.role == "user")
    if user_turns >= 8 and intent != "confirm":
        print(f"[agent] Turn limit reached ({user_turns}). Forcing confirmation.")
        intent = "confirm"
    
    print(f"[agent] Intent: {intent} | Turn: {user_turns}/8 | Last msg: {_get_last_user_message(messages)[:80]}")

    # Route to handler
    Handler = Callable[[list[Message]], Awaitable[ChatResponse]]
    handlers: dict[str, Handler] = {
        "clarify": _handle_clarify,
        "recommend": _handle_recommend,
        "refine": _handle_refine,
        "compare": _handle_compare,
        "confirm": _handle_confirm,
        "off_topic": _handle_off_topic,
    }

    handler = handlers.get(intent, _handle_recommend)
    response = await handler(messages)

    return response
