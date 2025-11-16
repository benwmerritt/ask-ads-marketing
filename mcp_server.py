#!/usr/bin/env python3
"""
MCP Server for Ask Ads & Marketing

Exposes paid advertising and marketing strategy content as Model Context Protocol tools
for Claude Desktop using the fastmcp library.

Tools:
  - ask_ads_marketing(question, top_k?, max_tokens?, user_context?) -> Markdown answer with Sources
  - about() -> System information
"""

import os
import sys
from typing import List, Dict, Optional, Any
import yaml as __yaml

# Ensure imports from this repo work regardless of cwd
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
try:
    # Ensure relative paths in config.yaml resolve correctly
    os.chdir(REPO_ROOT)
except Exception:
    pass

# Allow per-profile config via environment variable
CONFIG_PATH = os.getenv('WIKI_VAULT_CONFIG', os.path.join(REPO_ROOT, 'config.yaml'))
try:
    with open(CONFIG_PATH, 'r') as __f:
        __CFG = __yaml.safe_load(__f)
except Exception:
    __CFG = {}

# Ensure imported scripts don't print to stdout and break MCP protocol
os.environ['WIKI_VAULT_SILENT'] = '1'

from fastmcp import FastMCP  # type: ignore

# Reuse existing functionality
from lib.query import KnowledgeQuery
from lib.full_notes import FullNotesReader


# MCP server name
PROVIDER_NAME = os.getenv('WIKI_VAULT_MCP_NAME') or "ask-ads-marketing"
mcp = FastMCP(PROVIDER_NAME)


class _Lazy:
    query: Optional[KnowledgeQuery] = None
    notes: Optional[FullNotesReader] = None


def _get_query() -> KnowledgeQuery:
    if _Lazy.query is None:
        _Lazy.query = KnowledgeQuery(config_path=CONFIG_PATH)
    return _Lazy.query


def _get_notes() -> FullNotesReader:
    if _Lazy.notes is None:
        _Lazy.notes = FullNotesReader(config_path=CONFIG_PATH)
    return _Lazy.notes


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _answer_with_openai(question: str, context: str, max_tokens: int = 2000, user_context: Optional[str] = None) -> Optional[str]:
    try:
        import openai as _openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or len(api_key) < 10:
            return None
        client = _openai.OpenAI(api_key=api_key)
        system_prompt = """
ROLE: You are a paid advertising and digital marketing strategist with deep expertise in Meta ads, email marketing, social media strategy, and ad spend optimization. Your knowledge comes from expert marketing practitioners who specialize in data-driven, ROI-focused campaigns.

EXPERTISE AREAS:
- Paid advertising (Meta/Facebook ads, Instagram ads, TikTok ads, YouTube ads)
- Social media marketing and organic growth strategies
- Email marketing campaigns and automation sequences
- Ad spend optimization and budget allocation
- Conversion funnel design and optimization
- Landing page strategy and CRO
- Audience targeting and segmentation
- Creative testing and ad copy frameworks
- Analytics, attribution, and performance tracking

TASK: Answer the user's question using only the provided marketing content excerpts. Match your answer's depth and complexity to the question:
- Tactical questions → Direct, actionable strategies with specific frameworks
- Strategic questions → High-level approaches with ROI considerations
- Technical questions → Platform-specific guidance and optimization techniques
- Vague questions → Ask clarifying questions about their goals, budget, or current setup

APPROACH:
- Lead with data-driven insights and ROI-focused recommendations
- Provide specific frameworks, formulas, and testing methodologies
- Include platform-specific tactics (Meta Ads Manager, email deliverability, etc.)
- Use only information explicitly stated in the context
- Focus on measurable outcomes and performance metrics
- Emphasize testing, iteration, and continuous optimization

FORMAT FLEXIBILITY:
Prioritize these elements:
1. Clear strategy or tactical framework
2. Specific action steps with expected outcomes
3. Metrics to track and success criteria
4. Common pitfalls and optimization tips
5. Testing approaches and iteration cycles
6. Budget considerations and scaling strategies

CITATIONS: Reference content naturally in your response. End with a "Sources" section listing content titles (no URLs).

CRITICAL CONSTRAINT: Only use information explicitly stated in the provided context. Do not make up metrics, invent strategies, or add information not present in the sources.

EXAMPLES:

Meta ads question: "How do I structure my Meta ad campaigns for maximum ROI?"
Response: Comprehensive breakdown covering campaign structure (awareness/consideration/conversion), audience segmentation strategies, creative testing frameworks, budget allocation across ad sets, optimization for specific conversion events, and scaling tactics from proven performers. Include specific targeting approaches and bidding strategies from the content.

Email marketing question: "What's the best way to improve my email open rates?"
Response: Direct explanation of subject line frameworks, sender reputation factors, list segmentation strategies, optimal send times, deliverability best practices, and A/B testing approaches. Include specific techniques for warming up domains and maintaining engagement metrics.

Social media strategy: "How do I grow my Instagram following for my e-commerce brand?"
Response: Actionable plan covering content strategy (Reels vs Posts), hashtag research and optimization, engagement tactics, collaboration approaches, conversion funnel from bio link, and organic growth frameworks. Include posting frequency recommendations and content themes that drive both engagement and sales.

Ad spend optimization: "My cost per acquisition is too high. How do I fix it?"
Response: Systematic diagnostic covering audience quality, creative performance analysis, landing page conversion rates, offer strength evaluation, and attribution model review. Include specific metrics to track, testing priorities, and optimization sequence based on highest-leverage improvements.
"""
        user_tailor = (f"\n\nUser context to tailor recommendations: {user_context}" if user_context else "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context excerpts:\n{context}{user_tailor}"},
            {"role": "user", "content": f"Question: {question}\n\nProvide helpful guidance based only on the context above."},
        ]
        resp = client.chat.completions.create(
            model=os.getenv('WIKI_VAULT_OPENAI_MODEL', 'gpt-4o-mini'),
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def _keywords(text: str) -> list:
    import re as _re
    toks = _re.findall(r"[A-Za-z0-9']+", text.lower())
    return [t for t in toks if len(t) >= 3]


def _rank_hits_by_keyword(question: str, hits: list, k: int) -> list:
    words = set(_keywords(question))
    if not words:
        return hits[:k]
    def score_hit(h: dict) -> float:
        txt = (h.get('content') or '').lower()
        score = sum(txt.count(w) for w in words)
        sim = h.get('score') or 0.0
        return score + sim
    ranked = sorted(hits, key=score_hit, reverse=True)
    seen = set()
    out = []
    for h in ranked:
        key = hash((h.get('content') or '')[:160])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= k:
            break
    return out


def _group_by_source(hits: list, max_sources: int = 5, per_source: int = 1) -> list:
    """Group hits by file_path/title and take top segments per source for diversity."""
    buckets = {}
    order = []
    for h in hits:
        meta = h.get('metadata', {}) or {}
        key = meta.get('file_path') or meta.get('title') or meta.get('source_title')
        if not key:
            key = h.get('id')
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(h)
    selected = []
    for key in order:
        segs = buckets[key][:per_source]
        selected.extend(segs)
        if len(selected) >= max_sources * per_source:
            break
    return selected


@mcp.tool()
def ask_ads_marketing(question: str, top_k: Any = 18, max_tokens: Any = 2600, user_context: Optional[str] = None, response_style: str = 'detailed') -> Dict[str, Any]:
    """Ask questions about paid advertising, Meta ads, email marketing, social media strategy, and ad spend optimization. Get expert marketing guidance with actionable frameworks and data-driven strategies. Returns {answer, sources}."""
    q = _get_query()
    k = _to_int(top_k) or 12
    mt = _to_int(max_tokens) or 2000
    # Adjust verbosity based on response_style
    style = (response_style or 'detailed').lower()
    if style == 'concise':
        mt = min(mt, 1200)
    elif style == 'comprehensive':
        mt = max(mt, 3200)
    initial = max(50, k * 5)
    hits = q.search(question, top_k=initial, collection='content')
    # Prefer transcripts
    filtered = [h for h in hits if (h.get('metadata', {}) or {}).get('doc_type') == 'transcript'] or hits
    filtered = _rank_hits_by_keyword(question, filtered, initial)
    # Diversify: take top segments grouped by source
    filtered = _group_by_source(filtered, max_sources=min(6, k//3 + 2), per_source=1)
    # Build context
    seen = set()
    parts = []
    for h in filtered:
        txt = h.get('content') or ''
        if not txt:
            continue
        key = hash(txt[:120])
        if key in seen:
            continue
        seen.add(key)
        meta = h.get('metadata', {}) or {}
        title = meta.get('title') or meta.get('source_title') or 'Untitled'
        section = meta.get('section') or meta.get('concept_name') or ''
        head = f"Source: {title}"
        if section:
            head += f" ({section})"
        parts.append(head + "\n" + txt)
    context = "\n\n---\n".join(parts)
    ans = _answer_with_openai(question, context, max_tokens=mt, user_context=user_context)
    if not ans:
        # Structured fallback
        parts = context.split("\n\n---\n")[:8]
        bullets = "\n\n".join(f"- {p[:300]}" for p in parts)
        ans = (
            f"# Answer (no OpenAI API available)\n\nBased on retrieved marketing content excerpts.\n\n"
            f"## Key Marketing Strategies\n\n{bullets}\n\n"
            f"## Suggested Next Steps\n\n- Review which framework or strategy applies to your current campaigns\n- Implement one optimization this week (audience testing, creative refresh, or funnel improvement)\n- Track your metrics and iterate based on performance data\n"
        )
    # Build sources list (titles only, no URLs)
    sources = []
    seen_src = set()
    used_scores = []
    for h in filtered:
        meta = h.get('metadata', {}) or {}
        title = meta.get('title') or meta.get('source_title') or 'Untitled'
        url = meta.get('url') or meta.get('source_url')
        fp = meta.get('file_path')
        key = (title, url or fp)
        if key in seen_src:
            continue
        seen_src.add(key)
        sources.append({'title': title, 'url': url})
        if isinstance(h.get('score'), (int, float)):
            used_scores.append(h.get('score'))
        if len(sources) >= 5:
            break
    confidence = round(sum(used_scores)/len(used_scores), 3) if used_scores else None
    return {'answer': ans, 'sources': sources, 'confidence': confidence}


@mcp.tool()
def about() -> Dict[str, Any]:
    """Returns a short description of this MCP provider and how to use it."""
    try:
        import yaml as _yaml
        with open(CONFIG_PATH, 'r') as _f:
            _cfg = _yaml.safe_load(_f)
        kb = _cfg.get('knowledge_base', {})
        return {
            'name': kb.get('name') or 'Ask Ads & Marketing KB',
            'purpose': 'Search and get expert paid advertising and marketing strategy advice',
            'topic': kb.get('topic') or 'Paid advertising, social media marketing, ad spend optimization, and email strategy',
            'recommended_tools': ['ask_ads_marketing', 'about'],
            'notes': 'Use ask_ads_marketing for questions about paid ads, marketing strategies, and campaign optimization. Example questions: "How do I structure Meta ad campaigns for e-commerce?", "What\'s the best email sequence for SaaS onboarding?", "How do I improve my Instagram engagement rate?", "What metrics should I track for paid social ROI?", "How do I scale profitable ad campaigns?". Get comprehensive answers with sources from expert marketing content.'
        }
    except Exception:
        return {
            'name': 'Ask Ads & Marketing KB',
            'purpose': 'Paid advertising and marketing strategy Q&A',
            'recommended_tools': ['ask_ads_marketing', 'about'],
            'notes': 'Ask questions about paid ads, Meta advertising, email marketing, social media strategy, and campaign optimization.'
        }


if __name__ == "__main__":
    # Run the MCP server (stdio)
    mcp.run()
