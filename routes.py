"""
sales-prospecting plugin routes
================================
Prefix: /prospecting  (set by plugin_loader from manifest.json)

Endpoints:
    GET    /prospecting/config                        — sectors, regions, tech-focus options
    POST   /prospecting/runs                          — start a discovery+scoring run (background)
    GET    /prospecting/runs                          — list user's runs
    GET    /prospecting/runs/{run_id}                 — run detail + prospects (poll until complete)
    GET    /prospecting/runs/{run_id}/prospects       — scored prospects only
    POST   /prospecting/prospects/{id}/promote        — add to sales_targets pipeline
    POST   /prospecting/targets/{target_id}/briefing  — generate 5-bucket briefing (background)
    GET    /prospecting/briefings/{briefing_id}       — briefing detail (poll until complete)
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests as _requests
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from psycopg2.extras import Json, RealDictCursor
from pydantic import BaseModel

from api.routes.auth import UserInfo, require_jwt
from api.routes.sales import _check_target_access
from core.access import resolve_vertical_id
from core.db.connection import get_connection

_log = logging.getLogger("sales_prospecting")


# ── Direct API helpers (no cvc_config dependency) ─────────────────────────────

def _llm_call(prompt: str, model: str = None, temperature: float = 0.1,
              max_tokens: int = 3000) -> str:
    api_key = os.environ.get("PNP_OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("PNP_OPENROUTER_API_KEY not set")
    resp = _requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model or _LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _brave_search(query: str, count: int = 5, user_id: int | None = None,
                  freshness: str | None = None) -> list[dict]:
    """
    Web search via Brave. `freshness` biases toward recent results — Brave accepts
    pd (past day), pw (week), pm (month), py (year), or a YYYY-MM-DDtoYYYY-MM-DD range.
    Used to keep prospect scoring anchored to current news, not stale model knowledge.
    """
    from api.middleware.ext_api_log import log_ext_call
    primary = os.environ.get("PNP_BRAVE_SEARCH_KEY", "")
    backup  = os.environ.get("PNP_BRAVE_SEARCH_KEY_BACKUP", "")
    for key in [primary, backup]:
        if not key:
            continue
        try:
            params = {"q": query, "count": count, "text_decorations": False}
            if freshness:
                params["freshness"] = freshness
            with log_ext_call("brave", endpoint="web-search", user_id=user_id,
                              cost_usd=0.003, detail=query[:120]) as _bctx:
                resp = _requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"Accept": "application/json", "X-Subscription-Token": key},
                    params=params,
                    timeout=15,
                )
                if resp.status_code == 429:
                    continue
                resp.raise_for_status()
                _bctx.set_status(200)
                results = resp.json().get("web", {}).get("results", [])
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "description": r.get("description", "")}
                for r in results
            ]
        except Exception as e:
            _log.warning(f"Brave search failed for '{query}': {e}")
    return []

def _now_context() -> tuple[str, str]:
    """
    Returns (recent_years, today_label) computed at call time so prospecting stays
    current as the calendar advances — e.g. ("2025 2026", "June 2026").
    """
    now = datetime.now(timezone.utc)
    return f"{now.year - 1} {now.year}", now.strftime("%B %Y")


router = APIRouter()

# ── Config ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(os.environ.get("PLATFORM_ROOT", str(Path(__file__).resolve().parents[3])))
_TEAM_CONFIG_PATH = _REPO_ROOT / "config" / "team.json"
_FALLBACK_SECTORS = [
    "Fintech",
    "Insurtech",
    "Automotive & Mobility",
    "Health & Wellness",
    "Energy & Sustainability",
    "Supply Chain & Logistics",
    "Food & Beverage",
    "Agtech",
    "Real Estate & Construction",
    "Brand & Retail",
    "Enterprise Software",
    "IoT & Smart Devices",
    "Travel & Hospitality",
    "Smart Cities",
    "Media & Advertising",
]

try:
    with open(_TEAM_CONFIG_PATH) as _f:
        _team_cfg = json.load(_f)
    _SECTORS = _team_cfg.get("sectors", []) or _FALLBACK_SECTORS
except Exception:
    _SECTORS = _FALLBACK_SECTORS

_REGIONS = [
    "Global (no region filter)",
    "North America", "Europe", "Asia Pacific", "Middle East & Africa",
    "Latin America",
]

# Granular sub-regions, keyed by region. Lets local teams narrow discovery to
# their locality. Regions not listed here fall back to ["Any"] on the frontend.
_SUB_REGIONS = {
    "North America": [
        "Any", "US Northeast", "US West / California", "US South / Texas",
        "US Midwest", "US Southeast", "Canada", "Mexico",
    ],
}

_LLM_MODEL = "qwen/qwen3-235b-a22b-2507"

# ── Pydantic models ────────────────────────────────────────────────────────────

class RunCreate(BaseModel):
    sectors: list[str]
    region: str = "Global (no region filter)"
    sub_region: str = "Any"          # granular locality within region
    locality: str = ""               # optional free-text locality (e.g. "Greater Boston")
    tech_focus: str = ""  # optional free-text refinement


# ── Helpers ────────────────────────────────────────────────────────────────────

def _serialize(r: dict) -> dict:
    out = dict(r)
    for k, v in out.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out


def _annotate_claims(cur, prospects: list[dict]) -> list[dict]:
    """
    Annotate each prospect with claimed_by / claimed_target_id by matching its
    company_name (case-insensitive) against existing cvc.sales_targets. Lets reps
    see when a corporate is already in someone's pipeline (warn-only). One query.
    """
    if not prospects:
        return prospects
    names = [p["company_name"].lower() for p in prospects if p.get("company_name")]
    claims: dict = {}
    if names:
        try:
            cur.execute("""
                SELECT DISTINCT ON (lower(company_name))
                       lower(company_name) AS lname, assigned_to, id AS target_id
                FROM cvc.sales_targets
                WHERE lower(company_name) = ANY(%s)
                ORDER BY lower(company_name), id
            """, (names,))
            for r in cur.fetchall():
                claims[r["lname"]] = {"assigned_to": r["assigned_to"], "target_id": r["target_id"]}
        except Exception:
            claims = {}
    for p in prospects:
        c = claims.get((p.get("company_name") or "").lower())
        p["claimed_by"] = c["assigned_to"] if c else None
        p["claimed_target_id"] = c["target_id"] if c else None
    return prospects


def _parse_llm_json(text: str) -> dict | list:
    """Parse JSON from LLM output. Strips markdown fences, falls back to regex."""
    t = text.strip()
    if "```" in t:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", t)
        t = m.group(1).strip() if m else t
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", t)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return {}


def _brave_news_for_company(company_name: str) -> str:
    """Fetch recent news for a company from Brave. Returns formatted text block."""
    try:
        recent_years, _ = _now_context()
        results = _brave_search(
            f"{company_name} technology strategy innovation news {recent_years}",
            count=5, freshness="py",
        )
        return "\n".join(
            f"- {r['title']} — {r['description']}" for r in results
        )
    except Exception:
        return ""


def _platform_intel_for_company(company_name: str, sector: str) -> str:
    """
    Pull intel from platform tables (category_news, content_items, briefing_insights).
    All queries are optional — returns empty string if tables missing or empty.
    """
    snippets = []

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # category_news
            try:
                cur.execute("""
                    SELECT title, published_at FROM cvc.category_news
                    WHERE company_name ILIKE %s AND hidden IS NOT TRUE
                    ORDER BY published_at DESC LIMIT 5
                """, (f"%{company_name}%",))
                for r in cur.fetchall():
                    snippets.append(f"[News] {r['title']}")
            except Exception:
                pass

            # content_items by key_entities
            try:
                cur.execute("""
                    SELECT title, summary, published_at FROM cvc.content_items
                    WHERE key_entities @> %s::jsonb
                    ORDER BY published_at DESC LIMIT 5
                """, (json.dumps(company_name),))
                for r in cur.fetchall():
                    snippets.append(f"[Intel] {r['title']}: {(r.get('summary') or '')[:200]}")
            except Exception:
                pass

            # briefing_insights by sector
            try:
                cur.execute("""
                    SELECT insight, confidence FROM cvc.briefing_insights
                    WHERE sector ILIKE %s
                    ORDER BY week_start DESC LIMIT 5
                """, (f"%{sector}%",))
                for r in cur.fetchall():
                    snippets.append(f"[Sector signal] {r['insight']}")
            except Exception:
                pass

    return "\n".join(snippets)


# ── Background: discovery + scoring ───────────────────────────────────────────

def _run_discovery_and_scoring(run_id: int, criteria: dict) -> None:
    """
    Plain def — runs in Starlette threadpool so the API stays responsive.

    1. Fast-fail if BRAVE_SEARCH_KEY unset.
    2. Qwen enumerates ~18 established corporates matching criteria.
    3. Per candidate: scores innovation / budget / tech-fit using Brave news + platform intel.
    4. Stores scored prospects into cvc.sales_prospects.
    """
    if not os.environ.get("PNP_BRAVE_SEARCH_KEY") and not os.environ.get("PNP_BRAVE_SEARCH_KEY_BACKUP"):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE cvc.sales_prospect_runs
                    SET status='error', error_message=%s, updated_at=NOW()
                    WHERE id=%s
                """, ("PNP_BRAVE_SEARCH_KEY not configured — set this env var to enable prospecting.", run_id))
                conn.commit()
        return

    # Mark running
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE cvc.sales_prospect_runs
                SET status='running', started_at=NOW(), updated_at=NOW()
                WHERE id=%s
            """, (run_id,))
            conn.commit()

    try:
        sectors = criteria.get("sectors", [])
        region  = criteria.get("region", "global")
        sub_region = criteria.get("sub_region", "Any")
        locality   = criteria.get("locality", "")
        tech_focus = criteria.get("tech_focus", "")

        recent_years, today_label = _now_context()
        sectors_str = ", ".join(sectors) if sectors else "any industry"
        region_str  = region if region != "Global (no region filter)" else "worldwide"
        # Compose a granular geography string: region + sub-region + free-text locality
        geo_str = region_str
        if sub_region and sub_region != "Any":
            geo_str += f", specifically the {sub_region} area"
        if locality:
            geo_str += f", concentrated in/around {locality}"
        tech_str    = f" with a focus on {tech_focus}" if tech_focus else ""

        # ── Step 1: Qwen enumerates candidates ────────────────────────────────
        enum_prompt = f"""You are a B2B sales intelligence analyst. Today is {today_label}. List 18 real corporations that match ALL of these criteria:
- Sector: {sectors_str}
- Geography: {geo_str}{tech_str}
- Only include companies whose headquarters or a major operating presence is within the specified geography.
- They are established enterprises (not early-stage startups) that could benefit from startup-corporate innovation partnerships
- They have shown genuine interest in innovation (CVC arm, R&D labs, innovation programs, accelerator participation, or patent activity)
- Prioritize companies with demonstrable innovation activity in the last 12-18 months. Do not rely on outdated information.

Return ONLY a JSON array of objects. No explanation. Each object:
{{
  "company_name": "Official company name",
  "website": "https://...",
  "country": "US",
  "hq_city": "City, State/Province (best guess if unsure)",
  "hq_region": "The sub-region bucket it belongs to (e.g. US Northeast), or empty",
  "sector": "matched sector",
  "why_eligible": "One sentence — why this company fits the criteria"
}}

JSON array:"""

        raw = _llm_call(enum_prompt, temperature=0.1, max_tokens=3000)
        candidates = _parse_llm_json(raw)
        if not isinstance(candidates, list):
            candidates = []

        stored = 0
        rejected = 0

        for candidate in candidates[:20]:  # cap at 20
            try:
                name    = (candidate.get("company_name") or "").strip()
                website = (candidate.get("website") or "").strip()
                sector  = (candidate.get("sector") or sectors_str)
                if not name:
                    continue

                # Location (LLM best-effort). hq_location is a single display string.
                country   = (candidate.get("country") or "").strip() or None
                hq_city   = (candidate.get("hq_city") or "").strip()
                hq_region = (candidate.get("hq_region") or "").strip()
                hq_location = hq_city or hq_region or country or None

                # ── Step 2: Classify as PnP lead (High / Medium / Low) ────────
                revenue_verified = True
                revenue_evidence = ""
                prospect_status = "scored"

                score_innovation = score_budget = score_tech_fit = score_overall = None
                tech_direction = {}
                scoring_json = {}

                # Gather intel — platform tables + always-fresh Brave news so scoring
                # reflects current reality, not the model's stale training knowledge.
                platform_intel = _platform_intel_for_company(name, sector)
                fresh_news = _brave_news_for_company(name)
                if fresh_news:
                    platform_intel = f"{platform_intel}\n{fresh_news}".strip() if platform_intel else fresh_news

                # Search for the key innovation/tech leader at this company
                contact_results = _brave_search(
                    f"{name} chief innovation officer CTO digital transformation AI head {recent_years} interview",
                    count=3, freshness="py",
                )
                contact_snippets = "\n".join(
                    f"- {r.get('title', '')} — {r.get('description', '')[:200]}"
                    for r in contact_results
                )

                classify_prompt = f"""You are a partnership analyst at Plug and Play Tech Center, the world's largest corporate innovation platform.

Today is {today_label}. Weight evidence from the last 12-18 months most heavily; treat older signals as historical context only. Base your assessment on the intelligence gathered below, not on potentially outdated assumptions.

Evaluate {name} as a potential corporate partner for innovation programs and startup collaboration{tech_str}.

Intelligence gathered:
{platform_intel or "(no specific intel — use general knowledge)"}

Contact search:
{contact_snippets or "(no contact results)"}

Classify using this rubric:

HIGH: Strong, verifiable signals across 3+ categories in the last 12-18 months:
- Innovation Budget: dedicated innovation labs, CVC arm, announced AI/digital budgets, Chief Innovation Officer
- Active Initiatives: running or participating in accelerators, incubators, innovation programs
- Startup Collaboration: history of startup partnerships, CVC investments, open innovation calls
- Pilot Activity: evidence of POCs, sandboxes, or rapid experimentation across business units
- PnP Fit: aligns with PnP verticals, realistic candidate to run pilots with our startup network

MEDIUM: Some signals (1-2 categories), limited or unclear execution history

LOW: Weak or no evidence of innovation activity or startup engagement

Also identify the most prominent person at this company who speaks publicly about innovation, digital transformation, or AI strategy (CIO, CTO, Chief Innovation Officer, Head of AI, etc.). Only name real people with evidence — return null if genuinely unknown.

Reply with JSON only — no markdown, no explanation:
{{
  "classification": "High",
  "confidence": "High",
  "key_signals": ["Concrete bullet — specific program, investment, or initiative", "..."],
  "recent_examples": ["Specific initiative or pilot with year if known", "..."],
  "why_good_for_plug_and_play": "2-3 sentences on strategic fit for PnP programs",
  "missing_signals": ["What concrete evidence would upgrade this classification"],
  "key_contact": {{
    "name": "Full Name or null",
    "title": "Exact title",
    "why": "Why they are the key person to engage"
  }},
  "tech_direction": {{
    "summary": "2-sentence summary of their technology direction",
    "signals": ["tech area 1", "tech area 2", "tech area 3"]
  }}
}}

JSON:"""

                class_raw = _llm_call(classify_prompt, temperature=0.1, max_tokens=1000)
                class_data = _parse_llm_json(class_raw)
                if isinstance(class_data, dict):
                    classification = class_data.get("classification", "Medium")
                    _score_map = {"High": 90, "Medium": 60, "Low": 25}
                    score_overall    = _score_map.get(classification, 60)
                    score_innovation = score_budget = score_tech_fit = score_overall
                    tech_direction   = class_data.get("tech_direction", {})
                    scoring_json     = {
                        "classification":         class_data.get("classification", "Medium"),
                        "confidence":             class_data.get("confidence", "Medium"),
                        "key_signals":            class_data.get("key_signals", []),
                        "recent_examples":        class_data.get("recent_examples", []),
                        "why_good_for_plug_and_play": class_data.get("why_good_for_plug_and_play", ""),
                        "missing_signals":        class_data.get("missing_signals", []),
                        "key_contact":            class_data.get("key_contact", {}),
                    }

                # ── Step 3: Store prospect ─────────────────────────────────────
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO cvc.sales_prospects
                                (run_id, company_name, website, revenue_verified, revenue_evidence,
                                 score_innovation, score_budget, score_tech_fit, score_overall,
                                 tech_direction, scoring_json, status, country, hq_location)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (run_id, company_name) DO NOTHING
                        """, (
                            run_id, name, website or None, revenue_verified, revenue_evidence,
                            score_innovation, score_budget, score_tech_fit, score_overall,
                            Json(tech_direction), Json(scoring_json), prospect_status,
                            country, hq_location,
                        ))
                        conn.commit()

                stored += 1

            except Exception as e:
                _log.warning(f"prospecting: candidate error ({name}): {e}")
                continue

        # Mark complete
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE cvc.sales_prospect_runs
                    SET status='complete', completed_at=NOW(), updated_at=NOW(),
                        candidate_count=%s
                    WHERE id=%s
                """, (stored, run_id))
                conn.commit()

        _log.info(f"prospecting run {run_id} complete — {stored} scored, {rejected} rejected")

    except Exception as exc:
        _log.error(f"prospecting run {run_id} failed: {exc}")
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE cvc.sales_prospect_runs
                        SET status='error', error_message=%s, updated_at=NOW()
                        WHERE id=%s
                    """, (str(exc)[:500], run_id))
                    conn.commit()
        except Exception:
            pass


# ── Background: briefing generation ───────────────────────────────────────────

# (key, query template, fresh) — `fresh` buckets are time-sensitive and get a
# recency filter + current-year hint so the briefing reflects today, not stale data.
_BRIEFING_BUCKETS = [
    ("overview",    "{name} company overview business model products",                 False),
    ("financials",  "{name} revenue growth financials annual report {years}",          True),
    ("tech",        "{name} technology stack R&D innovation AI digital transformation", False),
    ("news",        "{name} news press release announcement {years}",                  True),
    ("leadership",  "{name} CEO leadership team strategy vision",                      False),
]


def _run_briefing(target_id: int, briefing_id: int) -> None:
    """Generate 5-bucket briefing for a sales target. Plain def (threadpool)."""
    # Mark running
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                UPDATE cvc.sales_briefings
                SET status='running', updated_at=NOW()
                WHERE id=%s
                RETURNING *
            """, (briefing_id,))
            row = cur.fetchone()
            conn.commit()
            if not row:
                return

    try:
        # Get target info
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM cvc.sales_targets WHERE id=%s", (target_id,))
                target = cur.fetchone()
        if not target:
            raise ValueError(f"sales_target {target_id} not found")

        name   = target["company_name"]
        sector = target.get("sector") or ""

        # ── 5 Brave buckets ───────────────────────────────────────────────────
        recent_years, today_label = _now_context()
        research_json = {}
        for bucket_key, query_tmpl, fresh in _BRIEFING_BUCKETS:
            query = query_tmpl.format(name=name, years=recent_years)
            try:
                results = _brave_search(query, count=5, freshness="py" if fresh else None)
                text = "\n".join(f"- {r['title']} — {r['description']}" for r in results)
                research_json[bucket_key] = text or "[no results]"
            except Exception as e:
                research_json[bucket_key] = f"[search failed: {e}]"

        combined = "\n\n---\n\n".join(
            f"## {k.upper()}\n{v}" for k, v in research_json.items()
        )

        # ── Qwen synthesis ────────────────────────────────────────────────────
        synthesis_prompt = f"""You are a senior B2B sales intelligence analyst. Today is {today_label}. Using the research below — which is weighted toward recent sources — write a comprehensive, up-to-date sales briefing for {name}. Favor the most recent developments and flag anything that looks dated.

RESEARCH:
{combined[:8000]}

Return JSON with these exact keys:
{{
  "company_overview": "2-3 sentence company description: what they do, who they serve, market position",
  "products_and_services": "Key products/services and value proposition (2-3 sentences)",
  "financials_snapshot": "Revenue scale, growth trajectory, financial health signals (2-3 sentences)",
  "technology_direction": "Where they are heading technologically — AI, automation, digital transformation initiatives (2-3 sentences)",
  "why_now": "Why this is a good time to approach them — triggers, initiatives, or signals indicating openness to new partnerships",
  "key_contacts_hint": "Typical title/function of decision-makers at companies like this (do not invent names)",
  "recommended_angle": "Specific recommended angle for outreach — what problem to lead with based on their current direction"
}}

JSON:"""

        synthesis_raw = _llm_call(synthesis_prompt, temperature=0.2, max_tokens=2000)
        briefing_json = _parse_llm_json(synthesis_raw)
        if not isinstance(briefing_json, dict):
            briefing_json = {"raw": synthesis_raw}

        # ── Tech interests extraction ──────────────────────────────────────────
        tech_prompt = f"""Based on this company briefing for {name}, extract 3-5 specific technology interest areas.

Briefing:
{json.dumps(briefing_json, indent=2)[:2000]}

Return JSON array:
[
  {{"area": "AI / Machine Learning", "confidence": "HIGH", "evidence": "brief evidence", "source_category": "tech"}},
  ...
]

JSON array:"""

        tech_raw = _llm_call(tech_prompt, temperature=0.1, max_tokens=800)
        tech_interests = _parse_llm_json(tech_raw)
        if not isinstance(tech_interests, list):
            tech_interests = []

        # ── Outreach drafts ───────────────────────────────────────────────────
        outreach_prompt = f"""Write 2 outreach email drafts for {name} — one for a C-suite executive (CTO/CDO), one for a VP/Director of Innovation or Operations.

Context from briefing:
- Company: {name}
- Recommended angle: {briefing_json.get("recommended_angle", "")}
- Why now: {briefing_json.get("why_now", "")}

Each email: subject line + 3-4 sentence body. Professional, specific, no generic phrases.

Return JSON array:
[
  {{"persona": "C-suite (CTO/CDO)", "subject": "...", "body": "..."}},
  {{"persona": "VP Innovation/Operations", "subject": "...", "body": "..."}}
]

JSON array:"""

        outreach_raw = _llm_call(outreach_prompt, temperature=0.3, max_tokens=800)
        outreach_drafts = _parse_llm_json(outreach_raw)
        if not isinstance(outreach_drafts, list):
            outreach_drafts = []

        # ── Persist ───────────────────────────────────────────────────────────
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE cvc.sales_briefings
                    SET status='complete', research_json=%s, briefing_json=%s,
                        tech_interests=%s, outreach_drafts=%s,
                        generated_at=NOW(), updated_at=NOW()
                    WHERE id=%s
                """, (
                    Json(research_json), Json(briefing_json),
                    Json(tech_interests), Json(outreach_drafts),
                    briefing_id,
                ))
                conn.commit()

        _log.info(f"briefing {briefing_id} complete for target {target_id} ({name})")

    except Exception as exc:
        _log.error(f"briefing {briefing_id} failed: {exc}")
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE cvc.sales_briefings
                        SET status='error', error_message=%s, updated_at=NOW()
                        WHERE id=%s
                    """, (str(exc)[:500], briefing_id))
                    conn.commit()
        except Exception:
            pass


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/config")
def get_config(user: UserInfo = Depends(require_jwt)):
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT name FROM cvc.verticals ORDER BY name")
                rows = cur.fetchall()
        sectors = [r["name"] for r in rows] if rows else _SECTORS
    except Exception:
        sectors = _SECTORS
    return {"sectors": sectors, "regions": _REGIONS, "sub_regions": _SUB_REGIONS}


@router.post("/runs", status_code=201)
def create_run(
    body: RunCreate,
    background_tasks: BackgroundTasks,
    user: UserInfo = Depends(require_jwt),
):
    if not body.sectors:
        raise HTTPException(400, "At least one sector is required")

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 409 if user already has a run actively running
            cur.execute("""
                SELECT id FROM cvc.sales_prospect_runs
                WHERE created_by=%s AND status='running'
                LIMIT 1
            """, (user.username,))
            if cur.fetchone():
                raise HTTPException(409, "You already have a prospecting run in progress. Wait for it to complete before starting a new one.")

            # Auto-cancel any stuck pending runs (worker never picked them up)
            cur.execute("""
                UPDATE cvc.sales_prospect_runs
                SET status='error', error_message='Cancelled — new search started', updated_at=NOW()
                WHERE created_by=%s AND status='pending'
            """, (user.username,))

            criteria = {
                "sectors": body.sectors,
                "region": body.region,
                "sub_region": body.sub_region,
                "locality": body.locality,
                "tech_focus": body.tech_focus,
            }
            cur.execute("""
                INSERT INTO cvc.sales_prospect_runs (status, criteria, created_by)
                VALUES ('pending', %s, %s)
                RETURNING id, status, criteria, created_by, created_at
            """, (Json(criteria), user.username))
            row = dict(cur.fetchone())
            conn.commit()

    run_id = row["id"]
    background_tasks.add_task(_run_discovery_and_scoring, run_id, criteria)

    return _serialize(row)


@router.get("/runs")
def list_runs(user: UserInfo = Depends(require_jwt)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, status, criteria, error_message, candidate_count,
                       created_by, started_at, completed_at, created_at, updated_at
                FROM cvc.sales_prospect_runs
                WHERE created_by=%s
                ORDER BY created_at DESC
                LIMIT 20
            """, (user.username,))
            return [_serialize(dict(r)) for r in cur.fetchall()]


@router.get("/runs/{run_id}")
def get_run(run_id: int, user: UserInfo = Depends(require_jwt)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, status, criteria, error_message, candidate_count,
                       created_by, started_at, completed_at, created_at, updated_at
                FROM cvc.sales_prospect_runs
                WHERE id=%s AND created_by=%s
            """, (run_id, user.username))
            run = cur.fetchone()
            if not run:
                raise HTTPException(404, "Run not found")
            run = _serialize(dict(run))

            cur.execute("""
                SELECT id, company_name, website, revenue_verified, revenue_evidence,
                       score_innovation, score_budget, score_tech_fit, score_overall,
                       tech_direction, scoring_json, status, promoted_target_id,
                       country, hq_location, created_at
                FROM cvc.sales_prospects
                WHERE run_id=%s
                ORDER BY score_overall DESC NULLS LAST, company_name
            """, (run_id,))
            run["prospects"] = _annotate_claims(cur, [_serialize(dict(r)) for r in cur.fetchall()])

    return run


@router.get("/runs/{run_id}/prospects")
def list_prospects(run_id: int, user: UserInfo = Depends(require_jwt)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # verify ownership
            cur.execute("SELECT id FROM cvc.sales_prospect_runs WHERE id=%s AND created_by=%s", (run_id, user.username))
            if not cur.fetchone():
                raise HTTPException(404, "Run not found")

            cur.execute("""
                SELECT id, company_name, website, revenue_verified, revenue_evidence,
                       score_innovation, score_budget, score_tech_fit, score_overall,
                       tech_direction, scoring_json, status, promoted_target_id,
                       country, hq_location, created_at
                FROM cvc.sales_prospects
                WHERE run_id=%s
                ORDER BY score_overall DESC NULLS LAST, company_name
            """, (run_id,))
            return _annotate_claims(cur, [_serialize(dict(r)) for r in cur.fetchall()])


@router.post("/prospects/{prospect_id}/promote", status_code=201)
def promote_prospect(prospect_id: int, user: UserInfo = Depends(require_jwt), request: Request = None):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT p.*, r.criteria FROM cvc.sales_prospects p
                JOIN cvc.sales_prospect_runs r ON r.id = p.run_id
                WHERE p.id=%s AND r.created_by=%s
            """, (prospect_id, user.username))
            prospect = cur.fetchone()
            if not prospect:
                raise HTTPException(404, "Prospect not found")
            if prospect["status"] == "rejected":
                raise HTTPException(400, "Cannot promote a rejected prospect")
            if prospect["status"] == "promoted":
                # Already promoted — return the existing target
                if prospect["promoted_target_id"]:
                    cur.execute("SELECT id FROM cvc.sales_targets WHERE id=%s", (prospect["promoted_target_id"],))
                    if cur.fetchone():
                        return {"target_id": prospect["promoted_target_id"], "already_promoted": True}

            name    = prospect["company_name"]
            website = prospect.get("website")
            criteria = prospect.get("criteria") or {}
            sector  = (criteria.get("sectors") or [None])[0]

            # Dedupe: check if company already in sales_targets
            cur.execute(
                "SELECT id FROM cvc.sales_targets WHERE lower(company_name)=lower(%s) LIMIT 1",
                (name,),
            )
            existing = cur.fetchone()
            if existing:
                target_id = existing["id"]
            else:
                cur.execute("""
                    INSERT INTO cvc.sales_targets
                        (company_name, website, sector, assigned_to, rationale, created_by, vertical_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    name, website, sector,
                    user.username,
                    f"Auto-added from prospecting run. Tech direction: {(prospect.get('tech_direction') or {}).get('summary', '')}",
                    user.username,
                    resolve_vertical_id(user, request),
                ))
                target_id = cur.fetchone()["id"]

                # Auto-enroll in news watch (defensive — plugin may not be installed)
                try:
                    cur.execute("""
                        INSERT INTO cvc.news_watch_companies (company_name, category)
                        VALUES (%s, 'sales')
                        ON CONFLICT (company_name, category) DO NOTHING
                    """, (name,))
                except Exception:
                    conn.rollback()

            # Mark prospect promoted
            cur.execute("""
                UPDATE cvc.sales_prospects
                SET status='promoted', promoted_target_id=%s
                WHERE id=%s
            """, (target_id, prospect_id))
            conn.commit()

    return {"target_id": target_id, "already_promoted": bool(existing)}


@router.post("/targets/{target_id}/briefing", status_code=201)
def generate_briefing(
    target_id: int,
    background_tasks: BackgroundTasks,
    user: UserInfo = Depends(require_jwt),
):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            _check_target_access(cur, target_id, user)

            # Reuse existing row or create new one
            cur.execute("""
                SELECT id, status FROM cvc.sales_briefings
                WHERE sales_target_id=%s
                ORDER BY created_at DESC LIMIT 1
            """, (target_id,))
            existing = cur.fetchone()

            if existing and existing["status"] in ("running", "pending"):
                return _serialize(dict(existing))

            if existing:
                # Reset and reuse
                cur.execute("""
                    UPDATE cvc.sales_briefings
                    SET status='pending', research_json='{}', briefing_json='{}',
                        tech_interests='[]', outreach_drafts='[]',
                        error_message=NULL, generated_at=NULL, updated_at=NOW()
                    WHERE id=%s
                    RETURNING id, status, sales_target_id, created_at
                """, (existing["id"],))
                row = dict(cur.fetchone())
            else:
                cur.execute("""
                    INSERT INTO cvc.sales_briefings (sales_target_id, status)
                    VALUES (%s, 'pending')
                    RETURNING id, status, sales_target_id, created_at
                """, (target_id,))
                row = dict(cur.fetchone())
            conn.commit()

    background_tasks.add_task(_run_briefing, target_id, row["id"])
    return _serialize(row)


@router.get("/briefings/{briefing_id}")
def get_briefing(briefing_id: int, user: UserInfo = Depends(require_jwt)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT b.id, b.sales_target_id, b.status, b.error_message,
                       b.research_json, b.briefing_json, b.tech_interests,
                       b.outreach_drafts, b.generated_at, b.created_at, b.updated_at,
                       t.company_name, t.website, t.sector
                FROM cvc.sales_briefings b
                JOIN cvc.sales_targets t ON t.id = b.sales_target_id
                WHERE b.id=%s
            """, (briefing_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Briefing not found")
    return _serialize(dict(row))
