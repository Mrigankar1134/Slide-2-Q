from __future__ import annotations

import os
import io
import pickle
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from backend.utils.rules import generate_varied_questions, clean_text
import logging
from backend.utils.pptx_parser import extract_slides_text
from dotenv import load_dotenv
import google.generativeai as genai
import json as _json
import re

# Attempt fast JSON
try:
    import orjson

    def orjson_dumps(v, *, default):
        return orjson.dumps(v, default=default)

    json_response_class = JSONResponse
except Exception:
    json_response_class = JSONResponse

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
app = FastAPI(title="PPTX Question Generator", version="0.1.0")
logger = logging.getLogger("pptx-question-generator")

# CORS (configurable via env). Set ALLOWED_ORIGINS as a comma-separated list.
# Example: "https://your-site.netlify.app,http://localhost:5173"
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
allowed_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", "").strip() or None
if allowed_origins_env:
    allow_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Sensible defaults for local dev
    allow_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://slide2q.netlify.app"
    ]

# Normalize by removing trailing slashes to match browser Origin exactly
allow_origins = [o.rstrip('/') if o != "*" else o for o in allow_origins]

allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
# Starlette forbids "*" with credentials. If user sets "*", drop credentials.
if "*" in allow_origins and allow_credentials:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log effective CORS config for debugging deployments
try:
    logger.info(
        "CORS config: allow_origins=%s, allow_origin_regex=%s, allow_credentials=%s",
        allow_origins,
        allowed_origin_regex,
        allow_credentials,
    )
except Exception:
    pass

VECTORIZER = None
LDA_MODEL = None
GEMINI_MODEL = None


@app.on_event("startup")
def load_models() -> None:
    global VECTORIZER, LDA_MODEL, GEMINI_MODEL
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    lda_path = os.path.join(models_dir, "lda_model.pkl")
    if os.path.exists(vec_path):
        try:
            with open(vec_path, "rb") as f:
                VECTORIZER = pickle.load(f)
        except Exception as exc:
            VECTORIZER = None
            try:
                logger.warning("Failed to load TF-IDF vectorizer from %s: %s", vec_path, exc)
            except Exception:
                pass
    if os.path.exists(lda_path):
        try:
            with open(lda_path, "rb") as f:
                LDA_MODEL = pickle.load(f)
        except Exception as exc:
            LDA_MODEL = None
            try:
                logger.warning("Failed to load LDA model from %s: %s", lda_path, exc)
            except Exception:
                pass

    # Configure Gemini if API key present
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Use flash for speed; can be overridden later
            GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as exc:
            GEMINI_MODEL = None
            try:
                logger.warning("Failed to configure Gemini: %s", exc)
            except Exception:
                pass


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "vectorizer_loaded": VECTORIZER is not None,
        "lda_loaded": LDA_MODEL is not None,
        "ai_ready": GEMINI_MODEL is not None,
    }


def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    if not text or VECTORIZER is None:
        return []
    try:
        X = VECTORIZER.transform([text])
        if hasattr(VECTORIZER, "get_feature_names_out"):
            feature_names = VECTORIZER.get_feature_names_out()
        else:
            feature_names = VECTORIZER.get_feature_names()
        # For sparse vector, get top non-zero indices
        row = X.tocoo()
        if row.nnz == 0:
            return []
        indices = row.col
        data = row.data
        order = np.argsort(-data)
        top_indices = [indices[i] for i in order[:top_k]]
        keywords = [feature_names[i] for i in top_indices]
        # unique preserving order
        seen = set()
        uniq = []
        for k in keywords:
            if k not in seen:
                uniq.append(k)
                seen.add(k)
        return uniq
    except Exception:
        return []


def infer_topics(text: str, top_n: int = 3) -> List[str]:
    if not text or VECTORIZER is None or LDA_MODEL is None:
        return []
    try:
        X = VECTORIZER.transform([text])
        if hasattr(LDA_MODEL, "transform"):
            dist = LDA_MODEL.transform(X)
            if dist is None or len(dist) == 0:
                return []
            topics = np.argsort(-dist[0])[:top_n]
            return [f"Topic {int(t)}" for t in topics]
        return []
    except Exception:
        return []


def _heuristic_refine_questions(
    slide_text: str,
    questions: List[Dict[str, str]],
    keywords: List[str],
    limit: int = 10,
) -> List[Dict[str, str]]:
    """Refine questions deterministically without AI when toggle is on but AI unavailable.

    Drops low-signal numeric questions, enforces reasonable length, deduplicates,
    and balances types up to a small per-type cap.
    """
    if not questions:
        return []

    def has_units(q: str) -> bool:
        return any(u in q for u in ["%", "Cr", "₹"]) or bool(re.search(r"\d{2,}|\d{1,3},\d{3}", q))

    def low_signal_numeric(q: str) -> bool:
        # Bare single-digit like '1' or quoted '1' without units
        if re.search(r"'\d{1,2}'", q) and not has_units(q):
            return True
        if re.search(r"\b\d\b", q) and not has_units(q):
            return True
        return False

    def generic_filler(q: str) -> bool:
        ql = q.lower()
        if "how might" in ql and "evolve" in ql:
            return True
        if ql.startswith("do you agree") or ql.startswith("is it important"):
            return True
        return False

    # Filter
    filtered: List[Dict[str, str]] = []
    seen = set()
    for item in questions:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        if len(q) < 20 or len(q) > 160:
            continue
        if low_signal_numeric(q) or generic_filler(q):
            continue
        if q in seen:
            continue
        seen.add(q)
        filtered.append({"type": item.get("type", ""), "question": q})

    if not filtered:
        return questions[: min(limit, len(questions))]

    # Score by keyword hit and lexical richness
    def score(item: Dict[str, str]) -> float:
        q = item["question"].lower()
        kw_hits = sum(1 for k in keywords[:5] if k.lower() in q)
        tokens = re.findall(r"[a-zA-Z]{3,}", q)
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        return kw_hits * 2.0 + unique_ratio

    filtered.sort(key=score, reverse=True)

    # Balance by type: max 2 per type initially
    per_type_cap = 2
    type_counts: Dict[str, int] = {}
    balanced: List[Dict[str, str]] = []
    for item in filtered:
        t = (item.get("type") or "other").lower()
        if type_counts.get(t, 0) >= per_type_cap:
            continue
        type_counts[t] = type_counts.get(t, 0) + 1
        balanced.append(item)
        if len(balanced) >= limit:
            break

    if len(balanced) < min(limit, len(filtered)):
        # Fill remaining slots ignoring type cap
        for item in filtered:
            if item in balanced:
                continue
            balanced.append(item)
            if len(balanced) >= limit:
                break

    return balanced


def _enforce_variety(questions: List[Dict[str, str]], min_count: int = 6, max_count: int = 10) -> List[Dict[str, str]]:
    """Reduce near-duplicates and cap per-type to improve variety.

    - Deduplicate by normalized text
    - Cap common types (e.g., 'factual') to avoid repetitive who/which questions
    - Preserve order from input
    """
    if not questions:
        return []

    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    per_type_cap_default = 1
    type_overrides = {
        "conceptual": 2,
        "process": 2,
        "application": 2,
    }

    seen_norm: set[str] = set()
    counts: Dict[str, int] = {}
    primary: List[Dict[str, str]] = []
    remainder: List[Dict[str, str]] = []
    for item in questions:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        n = norm(q)
        if n in seen_norm:
            continue
        seen_norm.add(n)
        t = (item.get("type") or "other").lower()
        cap = type_overrides.get(t, per_type_cap_default)
        if counts.get(t, 0) < cap and len(primary) < max_count:
            counts[t] = counts.get(t, 0) + 1
            primary.append(item)
        else:
            remainder.append(item)

    out = primary
    for item in remainder:
        if len(out) >= max_count:
            break
        out.append(item)

    if len(out) < min_count:
        out.extend(remainder[: (min_count - len(out))])

    return out[:max_count]


@app.post("/generate-questions")
async def generate_questions(
    file: UploadFile = File(...),
) -> Any:
    if not file.filename.lower().endswith(".pptx"):
        return JSONResponse(status_code=400, content={"error": "Only .pptx files are supported"})

    # Save to temp in-memory then to disk for python-pptx
    contents = await file.read()
    tmp_path = os.path.join("/tmp", file.filename)
    with open(tmp_path, "wb") as f:
        f.write(contents)

    slides = extract_slides_text(tmp_path)
    os.remove(tmp_path)

    # Always refine via AI pipeline: prefer Gemini, otherwise heuristic
    use_ai = True
    results: List[Dict[str, Any]] = []
    ai_possible = bool(GEMINI_MODEL is not None)
    for idx, raw_text in enumerate(slides):
        text = clean_text(raw_text)
        kws = extract_keywords(text)
        topics = infer_topics(text)

        # Split text into sentences in a simple way; spaCy could be used but keep simple
        sentences = [s.strip() for s in re_split_sentences(text) if s.strip()]

        slide_qs: List[Dict[str, str]] = []
        for sent in sentences or [text]:
            pairs = generate_varied_questions(sent, slide_kws=kws, slide_topics=topics)
            for typ, q in pairs:
                slide_qs.append({"type": typ, "question": q})

        # Optional: filter and deduplicate via Gemini for essential, meaningful questions
        if ai_possible and slide_qs:
            try:
                prompt = (
                    "ROLE: You are an expert educational content editor. Given slide text and candidate questions, "
                    "curate a final list of high-quality questions.\n\n"
                    "GOAL: Return ONLY the most essential and meaningful 6–10 questions that best assess understanding of the slide.\n\n"
                    "HARD CONSTRAINTS:\n"
                    "- Questions MUST be unambiguously grounded in the slide text.\n"
                    "- DISCARD low-signal or generic items: single bare numbers (e.g., '1') without units/context; vague 'trend/impact' with no subject; tautologies; yes/no questions; filler like 'How might X evolve in 5 years?' if not slide-relevant.\n"
                    "- Remove duplicates and near-duplicates (semantic similarity).\n"
                    "- Each question should be clear, grammatical, specific, <= 140 chars, end with '?'.\n"
                    "- If using numbers, include units or context from the slide; avoid quoting bare digits.\n"
                    "- Do NOT invent facts beyond the slide.\n"
                    "- If the slide text is very short, return fewer (3–5) rather than padding with generic questions.\n\n"
                    "PREFER (diversify across these where possible): conceptual (what/why/how), application, process, cause_effect, compare, evaluation, example.\n"
                    "STYLE: Prefer concrete nouns from the slide over pronouns — e.g., refer to named teams, institutions, metrics, units explicitly. Avoid phrases like 'these members', 'this team' — use the actual names from the slide when present.\n"
                    "EDITING: You MAY rewrite selected questions to improve specificity, clarity, and grounding (replace pronouns with names; add units; remove vague words). Preserve original meaning.\n"
                    "Keep type labels consistent with the closest of: [factual, conceptual, process, application, cause_effect, compare, evaluation, example, keyword, topic, impact, challenge]. Preserve original type when sensible; otherwise map to the closest.\n\n"
                    "OUTPUT: Return ONLY a JSON array (no prose) of objects {type, question}. Strictly valid JSON.\n\n"
                    f"SLIDE TEXT:\n{text}\n\n"
                    f"KEYWORDS (hints, optional): {', '.join(kws) if kws else '[]'}\n\n"
                    "CANDIDATE QUESTIONS (type: question):\n" +
                    "\n".join(f"- {q['type']}: {q['question']}" for q in slide_qs)
                )
                r = GEMINI_MODEL.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.3,
                        "max_output_tokens": 1024,
                    },
                )
                # Try to get JSON string from response
                raw = None
                if hasattr(r, "text") and r.text:
                    raw = r.text
                elif getattr(r, "candidates", None):
                    # Join all text parts from the top candidate
                    parts = []
                    for p in r.candidates[0].content.parts:
                        if getattr(p, "text", None):
                            parts.append(p.text)
                    raw = "".join(parts) if parts else None
                if raw:
                    try:
                        parsed = _json.loads(raw)
                    except Exception:
                        # Fallback: extract JSON array substring
                        start = raw.find("[")
                        end = raw.rfind("]")
                        parsed = _json.loads(raw[start:end+1]) if start != -1 and end != -1 and end > start else None
                    if isinstance(parsed, list):
                        cleaned: List[Dict[str, str]] = []
                        for item in parsed:
                            qt = str(item.get("type", "ai")).strip()
                            qq = str(item.get("question", "")).strip()
                            if qq:
                                cleaned.append({"type": qt, "question": qq})
                        if cleaned:
                            slide_qs = cleaned
            except Exception:
                # On any failure, keep original questions
                pass

        # If toggle is on but AI not available or produced nothing new, apply heuristic refinement
        if (not ai_possible or not slide_qs):
            slide_qs = _heuristic_refine_questions(text, slide_qs, kws, limit=10)

        # Always enforce variety when user asked for AI refinement
        slide_qs = _enforce_variety(slide_qs, min_count=6, max_count=10)

        results.append({
            "slide_index": idx + 1,
            "text": text,
            "keywords": kws,
            "topics": topics,
            "questions": slide_qs,
        })

    return {
        "slides": results,
        "ai_used": True,
        "ai_model": ("gemini-1.5-flash" if ai_possible else "heuristic"),
    }


# Simple sentence splitter avoiding dependency on spaCy pipelines for senter
_PUNCT_SPLIT = ".!?\n\r"

def re_split_sentences(text: str) -> List[str]:
    import re
    # Split on punctuation followed by space/newline; keep basic robustness
    parts = re.split(r"(?<=[\.!?])\s+|\n+|\r+", text)
    return parts
