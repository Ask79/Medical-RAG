# rag_med/summarize.py
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# Parsing: tolerant to logs
# ---------------------------
def _parse_payload_loose(data: str):
    import json, ast
    s = (data or "").strip()
    if not s:
        raise ValueError("Empty input")

    # Extract all top-level {...} blocks so router logs don't break us
    stack = 0
    start = None
    blocks: List[str] = []
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    blocks.append(s[start : i + 1])
                    start = None

    def _try_load(txt: str):
        try:
            return json.loads(txt)
        except Exception:
            return ast.literal_eval(txt)

    # Prefer the last block that looks like a router payload
    candidates: List[Dict[str, Any]] = []
    for raw in (blocks or [s]):
        try:
            obj = _try_load(raw)
        except Exception:
            continue
        if isinstance(obj, dict) and (
            isinstance(obj.get("evidence"), list) or isinstance(obj.get("drug"), dict)
        ):
            candidates.append(obj)

    if candidates:
        return candidates[-1]
    # Fallback to the last parsed block or the entire string
    raw = blocks[-1] if blocks else s
    return _try_load(raw)


INTENT_TITLES = {
    "indication": "Indications & Usage",
    "adverse_effect": "Adverse Effects",
    "contraindication": "Contraindications",
    "interaction": "Drug Interactions",
    "dosage": "Dosage & Administration",
    "warning": "Warnings & Precautions",
    "mechanism": "Mechanism of Action",
    "pregnancy": "Use in Specific Populations",
    "overview": "Product Overview",
}

def _safe_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()

def _intent_title(intent: Optional[str]) -> str:
    return INTENT_TITLES.get(_safe_str(intent).lower(), "Summary")

def _normalize_section(s: Optional[str]) -> str:
    return _safe_str(s).strip().lower().replace(" ", "_")

# ---------------------------
# Canonical link helpers (Phase 1)
# ---------------------------
def _rxnorm_links_from_payload(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    drug = payload.get("drug") or {}
    rxcui = _safe_str(drug.get("rxcui"))
    ent_links = (payload.get("sources") or {}).get("entity_links") or {}
    # Prefer RXCUI if present; otherwise fall back to entity link passed through
    seen = set()

    def _add(label: str, url: str):
        u = _safe_str(url)
        if not u or u in seen:
            return
        seen.add(u)
        rows.append({"label": label, "url": u})

    if rxcui:
        _add("RxNorm (UI)", f"https://rxnav.nlm.nih.gov/#!search={rxcui}")
        _add("RxNorm (REST)", f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}.json")
    # Preserve any existing rxnav entity link if different (back-compat)
    rxnav_passthru = _safe_str(ent_links.get("rxnav"))
    if rxnav_passthru:
        _add("RxNorm (drug record)", rxnav_passthru)

    return rows

def _snomed_link_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    # Try entity-level first
    ent_links = (payload.get("sources") or {}).get("entity_links") or {}
    sn = _safe_str(ent_links.get("snomed"))
    if sn:
        return {"label": "SNOMED CT (concept)", "url": sn}
    # Otherwise, scan evidence for the first SCTID we can find
    for ev in payload.get("evidence") or []:
        sct = _safe_str(ev.get("condition_sctid") or ev.get("sctid"))
        if sct:
            return {
                "label": "SNOMED CT (concept)",
                "url": f"https://browser.ihtsdotools.org/?perspective=full&conceptId1={sct}",
            }
    return None

def _build_citation_index(payload: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str], int], List[Dict[str, str]]]:
    """
    Returns:
      - map[(spl_set_id, section_norm)] -> citation number (1-based)  [for DailyMed items only]
      - list[index-1] -> {"label": "...", "url": "..."}

    Robust to evidence_links being strings (URLs) or dicts with fields.
    DailyMed evidence is numbered; RxNorm/SNOMED links are appended after (not numbered).
    """
    src = payload.get("sources", {}) or {}
    ev_links = src.get("evidence_links") or []
    ent_links = src.get("entity_links") or {}

    # Build a quick reverse index from evidence URL -> (spl_set_id, section)
    url_to_meta: Dict[str, Tuple[str, str]] = {}
    for ev in payload.get("evidence") or []:
        try:
            sid = _safe_str(ev.get("spl_set_id"))
            sec = _normalize_section(ev.get("section"))
            dm = _safe_str((ev.get("links") or {}).get("dailymed"))
            if sid and dm:
                url_to_meta[dm] = (sid, sec)
        except Exception:
            pass

    # Normalize evidence links: allow strings or dicts
    norm_ev_links: List[Dict[str, str]] = []
    for link in ev_links:
        if isinstance(link, dict):
            dm_url = _safe_str(link.get("dailymed")) or _safe_str(link.get("url"))
            sid = _safe_str(link.get("spl_set_id"))
            sec = _normalize_section(link.get("section"))
            if (not sid or not sec) and dm_url in url_to_meta:
                sid2, sec2 = url_to_meta[dm_url]
                sid = sid or sid2
                sec = sec or sec2
            if dm_url:
                norm_ev_links.append({"dailymed": dm_url, "spl_set_id": sid, "section": sec})
        elif isinstance(link, str):
            dm_url = _safe_str(link)
            sid, sec = url_to_meta.get(dm_url, ("", ""))
            norm_ev_links.append({"dailymed": dm_url, "spl_set_id": sid, "section": sec})

    index_map: Dict[Tuple[str, str], int] = {}
    rows: List[Dict[str, str]] = []

    # 1) Numbered citations for DailyMed evidence (these back the bullets)
    for link in norm_ev_links:
        sid = _safe_str(link.get("spl_set_id"))
        sec = _normalize_section(link.get("section"))
        dm_url = _safe_str(link.get("dailymed"))
        if not sid or not dm_url:
            continue
        key = (sid, sec or "")
        if key in index_map:
            continue
        rows.append({"label": f"DailyMed (SPL {sid})", "url": dm_url})
        index_map[key] = len(rows)

    # 2) Fallback: entity-level DailyMed if nothing else resolved
    entity_dm = _safe_str(ent_links.get("dailymed"))
    if entity_dm and not rows:
        rows.append({"label": "DailyMed (product overview)", "url": entity_dm})
        index_map[("", "")] = 1  # generic fallback

    # 3) Always append vocabulary/ID links (not used for bullet numbering)
    # RxNorm canonical (UI + REST), then passthrough if any
    for row in _rxnorm_links_from_payload(payload):
        rows.append(row)

    # SNOMED (from entity link or first SCTID in evidence)
    sn = _snomed_link_from_payload(payload)
    if sn:
        rows.append(sn)

    return index_map, rows

def _lookup_cite_num(cite_index: Dict[Tuple[str, str], int], sid: Optional[str], section: Optional[str]) -> Optional[int]:
    sid = _safe_str(sid)
    sec = _normalize_section(section)
    # exact (sid, section)
    if (sid, sec) in cite_index:
        return cite_index[(sid, sec)]
    # any section for this SPL
    for k in cite_index.keys():
        if k[0] == sid:
            return cite_index[k]
    # final fallback: single entity-level
    if ("", "") in cite_index:
        return cite_index[("", "")]
    return None

# ---------------------------
# Bullet shaping
# ---------------------------
def _clean_snip(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    s = re.sub(r"(Manufactured by|Distributed by|Revised:).+$", "", s, flags=re.I)
    return s.strip()

def _bullet_text(ev: Dict[str, Any], intent: str) -> str:
    """
    Produce a short, human-friendly bullet.
    - Prefer SNOMED PT; fall back to snippet (title-cased if short)
    - Append short quoted context only if snippet adds info
    """
    pt = _safe_str(ev.get("condition_pt"))
    snip = _clean_snip(_safe_str(ev.get("snippet")))
    if len(snip) > 160:
        snip = snip[:157] + "..."
    head = pt or (snip[:1].upper() + snip[1:] if snip else "")

    add_context = ""
    if snip and pt and snip.lower() != pt.lower():
        if len(snip) > 20 or re.search(r"[.;:!?]", snip):
            add_context = f' — "{snip}"'

    # Safety prefix for boxed items
    if ev.get("label") == "serious_safety" or "boxed" in _normalize_section(ev.get("section")):
        head = f"⚠️ {head}"

    if intent == "indication":
        # Avoid trailing "(disorder)" / "(finding)" in display
        head = re.sub(r"\s*\((disorder|finding)\)\s*$", "", head, flags=re.I)

    return f"{head}{add_context}".strip()

# ---------------------------
# Answer-first line (Phase 5)
# ---------------------------
_DOSE_RX = re.compile(r"\b\d{1,4}\s?(?:mg|mcg|g|units)(?:/\w+)?(?:\s*(?:once|twice|three|four|q\d+h|q\d+h|bid|tid|qid|daily|weekly|every\s+\d+\s*(?:hours|days|weeks)))?", re.I)

def _extract_heads(bullets: List[str]) -> List[str]:
    out = []
    for b in bullets:
        t = b.lstrip("•- ").strip()
        t = re.sub(r"^⚠️\s*", "", t)
        head = t.split("—", 1)[0].strip().strip('"')
        if head:
            out.append(head)
    # de-dup preserve order
    seen, uniq = set(), []
    for h in out:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq

def _extract_doses(bullets: List[str]) -> List[str]:
    seen, doses = set(), []
    for b in bullets:
        for m in _DOSE_RX.findall(b):
            mm = m.strip()
            if mm and mm.lower() not in seen:
                seen.add(mm.lower())
                doses.append(mm)
            if len(doses) >= 3:
                break
        if len(doses) >= 3:
            break
    return doses

def _answer_first(intent: str, drug_name: str, bullet_texts: List[str]) -> str:
    intent = (intent or "").lower()
    name = (drug_name or "").strip()
    heads = _extract_heads(bullet_texts)
    if intent == "dosage":
        doses = _extract_doses(bullet_texts)
        if doses:
            return f"Usual dosing examples for {name}: " + "; ".join(doses[:2]) + "."
        return f"Dosing for {name}: see label details below."
    if intent == "indication" and heads:
        return f"Indicated uses for {name}: " + "; ".join(heads[:3]) + "."
    if intent == "adverse_effect" and heads:
        return f"Reported adverse effects include: " + "; ".join(heads[:3]) + "."
    if intent == "interaction":
        return "Clinically significant interactions are noted below; review before co-administration."
    if intent == "contraindication" and heads:
        return "Contraindications include: " + "; ".join(heads[:3]) + "."
    if intent == "mechanism" and heads:
        return heads[0] + "."
    if intent == "pregnancy":
        return "See pregnancy/lactation guidance and risk discussion below."
    # fallback
    return heads[0] + "." if heads else ""

# ---------------------------
# Core summary struct
# ---------------------------
def _summarize_struct(payload: Dict[str, Any], top_k: int = 6) -> Tuple[Dict[str, Any], List[str], List[Optional[int]]]:
    """Build the deterministic summary structure + raw bullet texts + citation indices."""
    intent = _safe_str(
        payload.get("graph", {}).get("link_types", [payload.get("intent")])[0]
        if payload.get("graph", {}).get("link_types") else payload.get("intent")
    ).lower()

    drug = payload.get("drug") or {}
    drug_name = _safe_str(drug.get("matched_name") or drug.get("query_text"))
    title = f"{drug_name or 'This product'} — {_intent_title(intent)}"

    evidence = list(payload.get("evidence") or [])
    evidence.sort(key=lambda ev: float(ev.get("final_score", 0.0)), reverse=True)

    # Build cite index once
    cite_index, cite_rows = _build_citation_index(payload)

    bullets_data: List[Dict[str, Any]] = []
    raw_texts: List[str] = []
    raw_cites: List[Optional[int]] = []

    # Keep 3–6 concise bullets max
    max_bullets = max(3, min(6, int(top_k or 6)))

    taken = 0
    for ev in evidence:
        if taken >= max_bullets:
            break
        try:
            if ev.get("final_score") is not None and float(ev.get("final_score", 0.0)) < 0.1:
                continue
        except Exception:
            pass
        txt = _bullet_text(ev, intent)
        cite_num = _lookup_cite_num(cite_index, ev.get("spl_set_id"), ev.get("section"))
        bullets_data.append({
            "text": txt,
            "score": float(ev.get("final_score", 0.0)),
            "snomed": _safe_str(ev.get("condition_sctid")),
            "links": ev.get("links") or {},
            "citation": cite_num,
        })
        raw_texts.append(txt)
        raw_cites.append(cite_num)
        taken += 1

    disclaimer = (
        "Summary of labeled information; not a substitute for clinical judgment. "
        "Confirm organism susceptibility, local guidelines, and patient-specific factors."
    )

    # Answer-first one-liner (Phase 5)
    answer_line = _answer_first(intent, drug_name, raw_texts)

    struct = {
        "title": title,
        "intent": intent,
        "drug": {
            "name": drug_name,
            "rxcui": _safe_str(drug.get("rxcui")),
            "spl_set_id": _safe_str(drug.get("spl_set_id")),
        },
        "answer": answer_line,
        "bullets": bullets_data,    # text may be refined by model step
        "citations": cite_rows,     # [{label,url}] indexed 1..N (DailyMed numbered first; RxNorm/SNOMED appended)
        "disclaimer": disclaimer,
        "meta": {
            "total_evidence": len(evidence),
            "top_k": taken,
        },
    }
    return struct, raw_texts, raw_cites

# ---------------------------
# Grouping near-duplicates
# ---------------------------
QUALIFIERS = {
    "acute", "bacterial", "chronic", "viral", "recurrent",
    "severe", "mild", "moderate", "complicated", "uncomplicated"
}

def _head_and_rest(text: str) -> tuple[str, str]:
    """Split bullet text into head and optional '— rest'."""
    parts = [p.strip() for p in text.split("—", 1)]
    head = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    return head, rest

def _signature_from_head(head: str) -> str:
    """
    Signature for grouping: last two tokens of the head (letters only), lowercased.
    Groups 'Acute Otitis Media', 'Bacterial Otitis Media', 'Otitis Media' together.
    """
    toks = re.findall(r"[A-Za-z]+", head.lower())
    if not toks:
        return ""
    if len(toks) >= 2:
        return " ".join(toks[-2:])
    return toks[0]

def _qualifiers_from_head(head: str, signature: str) -> List[str]:
    """Qualifiers = tokens in head that aren’t in the signature (simple heuristic)."""
    sig_set = set(signature.split())
    toks = re.findall(r"[A-Za-z]+", head.lower())
    quals = [t for t in toks if t not in sig_set and t in QUALIFIERS]
    # de-duplicate, preserve order
    seen, out = set(), []
    for q in quals:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def _group_bullets(bullets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group bullets that share the same head signature. Merge qualifiers and citations.
    Keeps the shortest head as the group label. Preserves first appearance order.
    """
    if not bullets:
        return bullets

    groups: Dict[str, List[Dict[str, Any]]] = {}
    order_index: Dict[int, int] = {}

    for idx, b in enumerate(bullets):
        head, _ = _head_and_rest(b.get("text", ""))
        sig = _signature_from_head(head)
        groups.setdefault(sig, []).append(b)
        order_index[id(b)] = idx

    merged: List[Dict[str, Any]] = []
    for sig, items in groups.items():
        if sig == "" or len(items) == 1:
            merged.extend(items)
            continue

        # Choose the shortest head as the base label
        heads = [_head_and_rest(b.get("text", ""))[0] for b in items]
        base_head = min((h for h in heads if h), key=len, default=heads[0] if heads else "")

        # Collect qualifiers and citations
        qset: List[str] = []
        cites: List[int] = []

        for b in items:
            h, _ = _head_and_rest(b.get("text", ""))
            qset.extend(_qualifiers_from_head(h, sig))
            c = b.get("citation")
            if isinstance(c, list):
                cites.extend([int(x) for x in c if x is not None])
            elif c is not None:
                cites.append(int(c))

        # Unique, stable order
        seen_q, quniq = set(), []
        for q in qset:
            if q not in seen_q:
                seen_q.add(q)
                quniq.append(q)

        seen_c, cuniq = set(), []
        for c in cites:
            if c not in seen_c:
                seen_c.add(c)
                cuniq.append(c)

        text = base_head
        if quniq:
            text = f"{text} — " + "; ".join(quniq)

        # Merge: keep non-text fields from the highest-scoring item
        best = max(items, key=lambda x: float(x.get("score", 0.0)))
        merged.append({
            **best,
            "text": text,
            "citation": cuniq if len(cuniq) > 1 else (cuniq[0] if cuniq else None),
        })

    # Preserve first-appearance ordering by the earliest member of each group
    def first_idx(b: Dict[str, Any]) -> int:
        h = _head_and_rest(b.get("text", ""))[0]
        sig = _signature_from_head(h)
        members = groups.get(sig, [])
        if not members:
            return order_index.get(id(b), 10**9)
        return min(order_index.get(id(m), 10**9) for m in members)

    merged.sort(key=first_idx)
    return merged

# ---------------------------
# Optional model refinement
# ---------------------------
_PIPELINE = None

def _load_pipeline(model_name: str, device: Optional[str] = None):
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    try:
        from transformers import pipeline
    except Exception as e:
        sys.stderr.write(f"[summarize] transformers unavailable ({e}); falling back to deterministic bullets.\n")
        return None
    kwargs = {"task": "text2text-generation", "model": model_name}
    # Device mapping: "cuda" or "cpu". If None, HF auto-selects.
    if device:
        kwargs["device"] = 0 if device.lower().startswith("cuda") else -1
    try:
        _PIPELINE = pipeline(**kwargs)
        return _PIPELINE
    except Exception as e:
        sys.stderr.write(f"[summarize] Failed to load model '{model_name}': {e}\n")
        return None

def _refine_bullets_with_model(
    bullets: List[str],
    title: str,
    model_name: str,
    device: Optional[str],
    max_new_tokens: int = 96
) -> List[str]:
    """
    Ask a small local model to lightly polish the bullets WITHOUT adding new facts.
    Returns same-length list of bullets (best effort). If anything fails, returns input bullets.
    """
    if not bullets:
        return bullets

    n = len(bullets)
    pipe = _load_pipeline(model_name, device)
    if pipe is None:
        return bullets

    # Keep it short, faithful, same order, same count.
    prompt = (
        "You are a clinical summarizer. Rewrite the bullets to be concise and clear, "
        "without adding or removing any facts. Keep the SAME number of bullets and the SAME order. "
        "Keep any leading warning icons. Do NOT invent new information. "
        f"Title: {title}\n\n"
        "Bullets:\n" +
        "\n".join(f"- {b}" for b in bullets) +
        "\n\nReturn ONLY the bullets, one per line starting with '- '."
    )

    try:
        out = pipe(
            prompt,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_beams=1
        )
    except Exception as e:
        sys.stderr.write(f"[summarize] Generation failed: {e}\n")
        return bullets

    text = out[0]["generated_text"] if isinstance(out, list) and out else ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
    if len(lines) != n:
        return bullets
    return [ln[2:].strip() for ln in lines]

# ---------------------------
# Formatters
# ---------------------------
def _fmt_cites(c):
    if c is None:
        return ""
    if isinstance(c, list):
        return f" [{','.join(str(x) for x in c)}]"
    return f" [{c}]"

def _md_escape(s: str) -> str:
    return re.sub(r'([\\`*_{}\[\]()#+.!|-])', r'\\\1', s)

def format_cli(summary: Dict[str, Any]) -> str:
    lines = [summary.get("title", "Summary"), ""]
    # Answer-first line
    ans = _safe_str(summary.get("answer"))
    if ans:
        lines.append(ans)
        lines.append("")
    for b in summary.get("bullets", []):
        lines.append(f"• {b['text']}{_fmt_cites(b.get('citation'))}")
    lines.append("")
    if summary.get("citations"):
        lines.append("Sources:")
        for i, row in enumerate(summary["citations"], start=1):
            lines.append(f"[{i}] {row['label']}: {row['url']}")
        lines.append("")
    if summary.get("disclaimer"):
        lines.append(f"Note: {summary['disclaimer']}")
    return "\n".join(lines)

def format_markdown(summary: Dict[str, Any]) -> str:
    out = [f"# {_md_escape(summary.get('title', 'Summary'))}\n"]
    ans = _safe_str(summary.get("answer"))
    if ans:
        out.append(_md_escape(ans))
        out.append("")
    for b in summary.get("bullets", []):
        out.append(f"- {_md_escape(b['text'])}{_fmt_cites(b.get('citation'))}")
    out.append("")
    if summary.get("citations"):
        out.append("## Sources")
        for i, row in enumerate(summary["citations"], start=1):
            label = _md_escape(row['label'])
            url = row['url']
            out.append(f"{i}. [{label}]({url})")
        out.append("")
    if summary.get("disclaimer"):
        out.append(f"> {summary['disclaimer']}")
    return "\n".join(out)

# ---------------------------
# CLI
# ---------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phase 5: Summarize & Cite (answer-first + canonical links + optional grouping/model polish)")
    p.add_argument("--input", "-i", default="-", help="Payload JSON (file or '-' for stdin). Raw router logs OK.")
    p.add_argument("--format", "-f", default="cli", choices=["cli", "md", "json"], help="Output format")
    p.add_argument("--top-k", type=int, default=6, help="Max bullets (3–6 enforced)")
    p.add_argument("--group", action="store_true", help="Group near-duplicate bullets (e.g., 'Otitis media').")
    p.add_argument("--model", default=None, help="HF model name for optional bullet polishing (e.g., google/flan-t5-small)")
    p.add_argument("--device", default=None, help="Device hint: 'cpu' or 'cuda'. Default: auto")
    p.add_argument("--max-new", type=int, default=96, help="Max new tokens for generation")
    p.add_argument("--no-model", action="store_true", help="Force disable model even if specified")
    args = p.parse_args(argv)

    data = sys.stdin.read() if args.input == "-" else open(args.input, "r", encoding="utf-8").read()
    try:
        payload = _parse_payload_loose(data)
    except Exception as e:
        sys.stderr.write(f"[summarize] Failed to parse input: {e}\n")
        snippet = (data or "")[:400].replace("\n", " ")
        sys.stderr.write(f"[summarize] First 400 chars: {snippet}\n")
        return 2

    summary, raw_bullets, raw_cites = _summarize_struct(payload, top_k=args.top_k)

    # Optional grouping BEFORE model polish (so the model polishes the merged text)
    if args.group:
        summary["bullets"] = _group_bullets(summary["bullets"])
        raw_bullets = [b["text"] for b in summary["bullets"]]
        raw_cites = [b.get("citation") for b in summary["bullets"]]

    # Optional model polish
    use_model = (args.model is not None) and (not args.no_model)
    if use_model:
        refined = _refine_bullets_with_model(
            bullets=raw_bullets,
            title=summary.get("title") or "",
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_new
        )
        # Reattach to summary in order, preserving citations
        for i, b in enumerate(summary["bullets"]):
            b["text"] = refined[i] if i < len(refined) else b["text"]

    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.format == "md":
        print(format_markdown(summary))
    else:
        print(format_cli(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
