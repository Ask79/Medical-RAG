#!/usr/bin/env python3
"""
Build SPL→names mapping from DailyMed master JSONL by parsing title and fallbacks.

Outputs:
  --spl-names-out  TSV: spl_set_id, brand_names_norm, generic_names_norm
  --names-out      TSV: name_norm, name_type, unique_spl_count
  --report         coverage summary

Heuristics:
- Parse `title` like "RENESE® (polythiazide) TABLETS ..." → brand="renese", generic="polythiazide"
- Strip marks (® ™), collapse spaces, lowercase
- Fallback to `sections.unclassified.DESCRIPTION` first line in parentheses, if present
"""
import argparse, json, os, re
from collections import defaultdict, Counter

WS = re.compile(r"\s+")
PARENS = re.compile(r"\(([^)]+)\)")
MARKS = str.maketrans({"®":"", "™":"", "℞":""})

def norm(s):
    if not s: return None
    s = str(s).translate(MARKS)
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = WS.sub(" ", s).strip().lower()
    return s or None

def parse_from_title(title):
    if not title: return (None, None)
    t = title.replace("\n", " ")
    # generic often inside first parentheses
    m = PARENS.search(t)
    generic = norm(m.group(1)) if m else None
    # brand is leading token(s) before first parenthesis or dosage keywords
    stopper = m.start() if m else len(t)
    lead = t[:stopper]
    # remove common dosage/form phrases
    lead = re.sub(r"\b(tablets?|capsules?|injection|oral|solution|cream|ointment|gel|suspension|spray|for .*|usp|rx only)\b",
                  " ", lead, flags=re.IGNORECASE)
    brand = norm(lead)
    # brand may still be multi-word; keep as-is (RxNorm name join handles it)
    return (brand, generic)

def parse_from_description(desc):
    # try to find "(<generic>)" in the first 200 chars
    if not desc: return (None, None)
    s = desc.splitlines()[0][:200]
    m = PARENS.search(s)
    g = norm(m.group(1)) if m else None
    return (None, g)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--spl-names-out", dest="spl_names_out", required=True)
    ap.add_argument("--names-out", dest="names_out", required=True)
    ap.add_argument("--report", dest="report_path", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.spl_names_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.names_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    spl_to_brand = defaultdict(set)
    spl_to_generic = defaultdict(set)
    name_spl = Counter()

    total = 0
    with open(args.in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            spl = rec.get("set_id") or rec.get("spl_set_id")
            if not spl: continue

            title = rec.get("title")
            brand, generic = parse_from_title(title)

            # fallback: look into sections.unclassified.DESCRIPTION
            if not generic:
                sections = rec.get("sections") or {}
                if isinstance(sections, dict) and isinstance(sections.get("unclassified"), dict):
                    desc = sections["unclassified"].get("DESCRIPTION")
                    _, g2 = parse_from_description(desc)
                    if g2: generic = g2

            if brand: 
                spl_to_brand[spl].add(brand)
                name_spl[("brand", brand)] += 1
            if generic:
                spl_to_generic[spl].add(generic)
                name_spl[("generic", generic)] += 1

    with open(args.spl_names_out, "w", encoding="utf-8") as fout:
        fout.write("spl_set_id\tbrand_names_norm\tgeneric_names_norm\n")
        for spl in sorted(set(list(spl_to_brand.keys())+list(spl_to_generic.keys()))):
            b_list = sorted(spl_to_brand.get(spl, []))
            g_list = sorted(spl_to_generic.get(spl, []))
            fout.write(f"{spl}\t{','.join(b_list)}\t{','.join(g_list)}\n")

    with open(args.names_out, "w", encoding="utf-8") as fout:
        fout.write("name_norm\tname_type\tunique_spl_count\n")
        for (typ, n), cnt in sorted(name_spl.items(), key=lambda x: (-x[1], x[0][1])):
            fout.write(f"{n}\t{typ}\t{cnt}\n")

    spl_with_any = sum(1 for s in set(list(spl_to_brand.keys())+list(spl_to_generic.keys())) 
                       if spl_to_brand[s] or spl_to_generic[s])
    with open(args.report_path, "w", encoding="utf-8") as fout:
        fout.write("=== Names from Masters / Coverage ===\n")
        fout.write(f"Total master records: {total}\n")
        fout.write(f"SPLs with ≥1 name: {spl_with_any} ({(spl_with_any/max(1,total))*100:.2f}%)\n")

if __name__ == "__main__":
    main()
