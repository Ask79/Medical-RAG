#!/usr/bin/env python3
import argparse, json, os, sys, re
from collections import defaultdict, Counter

WS = re.compile(r"\s+")
def norm_name(s):
    if not s: return None
    s = str(s)
    # strip common marks and punctuation except spaces
    s = s.replace("®","").replace("™","").replace("℞","")
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
    s = WS.sub(" ", s).strip().lower()
    return s or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--names-out", dest="names_out", required=True)
    ap.add_argument("--spl-names-out", dest="spl_names_out", required=True)
    ap.add_argument("--report", dest="report_path", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.names_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.spl_names_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    name_hits = Counter()     # normalized name → chunk hits
    name_spl_hits = Counter() # normalized name → unique SPL count
    spl_to_brand = defaultdict(set)
    spl_to_generic = defaultdict(set)

    total_chunks = 0
    chunks_with_any_name = 0
    total_spl = set()

    with open(args.in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_chunks += 1
            spl = rec.get("spl_set_id"); 
            if not spl: continue
            total_spl.add(spl)

            b = rec.get("brand_name")
            g = rec.get("generic_name")
            anyname = False

            for raw, target in ((b, spl_to_brand), (g, spl_to_generic)):
                if raw is None: continue
                vals = raw if isinstance(raw, list) else [raw]
                seen_this_chunk = set()
                for v in vals:
                    n = norm_name(v)
                    if not n: continue
                    target[spl].add(n)
                    if n not in seen_this_chunk:
                        name_hits[n] += 1
                        seen_this_chunk.add(n)
                        anyname = True
            if anyname:
                chunks_with_any_name += 1

    for spl, names in spl_to_brand.items():
        for n in names:
            name_spl_hits[n] += 1
    for spl, names in spl_to_generic.items():
        for n in names:
            name_spl_hits[n] += 1

    with open(args.names_out, "w", encoding="utf-8") as fout:
        fout.write("name_norm\tname_type\ttotal_chunk_hits\tunique_spl_count\n")
        for dct, t in ((name_spl_hits, "mixed"),):
            # write combined inventory, caller can filter later
            for n, spl_count in sorted(dct.items(), key=lambda x: (-x[1], x[0])):
                fout.write(f"{n}\t{t}\t{name_hits.get(n,0)}\t{spl_count}\n")

    with open(args.spl_names_out, "w", encoding="utf-8") as fout:
        fout.write("spl_set_id\tbrand_names_norm\tgeneric_names_norm\n")
        for spl in sorted(set(list(spl_to_brand.keys())+list(spl_to_generic.keys()))):
            b_list = sorted(spl_to_brand.get(spl, []))
            g_list = sorted(spl_to_generic.get(spl, []))
            fout.write(f"{spl}\t{','.join(b_list)}\t{','.join(g_list)}\n")

    spl_with_name = sum(1 for s in set(list(spl_to_brand.keys())+list(spl_to_generic.keys())) if (spl_to_brand[s] or spl_to_generic[s]))
    with open(args.report_path, "w", encoding="utf-8") as fout:
        fout.write("=== Names Inventory / Coverage ===\n")
        fout.write(f"Total chunks: {total_chunks}\n")
        fout.write(f"Chunks with ≥1 name: {chunks_with_any_name} ({(chunks_with_any_name/max(1,total_chunks))*100:.2f}%)\n")
        fout.write(f"Unique SPL_SET_IDs: {len(total_spl)}\n")
        fout.write(f"SPLs with ≥1 brand/generic name: {spl_with_name} ({(spl_with_name/max(1,len(total_spl)))*100:.2f}%)\n")

if __name__ == "__main__":
    main()
