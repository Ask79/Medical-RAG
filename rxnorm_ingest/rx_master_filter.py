import json

INFILE  = r"data/processed/normalized/rxnorm/rx_master_v1.jsonl"
OUTFILE = r"data/processed/normalized/rxnorm/rx_master_v1_cleaned.jsonl"

ALLOWED_TTYS_NON_RX = {"IN", "PIN", "MIN", "BN"}

kept = 0
dropped = 0

with open(INFILE, "r", encoding="utf-8") as fin, open(OUTFILE, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        sab = obj.get("sab_sources", [])
        tty = obj.get("tty", "")

        # keep everything that includes RXNORM
        if "RXNORM" in sab:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
            continue

        # otherwise, apply non-RxNorm whitelist
        if tty in ALLOWED_TTYS_NON_RX:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
        else:
            dropped += 1

print(f"Kept: {kept}, Dropped (non-RxNorm disallowed TTYs): {dropped}")
