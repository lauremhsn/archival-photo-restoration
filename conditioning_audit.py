"""
What conditioning signal is actually available in the Matson metadata?

Coverage says how many records have text. This says whether that text carries
colour-relevant information or archival boilerplate.

python conditioning_audit.py
"""

import csv
import re
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "matson.csv"
FIG_DIR = "figures"
FIELDS = ["title", "subject", "location", "date", "notes"]

BOILERPLATE = [
    "title from", "general information about", "gift;", "forms part of",
    "on transparency", "caption on", "taken either by", "date from",
    "no known restrictions", "published in", "see also", "transferred from",
    "purchase;", "copy print", "american colony photo department",
    "matson photo service", "lot ", "negative sleeve", "attributed to",
]

DESCRIPTIVE = ["showing", "view of", "looking", "in the vicinity", "photograph taken",
               "in the background", "in the foreground", "taken from"]

COLOUR_TERMS = {
    "vegetation": r"oleander|flower|bloom|blossom|tree|palm|garden|grove|orchard|"
                  r"vine|grass|field|wheat|olive|cedar|fig|cactus|shrub|forest|crop",
    "stone/earth": r"rock|sandstone|stone|cliff|sand|desert|mountain|ruin|wall|"
                   r"marble|cave|gorge|canyon|hill|quarry|valley|cave",
    "water/sky": r"sea|river|lake|pool|spring|sky|cloud|snow|rain|water|shore|bay|"
                 r"waterfall|stream|harbor|harbour",
    "built": r"temple|tomb|mosque|church|dome|roof|tent|house|street|market|bazaar|"
             r"gate|bridge|tower|monastery|castle|fort|wall|building|camp",
    "people/dress": r"costume|dress|robe|headscarf|veil|bedouin|woman|women|man\b|men\b|"
                    r"group|soldier|priest|monk|child|shepherd|pilgrim|worker",
    "light/time": r"sunset|sunrise|dawn|dusk|night|shadow|moonlight|winter|summer|spring",
    "explicit colour": r"\bred\b|\bblue\b|\bgreen\b|\byellow\b|\bgolden\b|\bpink\b|\bbrown\b",
}

STOP = set("the a an of in on at and or to from with by for near".split())


def load():
    with open(CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2]


def is_boilerplate(line):
    low = line.lower()
    return any(p in low for p in BOILERPLATE)


def is_descriptive(line):
    low = line.lower()
    return any(p in low for p in DESCRIPTIVE)


def main():
    rows = load()
    n = len(rows)
    print(f"{n} records from {CSV}\n")

    print(f"{'field':<10} {'present':>8} {'rate':>7} {'med words':>10} {'vocab':>7}")
    print("-" * 46)
    for f in FIELDS:
        vals = [r.get(f, "") for r in rows if r.get(f)]
        if not vals:
            continue
        mw = int(statistics.median([len(v.split()) for v in vals]))
        vocab = len({w for v in vals for w in words(v)})
        print(f"{f:<10} {len(vals):>8} {len(vals)/n*100:>6.1f}% {mw:>10} {vocab:>7}")

    print("\nWHAT IS IN notes")
    lines, boiler, desc, other = [], 0, 0, []
    for r in rows:
        for line in (r.get("notes") or "").split("; "):
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if is_boilerplate(line):
                boiler += 1
            elif is_descriptive(line):
                desc += 1
            else:
                other.append(line)
    total_lines = len(lines)
    if total_lines:
        print(f"{total_lines} note lines across {n} records "
              f"({total_lines/n:.1f} per record)")
        print(f"  archival boilerplate {boiler} ({boiler/total_lines*100:.1f}%)")
        print(f"  scene description    {desc} ({desc/total_lines*100:.1f}%)")
        print(f"  unclassified         {len(other)} ({len(other)/total_lines*100:.1f}%)")

        print("\n  most repeated note lines:")
        for line, c in Counter(lines).most_common(6):
            print(f"    {c:>4}x  {line[:66]}")

        if other:
            print("\n  sample unclassified lines:")
            for line in list(dict.fromkeys(other))[:5]:
                print(f"    {line[:70]}")

    recs_with_desc = sum(
        1 for r in rows
        if any(is_descriptive(l) for l in (r.get("notes") or "").split("; ")))
    print(f"\nrecords with at least one descriptive note line: "
          f"{recs_with_desc} ({recs_with_desc/n*100:.1f}%)")

    print("\nCOLOUR-RELEVANT VOCABULARY")
    print("(share of records whose title+subject+notes mention each category)")
    print(f"{'category':<18} {'records':>8} {'rate':>7}")
    print("-" * 36)
    cat_counts = {}
    for cat, pat in COLOUR_TERMS.items():
        c = sum(1 for r in rows
                if re.search(pat, " ".join([r.get("title", ""), r.get("subject", ""),
                                            r.get("notes", "")]).lower()))
        cat_counts[cat] = c
        print(f"{cat:<18} {c:>8} {c/n*100:>6.1f}%")

    any_cat = sum(
        1 for r in rows
        if any(re.search(p, " ".join([r.get("title", ""), r.get("subject", ""),
                                      r.get("notes", "")]).lower())
               for p in COLOUR_TERMS.values()))
    print("-" * 36)
    print(f"{'any category':<18} {any_cat:>8} {any_cat/n*100:>6.1f}%")

    places = set()
    for r in rows:
        places.update(words(r.get("location", "")))
    places.update(words("palestine transjordan jordan egypt sinai jerusalem petra"))

    pure_place, has_extra = 0, 0
    for r in rows:
        tw = set(words(r.get("title", "")))
        if not tw:
            continue
        if tw <= places:
            pure_place += 1
        else:
            has_extra += 1
    print(f"\nTITLES")
    print(f"  only place names:        {pure_place} ({pure_place/n*100:.1f}%)")
    print(f"  carry extra description: {has_extra} ({has_extra/n*100:.1f}%)")

    freq = Counter()
    for r in rows:
        freq.update(w for w in words(r.get("title", "")) if w not in places)
    print("\n  most common non-place title words:")
    for w, c in freq.most_common(15):
        print(f"    {c:>4}  {w}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    cats = list(cat_counts)
    ax[0].barh(cats, [cat_counts[c] / n * 100 for c in cats])
    ax[0].set_xlabel("% of records mentioning")
    ax[0].set_title("colour-relevant vocabulary by category")
    ax[0].invert_yaxis()

    top = freq.most_common(15)
    ax[1].barh([w for w, _ in top][::-1], [c for _, c in top][::-1])
    ax[1].set_xlabel("records")
    ax[1].set_title("most common non-place title words")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/conditioning_audit.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {FIG_DIR}/conditioning_audit.png")


if __name__ == "__main__":
    main()