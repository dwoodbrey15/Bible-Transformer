"""Build Bible-only byte streams with a deterministic chapter-level holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCE = "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/json/KJV.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(raw, out, source):
    payload = json.loads(raw)
    if not payload.get("translation", "").startswith("KJV:"):
        raise ValueError("Expected scrollmapper KJV JSON, not another translation")
    books = payload["books"]
    if len(books) != 66:
        raise ValueError("Expected the 66-book Protestant KJV canon")
    chapters = []
    verse_count = 0
    for bi, book in enumerate(books):
        for chapter in book["chapters"]:
            verses = chapter["verses"]
            verse_count += len(verses)
            text = "\n".join(v["text"].strip() for v in verses)
            if any(tag in text for tag in ("<", ">", "Strong's", "http://", "https://")):
                raise ValueError("Markup or metadata found; inspect the source before training")
            key = f"{bi}:{chapter['chapter']}"
            chapters.append((key, text))
    if len(chapters) != 1189 or verse_count != 31102:
        raise ValueError(f"Unexpected canon: {len(chapters)} chapters, {verse_count} verses")
    # Select 59 complete chapters, then preserve canonical order in each stream.
    held = set(sorted((k for k, _ in chapters), key=lambda k: sha(("kjv-1m:42:" + k).encode()))[:59])
    streams = {
        "train": "\n\n".join(t for k, t in chapters if k not in held).encode("utf-8"),
        "val": "\n\n".join(t for k, t in chapters if k in held).encode("utf-8"),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, data in streams.items():
        (out / f"{name}.bin").write_bytes(data)
    manifest = {
        "source": source, "translation": payload["translation"], "raw_sha256": sha(raw),
        "canon": "66 books; no Apocrypha", "chapters": len(chapters), "verses": verse_count,
        "preprocessing": "Verse text only; strip surrounding whitespace; newline between verses; blank line between chapters. No headings, references, annotations, or external training text.",
        "tokenizer": "UTF-8 bytes, vocabulary 0..255; no learned tokenizer",
        "validation_chapters": sorted(held), "split_seed": "kjv-1m:42",
        "streams": {k: {"bytes": len(v), "sha256": sha(v)} for k, v in streams.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", help="Local scrollmapper-format KJV JSON; otherwise download")
    p.add_argument("--out", type=Path, default=Path("data"))
    a = p.parse_args()
    raw = Path(a.source).read_bytes() if a.source else urllib.request.urlopen(SOURCE, timeout=60).read()
    result = prepare(raw, a.out, a.source or SOURCE)
    print(json.dumps(result["streams"], indent=2))
