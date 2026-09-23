#!/usr/bin/env python3
"""
curriculum/build.py
===================
Reads all raw curriculum JSON files from curriculum/raw/class_*/subject/topics_*.json
and builds:
  - curriculum.db  (SQLite with FTS5 virtual table)
  - curriculum.json (flat list of all records, for reference)
  - manifest.json   (SHA-256 of curriculum.db for integrity verification)

Schema (multi-class / multi-language / multi-curriculum):
  curriculum_content
    id              TEXT PRIMARY KEY   (UUID)
    class_level     INTEGER            (6, 7, 8, 9, 10, ...)
    subject         TEXT               ('science', 'mathematics', 'english')
    language        TEXT               ('en', 'hi', 'or', 'mr')
    curriculum      TEXT               ('NCERT', 'SCERT-Odisha', ...)
    chapter         TEXT
    topic           TEXT
    content         TEXT
    source_ref      TEXT
    content_version TEXT DEFAULT '1.0'

  curriculum_fts (FTS5 virtual table — full-text search on content)

Usage:
  python build.py
"""

import json
import sqlite3
import hashlib
import uuid
import pathlib
import sys
from datetime import datetime, timezone

BASE_DIR   = pathlib.Path(__file__).parent
RAW_DIR    = BASE_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH    = OUTPUT_DIR / "curriculum.db"
JSON_PATH  = OUTPUT_DIR / "curriculum.json"
MANIFEST   = OUTPUT_DIR / "manifest.json"

CONTENT_VERSION = "1.0"


def read_raw_records() -> list[dict]:
    """Walk raw/ directory and load every topics_*.json file."""
    records = []
    if not RAW_DIR.exists():
        print(f"[ERROR] raw/ directory not found at {RAW_DIR}")
        sys.exit(1)

    for json_file in sorted(RAW_DIR.rglob("topics_*.json")):
        try:
            with open(json_file, encoding="utf-8-sig") as f:   # utf-8-sig handles BOM
                data = json.load(f)
            if not isinstance(data, list):
                print(f"  [WARN] {json_file} is not a list — skipping")
                continue
            for item in data:
                # Assign a stable UUID based on content hash so re-running is idempotent
                key = f"{item.get('class_level')}-{item.get('subject')}-{item.get('language')}-{item.get('chapter')}-{item.get('topic')}"
                item["id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
                item.setdefault("content_version", CONTENT_VERSION)
                records.append(item)
            print(f"  Loaded {len(data):3d} records from {json_file.relative_to(BASE_DIR)}")
        except Exception as e:
            print(f"  [WARN] Could not parse {json_file}: {e}")
    return records


REQUIRED_FIELDS = {"class_level", "subject", "language", "curriculum",
                   "chapter", "topic", "content"}


def validate(records: list[dict]) -> list[dict]:
    valid = []
    for r in records:
        missing = REQUIRED_FIELDS - r.keys()
        if missing:
            print(f"  [SKIP] Missing fields {missing} in record: {r.get('topic', '?')}")
        else:
            valid.append(r)
    return valid


def build_db(records: list[dict]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Main content table
    cur.executescript("""
        CREATE TABLE curriculum_content (
            id              TEXT PRIMARY KEY,
            class_level     INTEGER NOT NULL,
            subject         TEXT    NOT NULL,
            language        TEXT    NOT NULL,
            curriculum      TEXT    NOT NULL,
            chapter         TEXT    NOT NULL,
            topic           TEXT    NOT NULL,
            content         TEXT    NOT NULL,
            source_ref      TEXT,
            content_version TEXT    NOT NULL DEFAULT '1.0'
        );

        -- Index for fast filtered lookups (agent: class + subject + language)
        CREATE INDEX idx_class_subject_lang
            ON curriculum_content(class_level, subject, language);

        -- FTS5 virtual table — searches content column
        CREATE VIRTUAL TABLE curriculum_fts
            USING fts5(content, content=curriculum_content, content_rowid=rowid);
    """)

    insert_sql = """
        INSERT OR REPLACE INTO curriculum_content
            (id, class_level, subject, language, curriculum, chapter, topic, content, source_ref, content_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = [
        (
            r["id"], r["class_level"], r["subject"], r["language"],
            r["curriculum"], r["chapter"], r["topic"], r["content"],
            r.get("source_ref", ""), r.get("content_version", CONTENT_VERSION)
        )
        for r in records
    ]
    cur.executemany(insert_sql, rows)

    # Populate FTS5 index
    cur.execute("INSERT INTO curriculum_fts(curriculum_fts) VALUES('rebuild')")

    con.commit()
    con.close()


def write_json(records: list[dict]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_manifest(records: list[dict]) -> None:
    sha256 = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
    manifest = {
        "db_sha256":      sha256,
        "record_count":   len(records),
        "built_at":       datetime.now(timezone.utc).isoformat(),
        "content_version": CONTENT_VERSION,
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  SHA-256: {sha256}")


def print_summary(records: list[dict]) -> None:
    from collections import Counter
    classes  = Counter(r["class_level"] for r in records)
    subjects = Counter(r["subject"]      for r in records)
    langs    = Counter(r["language"]     for r in records)
    print(f"\n{'='*55}")
    print(f"  Total records  : {len(records)}")
    print(f"  Classes        : {dict(sorted(classes.items()))}")
    print(f"  Subjects       : {dict(sorted(subjects.items()))}")
    print(f"  Languages      : {dict(sorted(langs.items()))}")
    print(f"  Output DB      : {DB_PATH}")
    print(f"  Output JSON    : {JSON_PATH}")
    print(f"  Manifest       : {MANIFEST}")
    print(f"{'='*55}\n")


def main():
    print("\n=== Curriculum Build Pipeline ===\n")
    print("Reading raw files...")
    records = read_raw_records()
    print(f"\nValidating {len(records)} records...")
    records = validate(records)
    print(f"\nBuilding SQLite + FTS5 at {DB_PATH} ...")
    build_db(records)
    print("Writing JSON export...")
    write_json(records)
    print("Writing manifest (SHA-256)...")
    write_manifest(records)
    print_summary(records)
    print("Build complete.\n")


if __name__ == "__main__":
    main()
