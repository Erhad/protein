#!/usr/bin/env python3
"""
Drop PDFs in thesis/papers/ then run:
    python3 thesis/sync_papers.py

Posts each PDF to the local Zotero app via its connector API.
Zotero auto-recognizes metadata (same engine as drag-and-drop).
Then exports thesisreferences.bib directly via Better BibTeX.
No clicking required — just needs Zotero open.
"""

import os, json, time, uuid, urllib.request

PAPERS_DIR     = os.path.join(os.path.dirname(__file__), "papers")
BIB_FILE       = os.path.join(os.path.dirname(__file__), "aalto", "thesisreferences.bib")
SYNCED_FILE    = os.path.join(os.path.dirname(__file__), ".synced_papers.json")
ZOTERO_LOCAL   = "http://localhost:23119"
BBT_EXPORT_URL = f"{ZOTERO_LOCAL}/better-bibtex/export/library?/library.biblatex"


def zotero_is_running():
    try:
        urllib.request.urlopen(f"{ZOTERO_LOCAL}/connector/ping", timeout=3)
        return True
    except:
        return False


def upload_pdf(path):
    """POST PDF to local Zotero, trigger auto-recognition. Returns session_id."""
    session_id = str(uuid.uuid4())
    with open(path, "rb") as f:
        data = f.read()

    metadata = json.dumps({
        "sessionID": session_id,
        "url": f"file://{os.path.abspath(path)}",
        "title": os.path.basename(path),
    })
    req = urllib.request.Request(
        f"{ZOTERO_LOCAL}/connector/saveStandaloneAttachment",
        data=data,
        headers={
            "Content-Type": "application/pdf",
            "X-Metadata": metadata,
            "Content-Length": str(len(data)),
        }
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read().decode())
        return session_id, resp.get("canRecognize", False)


def poll_recognized(session_id, timeout=20):
    """Wait for Zotero to finish recognizing. Returns title or None."""
    req = urllib.request.Request(
        f"{ZOTERO_LOCAL}/connector/getRecognizedItem",
        data=json.dumps({"sessionID": session_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 204:
                    time.sleep(1)
                    continue
                result = json.loads(r.read().decode())
                return result.get("title")
        except urllib.request.HTTPError as e:
            if e.code == 204:
                time.sleep(1)
                continue
            raise
    return None


def load_synced():
    if os.path.exists(SYNCED_FILE):
        with open(SYNCED_FILE) as f:
            return set(json.load(f))
    return set()


def save_synced(synced):
    with open(SYNCED_FILE, "w") as f:
        json.dump(sorted(synced), f, indent=2)


def run():
    if not zotero_is_running():
        print("Zotero is not running — please open it first.")
        return

    synced = load_synced()
    print(f"  {len(synced)} papers already synced")

    pdfs = sorted(f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf"))
    added, skipped, failed = [], [], []

    for pdf in pdfs:
        path = os.path.join(PAPERS_DIR, pdf)
        print(f"\n{pdf[:70]}")

        if pdf in synced:
            print("  [SKIP] already synced")
            skipped.append(pdf)
            continue

        try:
            session_id, can_recognize = upload_pdf(path)
        except Exception as e:
            print(f"  [FAIL] upload error: {e}")
            failed.append(pdf)
            continue

        if not can_recognize:
            print("  [WARN] uploaded but Zotero can't recognize — will be standalone attachment")
        else:
            title = poll_recognized(session_id)
            if title:
                print(f"  [OK] {title[:65]}")
            else:
                print("  [WARN] uploaded but recognition timed out — check Zotero")

        added.append(pdf)
        synced.add(pdf)
        save_synced(synced)
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Added {len(added)}  |  Skipped {len(skipped)}  |  Failed {len(failed)}")
    if failed:
        print("\nFailed:")
        for f in failed:
            print(f"  {f}")

    # Export bib directly via Better BibTeX (no sync needed)
    print("\nExporting thesisreferences.bib via Better BibTeX...")
    time.sleep(2)  # give Zotero a moment to finish processing
    try:
        with urllib.request.urlopen(BBT_EXPORT_URL, timeout=30) as r:
            bib = r.read().decode()
        with open(BIB_FILE, "w") as f:
            f.write(bib)
        entry_count = bib.count("\n@")
        print(f"  Written {entry_count} entries to {os.path.basename(BIB_FILE)}")
    except Exception as e:
        print(f"  [WARN] BBT export failed: {e}")
        print("  Open Zotero → sync → bib will update automatically")


if __name__ == "__main__":
    run()
