"""Download Tesla's 'The Inventions, Researches and Writings of Nikola Tesla'
(Project Gutenberg #39272), strip the license header/footer and book clutter,
and save it as data/input.txt.

Run:  uv run python -m tinygpt.download_tesla
"""
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/ebooks/39272.txt.utf-8"
OUT = Path(__file__).resolve().parents[2] / "data" / "input.txt"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    # Gutenberg sometimes times out, so try a few times before giving up.
    request = urllib.request.Request(URL, headers={"User-Agent": "tinygpt-course/0.1"})
    for attempt in range(1, 4):
        try:
            raw = urllib.request.urlopen(request, timeout=60).read().decode("utf-8-sig")
            break
        except OSError as err:
            print(f"Download attempt {attempt} failed: {err}")
            if attempt == 3:
                sys.exit("Project Gutenberg is not responding; try again in a few minutes.")
            time.sleep(10)
    raw = raw.replace("\r\n", "\n")

    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    if not (start and end):
        sys.exit("Could not find the Gutenberg START/END markers; not saving anything.")
    body = raw[start.end():end.start()]

    # Book clutter that isn't Tesla's writing:
    body = body[body.find("[Illustration: Nikola Tesla]"):]  # proofreader credit before the title page
    body = body[:body.rfind("\nINDEX.\n")]                     # index at the back
    body = re.sub(r"\[Illustration[^\]]*\]", "", body)         # picture captions
    body = body.replace("_", "")                               # _italic_ markers
    body = re.sub(r"[ \t]+\n", "\n", body)                     # trailing spaces
    body = re.sub(r"\n{4,}", "\n\n\n", body).strip() + "\n"    # big blank gaps

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(body, encoding="utf-8", newline="\n")
    print(f"Saved {len(body):,} characters to {OUT}")


if __name__ == "__main__":
    main()
