# Scan → transcription → English translation pipeline

`pipeline.py` (stdlib Python + `pdftoppm`, `sips`, `pandoc`) turns a scanned book into:

- `out/<slug>.original.md`: full transcription, page starts marked `{p. 23}`
- `out/<slug>.en.md`: English translation, same page marks
- `out/index.html`: reader with English / Original / Parallel views, margin page numbers
  (each links to its scan; can be hidden), contents drawer, go-to-page, font size, dark mode,
  remembers reading position. `index.html?view=par` opens in parallel view.
- `out/<slug>.epub` (English) and `out/<slug>.original.epub`, with EPUB3 page-list markers

## Backends

Each page or chunk is sent as its own non-interactive call. **agy** is used until its Gemini quota
(`agy -p /usage`) drops below 10% or it returns an empty reply. After that, **codex** is used
until agy's reset time. If both are out, the run stops cleanly. Rerun the same command to resume.

| task      | agy                       | codex                         |
|-----------|---------------------------|-------------------------------|
| extract   | `gemini-3.8-flash-low`    | `gpt-6.1-sol`, effort low     |
| translate | `gemini-3.8-flash-medium` | `gpt-6.1-sol`, effort low     |

Change these in `MODELS` at the top of `pipeline.py`, or override them per run with `--agy-model`,
`--codex-model`, `--codex-effort` or `--backend agy|codex`.
Quota blocks are kept in `.backend_state.json`. Which backend did each page or chunk is logged in
`books/<slug>/logs/provenance.tsv`.

## Steps

```bash
# 1. Pages: from a PDF, or from a folder of HEIC/JPG/PNG/TIFF images (sorted by name)
python3 pipeline.py prepare book.pdf mybook --title "原題" --title-en "English Title" \
    --author "Author" --lang ja \
    --format-note "The original is in the Japanese language, printed vertically. Preserve old-form characters."
#    (--first/--last for a page range; images are scaled to a 2800 px longest edge)

# 2. Everything else (resumable):
python3 pipeline.py run mybook -j 3

# or step by step:
python3 pipeline.py extract mybook -j 3 [--pages 1-20]   # page -> extract/p0001.md
python3 pipeline.py merge mybook                         # -> merged.md with [[pN]] markers
python3 pipeline.py chunk mybook [--chars 1800]          # -> chunks/c0001.md (whole paragraphs)
python3 pipeline.py translate mybook -j 3                # -> translate/c0001.md
python3 pipeline.py build mybook                         # -> out/
python3 pipeline.py status mybook
python3 pipeline.py clean mybook                         # delete working pages/*.jpg (original untouched)
```

## How the text is kept together

- The extraction prompt (`prompts/extract.txt`) returns `PRINTED_PAGE: n`, the body as Markdown, footnotes
  after `---notes---`, and `<<CONTINUES>>` when the last paragraph runs on to the next page. It is
  told to ignore watermarks, library stamps (e.g. HathiTrust) and anything outside the page margins.
- `merge` joins a paragraph split across a page break into one paragraph, with an inline `[[pN]]` at the
  join. It places that page's footnotes (as a `::: notes` block) after the paragraph ends.
- `chunk` cuts only between paragraphs and prefers cuts before headings. The target is about 1800
  characters for CJK text and 5000 for alphabetic text.
- `translate` sends each chunk with the two paragraphs before it and the one after it as context.
  The reply is checked: every `[[pN]]` marker and notes block must come back, and the reply must not be
  too short. Otherwise the chunk is retried, and after two failures it moves to the other backend.
  A chunk is retranslated automatically if its source text changes (`.sha` files).
- An optional `books/<slug>/glossary.md` is added to every translation prompt, for consistent names and terms.

Edit `prompts/*.txt` to change the instructions. Placeholders: `{IMAGE} {PAGE} {LANG_HINT}
{FORMAT_NOTE}` and `{TEXT} {CONTEXT_BEFORE} {CONTEXT_AFTER} {GLOSSARY}`.
