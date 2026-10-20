# Translations project

The pipeline is `pipeline.py`. README.md explains how to use it.

## Disk space
- **Stop if the hard drive drops below 13 GB free** (`df -h ~`). Do not start or continue a job below
  that. `pipeline.py` checks this before each page or chunk and stops by itself.

## Cleanup after each book
- When a book is finished (outputs built and checked), delete its temporary page images in this
  project: `books/<slug>/pages/*.jpg`.
- **Never delete or modify the original source** (the PDF or HEIC folder wherever it lives, e.g.
  `/Volumes/Oma/...`). Only the working copies under `books/<slug>/pages/` are removed.
- Keep `books/<slug>/extract/`, `chunks/`, `translate/`, `merged.md`, `pages.json`, `book.json`,
  `pages/sources.tsv` and `out/`: they are the record of the work and are needed to rebuild.

## Git / GitHub after each book
- The user has authorised this (2026-10-16): when a book is finished, commit and push to GitHub.
- **Never commit images** (page JPEGs, HEIC, PDFs, crops). Commit only text outputs: `out/*.md`,
  `out/index.html`, `out/*.epub`, plus `extract/`, `chunks/`, `translate/`, `merged.md`, `book.json`,
  `pages.json`, and the pipeline code. `.gitignore` enforces this. Check `git status` before each commit.
