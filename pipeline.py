#!/usr/bin/env python3
"""Scan-to-translation pipeline: PDF -> page images -> per-page transcription
(agy / codex) -> merged original Markdown -> paragraph-safe chunks -> English
translation (agy / codex) -> original.md, en.md, index.html, epub.

Backends: agy (gemini-3.8-flash-high) is used until its Gemini quota runs low,
then codex. Every step is resumable: finished pages/chunks are skipped.

Usage (see README.md):
  pipeline.py prepare BOOK.pdf|IMAGE_DIR SLUG [--title T --author A --lang ja --format-note "…"]
  pipeline.py extract SLUG [-j 3]
  pipeline.py merge SLUG
  pipeline.py chunk SLUG [--chars N]
  pipeline.py translate SLUG [-j 3]
  pipeline.py build SLUG
  pipeline.py run SLUG            # extract, merge, chunk, translate, build
  pipeline.py status SLUG
"""
import argparse, concurrent.futures as cf, datetime as dt, hashlib, html, json
import os, re, shutil, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOKS = ROOT / "books"
PROMPTS = ROOT / "prompts"
STATE_FILE = ROOT / ".backend_state.json"
MIN_FREE_GB = 13                 # stop all work below this much free disk (project CLAUDE.md)

# Per-task models. Extraction is cheap vision transcription; translation gets more reasoning.
MODELS = {
    "extract":   {"agy": "gemini-3.8-flash-low",    "codex": "gpt-6.1-sol", "codex_effort": "low"},
    "translate": {"agy": "gemini-3.8-flash-medium", "codex": "gpt-6.1-sol", "codex_effort": "low"},
}
AGY_MIN_FRACTION = 0.10          # stop using agy below 10% of the Gemini 5h/weekly limit
USAGE_CACHE_SECS = 90
CALL_TIMEOUT = 600               # seconds per model call; normal pages/sections take 0.5-2 min
QUOTA_WORDS = re.compile(r"usage limit|rate limit|quota|exceeded your|too many requests|429", re.I)

MARK_RE = re.compile(r"\[\[p(\d+)\]\]")
SPLIT_OVERLAP = 0.07             # each half of a two-page photo runs 7% past the middle
HALF_NOTE = ("This image is the {SIDE}-hand page of a photographed two-page opening, cut with a margin of "
             "overlap. Transcribe ONLY the main page, which fills most of the image. A strip of the facing "
             "page may be visible along the {EDGE} edge: ignore it completely, even if it holds part of a "
             "line or a page number.")
LANG_NAMES = {"ja": "Japanese", "zh": "Chinese", "zh-Hant": "Chinese (traditional script)",
              "zh-Hans": "Chinese (simplified script)", "ko": "Korean", "fr": "French",
              "de": "German", "ru": "Russian", "es": "Spanish", "it": "Italian"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class QuotaExhausted(Exception):
    pass


# ---------------------------------------------------------------- backends
class Backends:
    """Picks agy while its Gemini quota lasts, then codex. Shared by worker threads."""

    def __init__(self, mode="auto", codex_model=None, codex_effort=None, agy_model=None):
        self.mode = mode
        # command-line overrides; otherwise the per-task MODELS table applies
        self.codex_model, self.codex_effort, self.agy_model = codex_model, codex_effort, agy_model
        self.lock = threading.Lock()
        self._usage_at = 0
        self._usage_ok = True
        self.state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

    def _save(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=1))

    def _blocked(self, name):
        until = self.state.get(name + "_blocked_until")
        return bool(until) and time.time() < until

    def block(self, name, until=None, why=""):
        with self.lock:
            until = until or time.time() + 3600
            self.state[name + "_blocked_until"] = until
            self._save()
        log(f"[quota] {name} blocked until {dt.datetime.fromtimestamp(until):%Y-%m-%d %H:%M} ({why})")

    def agy_usage(self):
        """Return (min remaining fraction for Gemini, reset epoch of the limiting bucket)."""
        r = subprocess.run(["agy", "-p", "/usage", "--output-format", "json"],
                           capture_output=True, text=True, timeout=120)
        data = json.loads(r.stdout)
        worst, reset = 1.0, None
        for g in data["command"]["data"]["groups"]:
            if g["name"].startswith("Gemini Models"):
                for b in g["buckets"]:
                    f = b["remaining_fraction"]
                    if f <= worst:
                        worst = f
                        reset = dt.datetime.fromisoformat(b["reset_time"].replace("Z", "+00:00")).timestamp()
        return worst, reset

    def _agy_ok(self):
        if self._blocked("agy"):
            return False
        if time.time() - self._usage_at < USAGE_CACHE_SECS:
            return self._usage_ok
        try:
            frac, reset = self.agy_usage()
        except Exception as e:
            log(f"[quota] could not read agy /usage ({e}); trying agy anyway")
            frac, reset = 1.0, None
        self._usage_at = time.time()
        self._usage_ok = frac >= AGY_MIN_FRACTION
        if not self._usage_ok:
            self.state["agy_blocked_until"] = reset or time.time() + 3600
            self._save()
            log(f"[quota] agy Gemini quota at {frac:.0%}; switching to codex until "
                f"{dt.datetime.fromtimestamp(self.state['agy_blocked_until']):%H:%M}")
        return self._usage_ok

    def pick(self, exclude=()):
        with self.lock:
            order = {"auto": ["agy", "codex"], "agy": ["agy"], "codex": ["codex"]}[self.mode]
            for name in order:
                if name in exclude:
                    continue
                if name == "agy" and self._agy_ok():
                    return "agy"
                if name == "codex" and not self._blocked("codex"):
                    return "codex"
        raise QuotaExhausted("no backend has quota left: " + ", ".join(
            f"{n} until {dt.datetime.fromtimestamp(self.state.get(n + '_blocked_until', 0)):%m-%d %H:%M}"
            for n in order if self._blocked(n)))

    # -- calls. Each returns reply text, or raises QuotaExhausted / RuntimeError
    def call_agy(self, prompt, cwd, logfile, task, add_dirs=()):
        cmd = ["taskpolicy", "-b", "agy", "-p", prompt, "--model", self.agy_model or MODELS[task]["agy"],
               "--mode", "accept-edits", "--print-timeout", f"{CALL_TIMEOUT // 60}m", "--output-format", "json",
               "--disable-slash-commands"]
        for d in add_dirs:
            cmd += ["--add-dir", str(d)]
        stdout, stderr = run_until_json(cmd, cwd, CALL_TIMEOUT + 60, logfile)
        Path(logfile).write_text(stdout + "\n--- stderr ---\n" + stderr)
        data = parse_last_json(stdout)
        r = subprocess.CompletedProcess(cmd, 0, stdout, stderr)
        text = (data.get("response") or "").strip()
        if "blocked by Gemini" in text:
            raise RuntimeError("Gemini content filter blocked the request")
        out_tokens = (data.get("usage") or {}).get("output_tokens", 0)
        if not text or not out_tokens:
            # An empty / 0-token reply is how agy shows an exhausted quota: stop, don't retry.
            if "permission" in (r.stdout + r.stderr):
                raise RuntimeError("agy asked for a tool permission (see log)")
            raise QuotaExhausted("agy returned an empty reply")
        return text

    def call_codex(self, prompt, cwd, logfile, task, images=()):
        cmd = ["taskpolicy", "-b", "codex", "exec", "-s", "read-only", "--skip-git-repo-check",
               "--ephemeral", "--color", "never"]
        model = self.codex_model or MODELS[task]["codex"]
        effort = self.codex_effort or MODELS[task]["codex_effort"]
        if model:
            cmd += ["-m", model]
        if effort:
            cmd += ["-c", f'model_reasoning_effort="{effort}"']
        for im in images:
            cmd += ["-i", str(im)]
        outf = Path(str(logfile) + ".last")
        outf.unlink(missing_ok=True)
        cmd += ["-o", str(outf), "-"]
        try:
            r = subprocess.run(cmd, cwd=cwd, input=prompt, capture_output=True, text=True, timeout=CALL_TIMEOUT + 60)
        except subprocess.TimeoutExpired as e:
            Path(logfile).write_text(f"TIMEOUT after {CALL_TIMEOUT + 60}s\n{e.stdout or ''}\n--- stderr ---\n{e.stderr or ''}")
            raise
        Path(logfile).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        text = outf.read_text().strip() if outf.exists() else ""
        if r.returncode != 0 or not text:
            tail = (r.stdout + r.stderr)[-3000:]
            if QUOTA_WORDS.search(tail):
                raise QuotaExhausted("codex reported a usage limit")
            raise RuntimeError(f"codex exit {r.returncode}, empty={not text}")
        return text

    def run(self, prompt, cwd, logbase, task, images=(), validate=None, tries=3):
        """Run on the best backend; fall over to the other on quota/validation failure."""
        exclude, last_err = set(), None
        for attempt in range(tries):
            name = self.pick(exclude)
            logfile = f"{logbase}.{name}.{attempt}.log"
            try:
                if name == "agy":
                    text = self.call_agy(prompt, cwd, logfile, task, add_dirs={Path(i).parent for i in images})
                else:
                    text = self.call_codex(prompt, cwd, logfile, task, images=images)
                text = strip_fences(text)
                if validate:
                    problem = validate(text)
                    if problem:
                        raise RuntimeError("validation: " + problem)
                return name, text
            except QuotaExhausted as e:
                self.block(name, self._reset_hint(name), str(e))
                exclude.add(name)
                last_err = e
                if self.mode != "auto" or len(exclude) == 2:
                    raise QuotaExhausted(str(e))
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                log(f"  ! {Path(logbase).name} via {name}: {e} (attempt {attempt + 1})")
                last_err = e
                if attempt >= 1 and self.mode == "auto":
                    exclude.add(name)       # second failure on one backend: try the other
                    if len(exclude) == 2:
                        exclude = set()
        raise RuntimeError(f"gave up after {tries} attempts: {last_err}")

    def _reset_hint(self, name):
        if name == "agy":
            try:
                return self.agy_usage()[1]
            except Exception:
                pass
        return time.time() + 3600


def parse_last_json(s):
    """The JSON result object agy prints (last line, or the last {...} in the output)."""
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s.splitlines()[-1])
    except json.JSONDecodeError:
        m = re.search(r"\{\"conversation_id\".*\}", s, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


def run_until_json(cmd, cwd, timeout, logfile):
    """Run agy, but don't wait for it to exit: it sometimes prints its complete JSON reply and then
    hangs. Stop it a few seconds after a complete result object has appeared on stdout."""
    import tempfile
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, text=True)
        t0, seen = time.time(), None
        while proc.poll() is None:
            time.sleep(2)
            out.seek(0)
            data = parse_last_json(out.read())
            if data.get("status"):
                seen = seen or time.time()
                if time.time() - seen > 5:          # complete reply, process lingering: stop it
                    proc.kill()
                    break
            if time.time() - t0 > timeout:
                proc.kill()
                proc.wait()
                out.seek(0); err.seek(0)
                Path(logfile).write_text(f"TIMEOUT after {timeout}s\n{out.read()}\n--- stderr ---\n{err.read()}")
                raise subprocess.TimeoutExpired(cmd, timeout)
        proc.wait()
        out.seek(0); err.seek(0)
        return out.read(), err.read()


def strip_fences(t):
    t = t.strip()
    m = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", t, re.S)
    return m.group(1).strip() if m else t


# ---------------------------------------------------------------- helpers
def book_dir(slug):
    d = BOOKS / slug
    if not d.exists():
        sys.exit(f"no such book: {d}")
    return d


def meta(d):
    return json.loads((d / "book.json").read_text())


def is_english(m):
    return (m.get("lang") or "").lower().split("-")[0] in ("en", "eng", "english")


def lang_hint(m):
    lang = m.get("lang")
    return f" written in {LANG_NAMES.get(lang, lang)}" if lang else ""


def sources(d):
    """pNNNN -> (source file name, side) from pages/sources.tsv (side is '' for whole images)."""
    tsv = d / "pages" / "sources.tsv"
    out = {}
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                out[int(parts[0][1:])] = (parts[1], parts[2] if len(parts) > 2 else "")
    return out


def half_note(d, n):
    side = sources(d).get(n, ("", ""))[1]
    if side not in ("left", "right"):
        return ""
    return HALF_NOTE.replace("{SIDE}", side).replace("{EDGE}", "left" if side == "right" else "right")


def page_images(d):
    return sorted((d / "pages").glob("p*.jpg")) + sorted((d / "pages").glob("p*.png"))


def pnum(path):
    return int(re.search(r"p(\d+)", Path(path).stem).group(1))


def is_cjk(ch):
    return bool(ch) and ("　" <= ch <= "鿿" or "가" <= ch <= "힯"
                         or "豈" <= ch <= "﫿" or "＀" <= ch <= "￯")


def free_gb():
    return min(shutil.disk_usage(Path.home()).free, shutil.disk_usage(ROOT).free) / 1e9


class DiskLow(Exception):
    pass


def run_parallel(items, fn, jobs):
    """Run fn over items; stop submitting new work once quota is exhausted everywhere."""
    stop = threading.Event()
    failures = []

    def wrap(it):
        if stop.is_set():
            return
        try:
            if free_gb() < MIN_FREE_GB:
                raise DiskLow(f"only {free_gb():.1f} GB free (minimum {MIN_FREE_GB} GB)")
            fn(it)
        except (QuotaExhausted, DiskLow) as e:
            if not stop.is_set():
                log(f"[stop] {e}. Rerun the same command later to resume.")
            stop.set()
        except Exception as e:
            failures.append((it, e))
            log(f"  !! {it}: {e}")

    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(wrap, items))
    if failures:
        log(f"{len(failures)} item(s) failed: " + ", ".join(str(Path(str(i)).name) for i, _ in failures[:20]))
    return not stop.is_set() and not failures


# ---------------------------------------------------------------- prepare
IMG_EXT = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def rel_source(src):
    """Source path relative to the project when it lives inside it, so no home path is committed."""
    try:
        return str(src.relative_to(ROOT))
    except ValueError:
        return str(src)


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def cmd_prepare(a):
    src = Path(a.source).resolve()
    d = BOOKS / a.slug
    (d / "pages").mkdir(parents=True, exist_ok=True)
    for sub in ("extract", "chunks", "translate", "logs", "out"):
        (d / sub).mkdir(exist_ok=True)
    m = json.loads((d / "book.json").read_text()) if (d / "book.json").exists() else {}
    m.update({k: v for k, v in dict(title=a.title, title_en=a.title_en, author=a.author, lang=a.lang,
                                    format_note=a.format_note, spreads=a.spreads, ia_identifier=a.ia,
                                    source=rel_source(src)).items() if v})
    m.setdefault("title", a.slug)
    (d / "book.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    free = shutil.disk_usage(d).free / 1e9
    if free < MIN_FREE_GB:
        sys.exit(f"only {free:.1f} GB free (minimum {MIN_FREE_GB} GB); clear space first")
    if src.is_dir():
        # a folder of photographed/scanned pages (HEIC etc.): convert each to a scaled JPEG, in name order
        imgs = sorted((f for f in src.iterdir() if f.suffix.lower() in IMG_EXT), key=natural_key)
        if a.first or a.last:
            imgs = imgs[(a.first or 1) - 1:a.last or None]
        log(f"converting {len(imgs)} images{' (splitting two-page spreads)' if a.spreads else ''} …")
        names, k = [], 1
        for f in imgs:
            if a.spreads:
                # one photo = two pages: cut at the middle with overlap, reading-order first
                sides = ["right", "left"] if a.spreads == "rtl" else ["left", "right"]
                for side in sides:
                    out = d / "pages" / f"p{k:04d}.jpg"
                    names.append(f"p{k:04d}\t{f.name}\t{side}")
                    k += 1
                    if out.exists():
                        continue
                    lo, hi = (0.5 - SPLIT_OVERLAP, 1.0) if side == "right" else (0.0, 0.5 + SPLIT_OVERLAP)
                    subprocess.run(["magick", str(f), "-auto-orient", "-crop",
                                    f"{(hi - lo) * 100:.1f}%x100%+%[fx:int(w*{lo})]+0", "+repage",
                                    "-resize", f"{a.scale}x{a.scale}>", "-quality", "88", str(out)],
                                   check=True, capture_output=True)
            else:
                out = d / "pages" / f"p{k:04d}.jpg"
                names.append(f"p{k:04d}\t{f.name}")
                k += 1
                if not out.exists():
                    subprocess.run(["magick", str(f), "-auto-orient", "-resize", f"{a.scale}x{a.scale}>",
                                    "-quality", "88", str(out)], check=True, capture_output=True)
        (d / "pages" / "sources.tsv").write_text("\n".join(names) + "\n")
    else:
        cmd = ["pdftoppm", "-jpeg", "-jpegopt", "quality=88", "-scale-to", str(a.scale)]
        if a.first:
            cmd += ["-f", str(a.first)]
        if a.last:
            cmd += ["-l", str(a.last)]
        tmp = d / "pages" / "_tmp"
        tmp.mkdir(exist_ok=True)
        log("rendering pages …")
        subprocess.run(cmd + [str(src), str(tmp / "x")], check=True)
        for f in tmp.glob("x-*.jpg"):
            n = int(f.stem.split("-")[-1])
            f.rename(d / "pages" / f"p{n:04d}.jpg")
        tmp.rmdir()
    log(f"{len(page_images(d))} page images in {d / 'pages'}")


# ---------------------------------------------------------------- extract
REFUSAL_RE = re.compile(r"(?i)safety filters?|blocked by gemini|(?:unable|not able|cannot|can't) (?:to )?(?:provide|transcribe|"
                        r"translate|complete) (?:a |the |this )?(?:full |complete )?(?:transcription|translation|"
                        r"this (?:page|passage|text|request))")


def extract_validate(t):
    if not t.lstrip().upper().startswith("PRINTED_PAGE"):
        return "missing PRINTED_PAGE line"
    if REFUSAL_RE.search(t):
        return "model refusal"
    return None


def cmd_extract(a, be=None):
    d = book_dir(a.slug)
    m = meta(d)
    be = be or Backends(a.backend, a.codex_model, a.codex_effort, a.agy_model)
    tpl = (PROMPTS / "extract.txt").read_text()
    todo = [p for p in page_images(d) if not (d / "extract" / f"p{pnum(p):04d}.md").exists()]
    if a.pages:
        lo, hi = (int(x) for x in a.pages.split("-"))
        todo = [p for p in todo if lo <= pnum(p) <= hi]
    log(f"extract: {len(todo)} page(s) to do")

    def one(img):
        n = pnum(img)
        prompt = (tpl.replace("{IMAGE}", str(img)).replace("{PAGE}", str(n)).replace("{LANG_HINT}", lang_hint(m))
                  .replace("{FORMAT_NOTE}", " ".join(filter(None, [m.get("format_note", ""), half_note(d, n)]))))
        t0 = time.time()
        name, text = be.run(prompt, d, d / "logs" / f"extract-p{n:04d}", "extract", images=[img],
                            validate=extract_validate)
        (d / "extract" / f"p{n:04d}.md").write_text(text + "\n")
        with open(d / "logs" / "provenance.tsv", "a") as f:
            f.write(f"extract\tp{n:04d}\t{name}\t{time.strftime('%Y-%m-%d %H:%M')}\n")
        log(f"  p{n:04d} ✓ {name} {time.time() - t0:.0f}s {len(text)} chars")

    return run_parallel(todo, one, a.jobs)


# ---------------------------------------------------------------- merge
def parse_page(raw):
    lines = raw.strip().splitlines()
    printed = None
    if lines and lines[0].upper().startswith("PRINTED_PAGE"):
        v = lines[0].split(":", 1)[1].strip() if ":" in lines[0] else ""
        v = v.strip(" -—–()[]（）〔〕")
        printed = None if v.lower() in ("", "none", "n/a") else v
        lines = lines[1:]
    text = "\n".join(lines).strip()
    cont = "<<CONTINUES>>" in text
    text = text.replace("<<CONTINUES>>", "").replace("<<BLANK>>", "").strip()
    body, _, notes = text.partition("---notes---")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body.strip()) if b.strip()]
    notes = [n.strip() for n in notes.strip().splitlines() if n.strip()]
    return printed, blocks, notes, cont


def join_across(prev, nxt, marker):
    """Join a paragraph split over a page break, placing the page marker at the seam."""
    if re.search(r"[A-Za-zÀ-ÿ]-$", prev) and nxt[:1].islower():      # hyphenated word
        word, rest = re.match(r"(\S*)(.*)", nxt, re.S).groups()
        return prev[:-1] + word + " " + marker + rest
    sep = "" if is_cjk(prev[-1:]) or is_cjk(nxt[:1]) else " "
    return prev + sep + marker + sep + nxt


def cmd_merge(a):
    d = book_dir(a.slug)
    files = sorted((d / "extract").glob("p*.md"))
    missing = [pnum(p) for p in page_images(d) if not (d / "extract" / f"p{pnum(p):04d}.md").exists()]
    if missing:
        log(f"warning: {len(missing)} page(s) not extracted yet: {missing[:15]}")
    blocks, printed_map, pending, open_para = [], {}, [], False
    for f in files:
        n = pnum(f)
        printed, pblocks, notes, next_cont = parse_page(f.read_text())
        marker = f"[[p{n}]]"
        printed_map[str(n)] = printed
        joined = (open_para and pblocks and not pblocks[0].startswith("#")
                  and not blocks[-1].startswith(("#", ":::")) and not MARK_RE.fullmatch(blocks[-1]))
        if joined:
            blocks[-1] = join_across(blocks[-1], pblocks.pop(0), marker)
            if pblocks or not next_cont:        # the split paragraph is now complete
                if pending:
                    blocks.append(notes_block(pending))
                pending = []
        else:
            if pending:
                blocks.append(notes_block(pending))
            pending = []
            blocks.append(marker)
        blocks.extend(pblocks)
        # notes are held back while the page's last paragraph runs on to the next page
        open_para = next_cont and (bool(pblocks) or joined)
        if open_para:
            pending += notes
        elif notes:
            blocks.append(notes_block(notes))
    if pending:
        blocks.append(notes_block(pending))
    blocks = apply_heading_rules(blocks, meta(d))
    (d / "merged.md").write_text("\n\n".join(blocks) + "\n")
    (d / "pages.json").write_text(json.dumps(printed_map, indent=0))
    log(f"merged {len(files)} pages -> {d / 'merged.md'} ({sum(map(len, blocks))} chars)")


def apply_heading_rules(blocks, m):
    """Per-book heading clean-up from book.json:
    heading_rules: [[regex, level, "join"?], ...] matched against the heading text (emphasis stripped);
      "join" merges the next heading block into this one ("II. Kapitel." + "Zwing-Ost.").
    heading_default: level for headings no rule matches (default: keep).
    plain_pages: image numbers (contents, lists) whose headings become plain lines, one per line,
      except headings matching plain_keep (e.g. "^Inhalt"), which stay level 1."""
    rules = [(re.compile(r[0]), r[1], len(r) > 2 and r[2] == "join") for r in m.get("heading_rules", [])]
    plain = set(m.get("plain_pages", []))
    keep = re.compile(m.get("plain_keep", "(?!x)x"))
    if not rules and not plain:
        return blocks
    is_h = lambda b: b.startswith("#") and "\n" not in b.strip()
    text_of = lambda b: re.sub(r"^#+\s*", "", b).replace("*", "").strip()
    # a block of several heading lines ("# II. Kapitel.\n# Zwing-Ost.") -> one block per heading
    # and a heading fused to the paragraph below it -> heading block + paragraph block
    split = []
    for b in blocks:
        lines = b.strip().splitlines()
        while len(lines) > 1 and lines[0].startswith("#"):
            split.append(lines.pop(0))
        split.append("\n".join(lines))
    blocks = split
    out, page, i = [], 0, 0
    while i < len(blocks):
        b = blocks[i]
        for mk in MARK_RE.findall(b):
            page = int(mk)
        if page in plain and not MARK_RE.fullmatch(b.strip()):
            lines = [text_of(l) if l.startswith("#") else l for l in b.splitlines()]
            first = lines[0]
            if is_h(b) and keep.search(first):
                out.append("# " + first)
            else:
                out.append(" \\\n".join(l.rstrip() for l in lines if l.strip()))
            i += 1
            continue
        if is_h(b):
            t = text_of(b)
            rule = next((r for r in rules if r[0].search(t)), None)
            level = rule[1] if rule else m.get("heading_default", len(b) - len(b.lstrip("#")))
            nxt = blocks[i + 1] if i + 1 < len(blocks) else ""
            # join the next heading, or a short subtitle line printed as text ("bis zur / Schlacht von Mukden.")
            short = (nxt and not is_h(nxt) and len(nxt) <= 80 and not MARK_RE.search(nxt)
                     and not nxt.startswith(("[", ":::")))
            if rule and rule[2] and ((is_h(nxt) and not any(r[0].search(text_of(nxt)) for r in rules)) or short):
                sub = " ".join(text_of(nxt).split())
                t = t.rstrip(".") + (" " if short else ". ") + sub
                i += 1
            out.append("#" * level + " " + t)
        else:
            out.append(b)
        i += 1
    return out


def notes_block(notes):
    return "::: notes\n" + "\n\n".join(notes) + "\n:::"


# ---------------------------------------------------------------- chunk
def split_blocks(text):
    """Blank-line separated blocks, keeping ::: fenced divs whole."""
    out, cur, fence = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(":::"):
            fence = s != ":::"          # "::: notes" opens, bare ":::" closes
        if not s and not fence:
            if cur:
                out.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out


def cmd_chunk(a):
    d = book_dir(a.slug)
    text = (d / "merged.md").read_text()
    blocks = split_blocks(text)
    sample = MARK_RE.sub("", text)[:20000]
    cjk = sum(map(is_cjk, sample)) / max(1, len(sample))
    target = a.chars or (1800 if cjk > 0.3 else 5000)
    # glue standalone page markers to the following block so a chunk never ends on one
    units, carry = [], []
    for b in blocks:
        if MARK_RE.fullmatch(b.strip()):
            carry.append(b)
            continue
        units.append("\n\n".join(carry + [b]))
        carry = []
    if carry:
        units.append("\n\n".join(carry))
    chunks, cur, size = [], [], 0
    for u in units:
        ulen = len(MARK_RE.sub("", u))
        body = MARK_RE.sub("", u).lstrip()
        heading = body.startswith("#")
        # break before a heading once the chunk is reasonably full, or when it would overflow
        if cur and ((heading and size >= target * 0.5) or size + ulen > target * 1.3 or size >= target):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(u)
        size += ulen
    if cur:
        chunks.append(cur)
    cdir = d / "chunks"
    for old in cdir.glob("c*.md"):
        old.unlink()
    for i, c in enumerate(chunks, 1):
        (cdir / f"c{i:04d}.md").write_text("\n\n".join(c) + "\n")
    log(f"{len(chunks)} chunks (target {target} chars, CJK ratio {cjk:.2f}) in {cdir}")


# ---------------------------------------------------------------- translate
def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def make_translate_validator(src):
    want = MARK_RE.findall(src)
    want_notes = src.count("::: notes")

    def v(t):
        if REFUSAL_RE.search(t):
            return "model refusal"
        got = MARK_RE.findall(t)
        if got != want:
            return f"page markers {got} != {want}"
        if t.count("::: notes") != want_notes:
            return "notes fences changed"
        if len(MARK_RE.sub("", t).strip()) < 0.3 * len(MARK_RE.sub("", src).strip()):
            return "translation suspiciously short"
        return None
    return v


def context(text, n, tail):
    paras = [b for b in split_blocks(text) if not MARK_RE.fullmatch(b.strip())]
    paras = paras[-n:] if tail else paras[:n]
    return MARK_RE.sub("", "\n\n".join(paras)).strip() or "(none)"


def cmd_translate(a, be=None):
    d = book_dir(a.slug)
    m = meta(d)
    if is_english(m):
        log("translate: source is English; nothing to translate")
        return True
    be = be or Backends(a.backend, a.codex_model, a.codex_effort, a.agy_model)
    tpl = (PROMPTS / "translate.txt").read_text()
    gl = d / "glossary.md"
    glossary = ("\nGlossary (use these renderings):\n" + gl.read_text().strip() + "\n") if gl.exists() else ""
    chunks = sorted((d / "chunks").glob("c*.md"))
    if not chunks:
        sys.exit("no chunks; run merge and chunk first")
    texts = [c.read_text() for c in chunks]

    def done(i):
        out = d / "translate" / chunks[i].name
        shaf = out.with_suffix(".sha")
        return out.exists() and shaf.exists() and shaf.read_text().strip() == sha(texts[i])

    todo = [i for i in range(len(chunks)) if not done(i)]
    log(f"translate: {len(todo)} of {len(chunks)} chunk(s) to do")

    def one(i):
        src = texts[i]
        prompt = (tpl.replace("{LANG_HINT}", f" from {LANG_NAMES.get(m['lang'], m['lang'])}" if m.get("lang") else "")
                  .replace("{GLOSSARY}", glossary)
                  .replace("{CONTEXT_BEFORE}", context(texts[i - 1], 2, True) if i else "(start of book)")
                  .replace("{CONTEXT_AFTER}", context(texts[i + 1], 1, False) if i + 1 < len(texts) else "(end of book)")
                  .replace("{TEXT}", src.strip()))
        t0 = time.time()
        name, text = be.run(prompt, d, d / "logs" / f"translate-{chunks[i].stem}", "translate",
                            validate=make_translate_validator(src))
        out = d / "translate" / chunks[i].name
        out.write_text(text + "\n")
        out.with_suffix(".sha").write_text(sha(src))
        with open(d / "logs" / "provenance.tsv", "a") as f:
            f.write(f"translate\t{chunks[i].stem}\t{name}\t{time.strftime('%Y-%m-%d %H:%M')}\n")
        log(f"  {chunks[i].stem} ✓ {name} {time.time() - t0:.0f}s")

    return run_parallel(todo, one, a.jobs)


# ---------------------------------------------------------------- build
def page_label(key, pmap):
    p = pmap.get(str(key))
    return p if p else f"[{key}]"


def markers_md(text, pmap):
    """[[pN]] -> plain-text reference for the final Markdown files."""
    return MARK_RE.sub(lambda mo: f"{{p. {page_label(int(mo.group(1)), pmap)}}}", text)


def scan_links(d, m):
    """Page number -> URL of the ORIGINAL scan, so links survive deleting pages/*.jpg."""
    links = {}
    if m.get("ia_identifier"):          # public scan: link each page to archive.org
        for k in json.loads((d / "pages.json").read_text()) if (d / "pages.json").exists() else []:
            n = int(k)
            links[n] = f"https://archive.org/details/{m['ia_identifier']}/page/n{n - 1}/mode/1up"
        return links
    if m.get("scan_url") or m.get("scan_urls"):    # other public scan (e.g. a library viewer)
        per = m.get("scan_urls", {})
        for k in json.loads((d / "pages.json").read_text()) if (d / "pages.json").exists() else []:
            links[int(k)] = per.get(str(k), m.get("scan_url"))
        return links
    src = Path(m.get("source") or m.get("source_pdf") or "")
    if not src.is_absolute():
        src = ROOT / src
    tsv = d / "pages" / "sources.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            if "\t" in line:
                k, name = line.split("\t")[:2]
                links[int(k[1:])] = (src / name).as_uri()
    elif src.suffix.lower() == ".pdf":
        for img in page_images(d):
            links[pnum(img)] = src.as_uri() + f"#page={pnum(img)}"
    return links


def markers_html(text, pmap, links, epub=False, idp="pg"):
    def inline(mo):
        n = int(mo.group(1))
        lab = html.escape(page_label(n, pmap))
        if epub:
            return f'<span epub:type="pagebreak" role="doc-pagebreak" id="pg{n}" title="{lab}" class="pb">{lab}</span>'
        return (f'<a class="pb" id="{idp}{n}" href="{html.escape(links.get(n, ''))}" target="_blank" '
                f'title="Scan {n}" data-page="{lab}">{lab}</a>')
    out = []
    for b in split_blocks(text):
        if MARK_RE.fullmatch(b.strip()):
            out.append(f'<div class="pbl">{inline(MARK_RE.fullmatch(b.strip()))}</div>')
        else:
            out.append(MARK_RE.sub(inline, b))
    return "\n\n".join(out)


def pandoc_html(md):
    r = subprocess.run(["pandoc", "-f", "markdown-implicit_figures-tex_math_dollars",
                        "-t", "html5", "--wrap=none"], input=md, capture_output=True, text=True, check=True)
    return r.stdout


def cmd_build(a):
    d = book_dir(a.slug)
    m = meta(d)
    pmap = json.loads((d / "pages.json").read_text()) if (d / "pages.json").exists() else {}
    links = scan_links(d, m)
    out = d / "out"
    out.mkdir(exist_ok=True)
    chunks = sorted((d / "chunks").glob("c*.md"))
    orig = [c.read_text().strip() for c in chunks]
    mono = is_english(m)                  # English original: transcription only, no translation
    en = list(orig) if mono else []
    for c, src in zip(chunks, orig) if not mono else ():
        t = d / "translate" / c.name
        en.append(t.read_text().strip() if t.exists() and t.with_suffix(".sha").exists()
                  and t.with_suffix(".sha").read_text().strip() == sha(c.read_text()) else None)
    missing = [c.stem for c, e in zip(chunks, en) if e is None]
    if missing:
        log(f"warning: {len(missing)} chunk(s) untranslated: {missing[:10]} — shown as original only")

    title, author = m.get("title", a.slug), m.get("author", "")
    cite = f"Source: {m['citation']}\n\n" if m.get("citation") else ""
    head = f"# {title}\n\n" + (f"*{author}*\n\n" if author else "") + cite
    note = "Page references in braces, e.g. {p. 23}, mark where each printed page begins.\n\n"
    (out / f"{a.slug}{'' if mono else '.original'}.md").write_text(
        head + note + markers_md("\n\n".join(orig), pmap) + "\n")
    en_title = m.get("title_en", title)
    if not mono:
        en_md = "\n\n".join(e if e else "> [untranslated]\n\n" + o for e, o in zip(en, orig))
        (out / f"{a.slug}.en.md").write_text(f"# {en_title}\n\n" + (f"*{author}*\n\n" if author else "")
                                             + cite + note + markers_md(en_md, pmap) + "\n")

    # ---- HTML: convert all chunks in one pandoc call, each wrapped in a fenced div
    def conv(texts, cls):
        md = "\n\n".join(f"::: {{.ck #{cls}{i}}}\n{markers_html(t or '', pmap, links, idp=cls + 'pg')}\n:::"
                         for i, t in enumerate(texts, 1))
        h = pandoc_html(md)
        parts = re.split(rf'<div (?=[^>]*class="ck")[^>]*id="{cls}\d+"[^>]*>', h)[1:]
        return [re.sub(r"</div>\s*$", "", p.strip()) for p in parts]

    e_html = conv([e or "" for e in en], "e")
    o_html = [""] * len(e_html) if mono else conv(orig, "o")
    rows = []
    for i, (oh, eh) in enumerate(zip(o_html, e_html), 1):
        cls = "row" + ("" if en[i - 1] else " untr")
        orig_div = "" if mono else f'<div class="orig" lang="{html.escape(m.get("lang", ""))}">{oh}</div>'
        rows.append(f'<section class="{cls}" id="c{i}"><div class="en" lang="en">'
                    f'{eh or "<p class=untr-note>[not yet translated]</p>"}</div>{orig_div}</section>')
    npages = len(pmap) or len(page_images(d))
    tpl = (ROOT / "template.html").read_text()
    page = (tpl.replace("{{TITLE}}", html.escape(en_title))
            .replace("{{TITLE_ORIG}}", html.escape(title) if title != en_title else "")
            .replace("{{AUTHOR}}", html.escape(author))
            .replace("{{LANG}}", html.escape(m.get("lang", "")))
            .replace("{{NOTE}}", cite_html({"citation": m.get("note", "")}))
            .replace("{{CITATION}}", cite_html(m))
            .replace("{{SLUG}}", html.escape(a.slug))
            .replace("{{BODYCLASS}}", "mono" if mono else "")
            .replace("{{STATS}}", f"{npages} page{"s" if npages != 1 else ""} · transcribed by machine; check against the scans "
                                  "(click a page number)" if mono else
                     f"{npages} page{"s" if npages != 1 else ""} · {len(chunks)} sections · {len(chunks) - len(missing)} "
                     "translated · transcribed and translated by machine; check against the scans "
                     "(click a page number)")
            .replace("{{BODY}}", "\n".join(rows)))
    (out / "index.html").write_text(page)

    # ---- EPUB (English, page-list via epub:type=pagebreak); also an original-language EPUB
    css = ROOT / "epub.css"
    editions = [("en", [e or o for e, o in zip(en, orig)], "en", en_title)]
    if not mono:
        editions.append(("original", orig, m.get("lang", "und"), title))
    for kind, texts, lang, ttl in editions:
        md = "\n\n".join(markers_html(t, pmap, {}, epub=True) for t in texts)
        metaf = out / f".{kind}.yaml"
        md_meta = {"title": ttl, "lang": lang, "date": time.strftime("%Y-%m-%d")}
        if author:
            md_meta["creator"] = author
        if m.get("citation"):
            md_meta["description"] = m["citation"].replace("*", "")
            md = f"*Source:* {m['citation']}\n\n" + md
        metaf.write_text(json.dumps(md_meta, ensure_ascii=False))
        target = out / f"{a.slug}{'' if kind == 'en' else '.original'}.epub"
        has_h1 = re.search(r"(?m)^# ", md)
        subprocess.run(["pandoc", "-f", "markdown-implicit_figures-tex_math_dollars",
                        "-t", "epub3", "--toc", "--toc-depth=2", "--split-level=2",
                        "--css", str(css), "--metadata-file", str(metaf), "-o", str(target)],
                       input=md, text=True, check=True)
        metaf.unlink()
    log(f"built {out}: " + ", ".join(sorted(f.name for f in out.iterdir() if not f.name.startswith("."))))
    cmd_site(a)


def cmd_clean(a):
    """Delete the working page images of a finished book. Never touches the original source."""
    d = book_dir(a.slug)
    if not (d / "out" / "index.html").exists():
        sys.exit("not built yet; refusing to clean")
    m = meta(d)
    src = Path(m.get("source") or m.get("source_pdf") or "/nonexistent")
    src = (src if src.is_absolute() else ROOT / src).resolve()
    pages = (d / "pages").resolve()
    assert pages != src and src not in pages.parents and pages not in src.parents
    imgs = list(pages.glob("p*.jpg"))
    size = sum(f.stat().st_size for f in imgs) / 1e6
    for f in imgs:
        f.unlink()
    log(f"removed {len(imgs)} working page images ({size:.0f} MB) from {pages}; original left at {src}")


# ---------------------------------------------------------------- status / run
SITE_CSS = """
:root{--bg:#fbf8f2;--fg:#24211c;--muted:#6f685c;--rule:#e6dfd2;--accent:#8a3b22;--panel:#f3eee4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1916;--fg:#e4dfd4;
  --muted:#a29a8b;--rule:#34312b;--accent:#e0936f;--panel:#23211d}}
:root[data-theme="dark"]{--bg:#1a1916;--fg:#e4dfd4;--muted:#a29a8b;--rule:#34312b;--accent:#e0936f;--panel:#23211d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:18px/1.6 "Iowan Old Style",Palatino,Georgia,serif}
main{max-width:46rem;margin:0 auto;padding:3.5rem 16px 5rem}
a{color:var(--accent)}
h1{font-size:2rem;margin:0 0 .3rem;font-weight:600}
.lede{color:var(--muted);margin:0 0 2.5rem;font-size:1rem}
.book{border-top:1px solid var(--rule);padding:1.4rem 0}
.book h2{font-size:1.3rem;margin:0;font-weight:600;line-height:1.3}
.book h2 a{color:var(--fg);text-decoration:none}
.book h2 a:hover{color:var(--accent)}
.orig{color:var(--muted);font-size:1rem;margin-top:.15rem}
.cite{color:var(--muted);font-size:.92rem;margin-top:.4rem}
.cite a{color:inherit}
.meta{color:var(--muted);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin-top:.5rem}
.links{margin-top:.6rem;display:flex;flex-wrap:wrap;gap:.4rem;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.links a{color:var(--fg);background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  padding:.25rem .6rem;text-decoration:none}
.links a.primary{background:var(--accent);border-color:var(--accent);color:var(--bg)}
footer{margin-top:3rem;color:var(--muted);font:13px -apple-system,BlinkMacSystemFont,sans-serif}
"""


def cite_html(m):
    """book.json "citation" (Markdown *italics* allowed; URLs become links) -> HTML."""
    c = html.escape(m.get("citation", ""))
    c = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", c)
    c = re.sub(r"(https?://[^\s<]+?)([.,;)]?(?:\s|$))", r'<a href="\1">\1</a>\2', c)
    return c


def cmd_site(a):
    """Root index.html listing every built book (for GitHub Pages)."""
    items = []
    for d in sorted(BOOKS.iterdir()):
        out = d / "out"
        if not (out / "index.html").exists():
            continue
        m = meta(d)
        slug = d.name
        e = html.escape
        title_en = m.get("title_en") or m.get("title", slug)
        pages = len(json.loads((d / "pages.json").read_text())) if (d / "pages.json").exists() else 0
        lang = LANG_NAMES.get(m.get("lang"), m.get("lang", ""))
        files = [("Read", "index.html", "primary")]
        for name, label in ((f"{slug}.epub", "EPUB (English)"), (f"{slug}.en.md", "English text"),
                            (f"{slug}.original.epub", "EPUB (original)"), (f"{slug}.original.md", "Original text"),
                            (f"{slug}.md", "Transcription")):
            if (out / name).exists():
                files.append((label, name, ""))
        links = "".join(f'<a class="{c}" href="books/{slug}/out/{f}">{e(l)}</a>' for l, f, c in files)
        src = (f' · scan: <a href="https://archive.org/details/{e(m["ia_identifier"])}">Internet Archive</a>'
               if m.get("ia_identifier") else "")
        if not src and (m.get("scan_url") or m.get("scan_urls")):
            u = m.get("scan_url") or next(iter(m["scan_urls"].values()))
            src = f' · scan: <a href="{e(u)}">{e(m.get("scan_label", "original"))}</a>'
        cite = f'<div class="cite">{cite_html(m)}</div>' if m.get("citation") else ""
        orig = f'<div class="orig">{e(m["title"])}</div>' if m.get("title") and m.get("title") != title_en else ""
        items.append(f'<article class="book"><h2><a href="books/{slug}/out/index.html">{e(title_en)}</a></h2>{orig}'
                     f'<div class="meta">{e(m.get("author", ""))}{" · " if m.get("author") else ""}'
                     f'{e(lang)} · {pages} page{"s" if pages != 1 else ""}{src}</div>{cite}<div class="links">{links}</div></article>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Translated Scans</title><style>{SITE_CSS}</style></head>
<body><main>
<h1>Translated Scans</h1>
<p class="lede">Books transcribed from page scans and translated into English, with the original text alongside.
Transcription and translation are by machine (Gemini via Antigravity), so check against the scans:
every page number in the reader links to its scan.</p>
{"".join(items) or "<p>No books yet.</p>"}
<footer>Built with <a href="https://github.com/kmlawson/translations">pipeline.py</a>.</footer>
</main></body></html>
"""
    (ROOT / "index.html").write_text(page)
    log(f"site index: {len(items)} book(s) -> {ROOT / 'index.html'}")


def cmd_status(a):
    d = book_dir(a.slug)
    imgs = page_images(d)
    ex = len(list((d / "extract").glob("p*.md")))
    ch = sorted((d / "chunks").glob("c*.md"))
    tr = sum(1 for c in ch if (d / "translate" / c.name).exists() and (d / "translate" / f"{c.stem}.sha").exists()
             and (d / "translate" / f"{c.stem}.sha").read_text().strip() == sha(c.read_text()))
    print(f"{a.slug}: pages {len(imgs)} · extracted {ex} · chunks {len(ch)} · translated {tr}")
    be = Backends()
    for n in ("agy", "codex"):
        if be._blocked(n):
            print(f"  {n} blocked until {dt.datetime.fromtimestamp(be.state[n + '_blocked_until']):%Y-%m-%d %H:%M}")
    try:
        f, r = be.agy_usage()
        print(f"  agy Gemini quota remaining {f:.0%} (resets {dt.datetime.fromtimestamp(r):%m-%d %H:%M})")
    except Exception as e:
        print(f"  agy usage unavailable: {e}")


def cmd_run(a):
    be = Backends(a.backend, a.codex_model, a.codex_effort, a.agy_model)
    if not cmd_extract(a, be):
        sys.exit("extraction incomplete; rerun to resume")
    cmd_merge(a)
    cmd_chunk(a)
    if not cmd_translate(a, be):
        sys.exit("translation incomplete; rerun to resume")
    cmd_build(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, workers=False):
        p.add_argument("slug")
        if workers:
            p.add_argument("-j", "--jobs", type=int, default=3)
            p.add_argument("--backend", choices=["auto", "agy", "codex"], default="auto")
            p.add_argument("--agy-model", help="override the per-task agy model")
            p.add_argument("--codex-model", help="override the per-task codex model")
            p.add_argument("--codex-effort", choices=["low", "medium", "high", "xhigh"])
        return p

    p = sp.add_parser("prepare")
    p.add_argument("source", help="a PDF, or a folder of page images (HEIC/JPG/PNG/TIFF)")
    p.add_argument("slug")
    p.add_argument("--title")
    p.add_argument("--title-en")
    p.add_argument("--ia", help="Internet Archive identifier: page numbers then link to archive.org")
    p.add_argument("--format-note", help='sentence for the extraction prompt, e.g. "The original is in '
                   'Japanese. Preserve old-form characters."')
    p.add_argument("--author")
    p.add_argument("--spreads", choices=["rtl", "ltr"],
                   help="each image is a photo of a two-page opening: split it into pages (rtl = right page first)")
    p.add_argument("--lang", help="source language code: ja, zh, zh-Hant, ko, fr …")
    p.add_argument("--first", type=int)
    p.add_argument("--last", type=int)
    p.add_argument("--scale", type=int, default=3000, help="max longest edge in px (default 3000)")
    p.set_defaults(fn=cmd_prepare)
    p = common(sp.add_parser("extract"), True)
    p.add_argument("--pages", help="only this PDF page range, e.g. 1-20")
    p.set_defaults(fn=cmd_extract)
    common(sp.add_parser("merge")).set_defaults(fn=cmd_merge)
    p = common(sp.add_parser("chunk"))
    p.add_argument("--chars", type=int, help="target chunk size in source characters")
    p.set_defaults(fn=cmd_chunk)
    common(sp.add_parser("translate"), True).set_defaults(fn=cmd_translate)
    common(sp.add_parser("build")).set_defaults(fn=cmd_build)
    common(sp.add_parser("status")).set_defaults(fn=cmd_status)
    common(sp.add_parser("clean")).set_defaults(fn=cmd_clean)
    sp.add_parser("site", help="write the root index.html listing all built books").set_defaults(fn=cmd_site)
    p = common(sp.add_parser("run"), True)
    p.add_argument("--pages")
    p.add_argument("--chars", type=int)
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
