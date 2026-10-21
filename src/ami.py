"""Resolve AMI's NITE stand-off annotation into offset-addressable documents.

The chain we care about:

    summary sentence  --summlink-->  dialogue act  --child href-->  word range
                                                                        |
                                                                        v
                                                      exact text + character offsets

Everything a node claims must resolve to text[start:end] in a document we hold.
Verified by assertion, never by trust.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

NITE = "{http://nite.sourceforge.net/}"

# Where the unzipped AMI annotations live. Override with the AMI_ROOT
# environment variable, or pass root_dir= to load_meeting().
ROOT = os.environ.get("AMI_ROOT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ami"))

# Gold labels live OUTSIDE the corpus tree, on purpose. No AMI_ROOT path can
# reach them, so no amount of browsing data/ -- by a person or an agent --
# stumbles into the answers. Only the scorer may read this.
GOLD_PATH = os.environ.get(
    "GOLD_PATH", os.path.join(os.path.expanduser("~"), "gold", "meetings.xml"))

HREF_RE = re.compile(r"^([^#]+)#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?$")


def nid(el) -> str:
    return el.attrib.get(f"{NITE}id", "")


def parse_href(href: str):
    """'ES2002a.A.words.xml#id(w0)..id(w12)' -> (file, first_id, last_id)."""
    m = HREF_RE.match(href.strip())
    if not m:
        return None
    f, a, b = m.groups()
    return f, a, b or a


# --------------------------------------------------------------------------- #
@dataclass
class Word:
    wid: str
    text: str
    start: float | None
    end: float | None
    punc: bool
    kind: str            # 'w' | 'vocalsound' | 'disfmarker' | ...
    off_start: int = -1  # character offsets into the rendered document
    off_end: int = -1


@dataclass
class Document:
    doc_id: str          # 'ES2002a.A'
    text: str
    words: list[Word]
    by_id: dict[str, Word] = field(default_factory=dict)
    order: dict[str, int] = field(default_factory=dict)

    def span_for(self, first_id: str, last_id: str):
        """Character span covering a word range, or None if unresolvable."""
        a, b = self.order.get(first_id), self.order.get(last_id)
        if a is None or b is None:
            return None
        if b < a:
            a, b = b, a
        return self.words[a].off_start, self.words[b].off_end

    def quote(self, span) -> str:
        return self.text[span[0]:span[1]]


def load_words(path: str, doc_id: str) -> Document:
    """Render a speaker's words into one string, recording each word's offset.

    Detokenisation matters: offsets must point into the SAME string we later
    quote from, so we build text and offsets in a single pass.
    """
    root = ET.parse(path).getroot()
    words, chunks, pos = [], [], 0
    for el in root:
        tag = el.tag.split("}")[-1]
        txt = (el.text or "").strip()
        if tag == "w":
            if not txt:
                continue
        elif tag in ("vocalsound", "disfmarker", "gap", "pause", "nonvocalsound"):
            t = el.attrib.get("type", tag)
            txt = f"<{t}>"
        else:
            if not txt:
                continue

        is_punc = el.attrib.get("punc") == "true"
        sep = "" if (is_punc or pos == 0) else " "
        if sep:
            chunks.append(sep)
            pos += len(sep)
        w = Word(
            wid=nid(el), text=txt,
            start=float(el.attrib["starttime"]) if "starttime" in el.attrib else None,
            end=float(el.attrib["endtime"]) if "endtime" in el.attrib else None,
            punc=is_punc, kind=tag, off_start=pos, off_end=pos + len(txt),
        )
        chunks.append(txt)
        pos += len(txt)
        words.append(w)

    text = "".join(chunks)
    doc = Document(doc_id=doc_id, text=text, words=words)
    doc.by_id = {w.wid: w for w in words}
    doc.order = {w.wid: i for i, w in enumerate(words)}
    # contract: every word must quote back exactly
    for w in words:
        assert text[w.off_start:w.off_end] == w.text, (doc_id, w.wid)
    return doc


# --------------------------------------------------------------------------- #
@dataclass
class Utterance:
    seg_id: str
    doc_id: str
    speaker: str
    start: float | None
    end: float | None
    span: tuple[int, int] | None
    text: str


def load_segments(path: str, doc: Document, speaker: str) -> list[Utterance]:
    root = ET.parse(path).getroot()
    out = []
    for seg in root:
        span, ids = None, []
        for ch in seg:
            p = parse_href(ch.attrib.get("href", ""))
            if p:
                ids.append((p[1], p[2]))
        if ids:
            spans = [doc.span_for(a, b) for a, b in ids]
            spans = [s for s in spans if s]
            if spans:
                span = (min(s[0] for s in spans), max(s[1] for s in spans))
        out.append(Utterance(
            seg_id=nid(seg), doc_id=doc.doc_id, speaker=speaker,
            start=float(seg.attrib["transcriber_start"]) if "transcriber_start" in seg.attrib else None,
            end=float(seg.attrib["transcriber_end"]) if "transcriber_end" in seg.attrib else None,
            span=span, text=doc.quote(span) if span else "",
        ))
    return out


# --------------------------------------------------------------------------- #
@dataclass
class DialogueAct:
    act_id: str
    doc_id: str
    speaker: str
    da_type: str
    span: tuple[int, int] | None
    text: str


def load_da_types(root_dir: str) -> dict[str, str]:
    p = os.path.join(root_dir, "ontologies", "da-types.xml")
    if not os.path.exists(p):
        return {}
    out = {}

    def walk(e):
        i = e.attrib.get(f"{NITE}id") or e.attrib.get("id")
        g = e.attrib.get("gloss") or e.attrib.get("name")
        if i and g:
            out[i] = g
        for c in e:
            walk(c)

    walk(ET.parse(p).getroot())
    return out


def load_dialogue_acts(path: str, doc: Document, speaker: str,
                       da_types: dict[str, str]) -> list[DialogueAct]:
    root = ET.parse(path).getroot()
    out = []
    for d in root:
        dtype, span = "?", None
        ids = []
        for c in d:
            tag = c.tag.split("}")[-1]
            href = c.attrib.get("href", "")
            if tag == "pointer":
                m = re.search(r"#id\(([^)]+)\)", href)
                if m:
                    dtype = da_types.get(m.group(1), m.group(1))
            elif tag == "child":
                p = parse_href(href)
                if p:
                    ids.append((p[1], p[2]))
        if ids:
            spans = [doc.span_for(a, b) for a, b in ids]
            spans = [s for s in spans if s]
            if spans:
                span = (min(s[0] for s in spans), max(s[1] for s in spans))
        out.append(DialogueAct(
            act_id=nid(d), doc_id=doc.doc_id, speaker=speaker, da_type=dtype,
            span=span, text=doc.quote(span) if span else "",
        ))
    return out


# --------------------------------------------------------------------------- #
@dataclass
class Meeting:
    meeting_id: str
    roles: dict[str, str]                    # always {} -- see load_roles()
    docs: dict[str, Document]
    utterances: list[Utterance]
    acts: dict[str, DialogueAct]
    summary: dict[str, list[tuple[str, str]]]   # section -> [(sentence_id, text)]
    sentences: dict[str, str]
    summlinks: list[tuple[str, str]]            # (sentence_id, act_id)


def load_roles(meeting_id: str, gold_path: str = GOLD_PATH) -> dict[str, str]:
    """GOLD LABELS. The scorer may call this. Nothing else may.

    This used to be called from load_meeting(), which quietly put the answer
    key on every Meeting object the extractors were handed. Nothing read it,
    so the published numbers stand -- but the separation was a comment rather
    than a property, and one careless line would have ended it. It now reads
    from GOLD_PATH, outside the corpus, and no loader calls it.
    """
    if not os.path.exists(gold_path):
        raise FileNotFoundError(
            f"gold labels not found at {gold_path}. They are deliberately kept "
            f"outside the corpus; set GOLD_PATH or place meetings.xml there.")
    for m in ET.parse(gold_path).getroot():
        if m.attrib.get("observation") == meeting_id:
            return {s.attrib["nxt_agent"]: s.attrib["role"] for s in m
                    if "nxt_agent" in s.attrib}
    return {}


def load_summary(root_dir: str, meeting_id: str):
    p = os.path.join(root_dir, "abstractive", f"{meeting_id}.abssumm.xml")
    sections, sentences = {}, {}
    if not os.path.exists(p):
        return sections, sentences
    for sec in ET.parse(p).getroot():
        name = sec.tag.split("}")[-1]
        rows = []
        for s in sec:
            sid, txt = nid(s), (s.text or "").strip()
            if txt and txt != "*NA*":
                rows.append((sid, txt))
                sentences[sid] = txt
        sections[name] = rows
    return sections, sentences


def load_summlinks(root_dir: str, meeting_id: str) -> list[tuple[str, str]]:
    p = os.path.join(root_dir, "extractive", f"{meeting_id}.summlink.xml")
    if not os.path.exists(p):
        return []
    out = []
    for link in ET.parse(p).getroot():
        ext = abst = None
        for ptr in link:
            role, href = ptr.attrib.get("role"), ptr.attrib.get("href", "")
            m = re.search(r"#id\(([^)]+)\)", href)
            if not m:
                continue
            if role == "extractive":
                ext = m.group(1)
            elif role == "abstractive":
                abst = m.group(1)
        if ext and abst:
            out.append((abst, ext))
    return out


def load_meeting(meeting_id: str, root_dir: str = ROOT) -> Meeting:
    da_types = load_da_types(root_dir)
    docs, utts, acts = {}, [], {}
    for spk in "ABCDE":
        wp = os.path.join(root_dir, "words", f"{meeting_id}.{spk}.words.xml")
        if not os.path.exists(wp):
            continue
        doc = load_words(wp, f"{meeting_id}.{spk}")
        docs[spk] = doc
        sp = os.path.join(root_dir, "segments", f"{meeting_id}.{spk}.segments.xml")
        if os.path.exists(sp):
            utts.extend(load_segments(sp, doc, spk))
        dp = os.path.join(root_dir, "dialogueActs", f"{meeting_id}.{spk}.dialog-act.xml")
        if os.path.exists(dp):
            for a in load_dialogue_acts(dp, doc, spk, da_types):
                acts[a.act_id] = a
    utts.sort(key=lambda u: (u.start if u.start is not None else 1e9))
    sections, sentences = load_summary(root_dir, meeting_id)
    return Meeting(
        # roles stays empty. A Meeting is evidence; the role mapping is the
        # answer key, and the extraction path never holds both.
        meeting_id=meeting_id, roles={}, docs=docs,
        utterances=utts, acts=acts, summary=sections, sentences=sentences,
        summlinks=load_summlinks(root_dir, meeting_id),
    )
