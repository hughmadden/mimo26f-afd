"""spike/needle_prompt.py — deterministic 4K needle-prompt builder + runner.

I3-I1b: one planted fact near token ~100, the verbatim-recall question near
token ~4,000 (docs/ADVISOR-I3.md §4 I3 row).  Deterministic filler from a fixed
dull-sentence pool with enumerated ``Document N:`` headers — no RNG, no
timestamps: two builds are byte-identical (pinned in
spike/tests/test_needle_prompt.py).

Trap visibility: the fact at ~100 is OUTSIDE every SWA-128 window from the
question at ~4,000 (real_loop.py window mask; T3/T8) — retrieval can only
arrive through the GA layers (family gate; ``add_full_attention_sink_bias=false``).
A T3-flawed loop (window on GA too) misses the needle, so this spike doubles as
T3 live detection.  The filler is needle-free (asserted).

I4-X1a (ADVISOR-I4 §3.0, trap T28 §3.5): ``--chat`` renders the SAME text
through ``chat_template.jinja`` (sha ``853650be…``) as a single user turn with
the assistant generation prompt (thinking off = the served default) and greedy-
decodes honouring the FULL EOS set from ``generation_config.json`` — the I1b
raw-mode run read past EOS and printed an RL-rollout artifact (``REWARD:True``).
Expected clean result: ``KESTREL-41`` then EOS.  The T28 output checks assert:
no ``REWARD:``, no role-less assistant marker, nothing after EOS.

Runner (``--run``, fired by the captain via ``spike/run.sh needle`` /
``needle-chat``): builds the prompt, encodes with the REAL tokenizer
(BASE/tokenizer.json), greedy-decodes through spike.real_loop on the dev host's RTX 4090
and asserts the planted code returns VERBATIM.  ``MIMO26_WEIGHTS_DIR`` defaults
to the local weights copy (dev-host only — never the coordinator's 5090, never the Sparks).
"""
from __future__ import annotations

import argparse
import hashlib
import os

LOCAL_WEIGHTS = os.path.expanduser("~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL")

FACT_CODE = "KESTREL-41"
FACT = f"The access code for the north gate is {FACT_CODE}."
QUESTION = "What is the access code for the north gate? Answer with only the code."

# X1a (ADVISOR-I4 §3.0): the served EOS set — generation_config.json
# ("bos_token_id": 151643, "eos_token_id": [151643, 151645, 151672]).
# The raw-mode I1b row generated 151645 FIRST and the fixed-step loop ignored
# it (F1): chat gates honour the set and stop.
EOS_IDS = [151643, 151645, 151672]

# T29: MiMo special tokens are BUILT FROM PARTS so the literal never appears in a
# model-emitted tool argument (docs write them in square brackets).
IM_START = "\x3c" + "|im_start|>"

_FILLER = (
    "The committee reviewed the quarterly paperwork and filed it in the usual cabinet.",
    "A meeting on Tuesday covered the budget for office supplies and nothing else.",
    "The minutes from the previous session were approved without amendments.",
    "Staff rotated the storage room labels and recorded the change in the ledger.",
    "The maintenance log lists three filter replacements and one door adjustment.",
    "Everyone agreed the hallway painting can wait until the next fiscal cycle.",
    "The supply order included paper clips, folders, and two boxes of staples.",
    "Attendance was noted for the record and the session was adjourned on time.",
    "The archive team indexed the old folders by date and by department code.",
    "A reminder was posted about the annual equipment inventory next month.",
)


def filler_doc(i: int) -> str:
    """Deterministic dull document ``i`` (1-based); pool cycles, index persists."""
    return f"Document {i}: {_FILLER[(i - 1) % len(_FILLER)]}"


def build_text(head_docs: int, body_docs: int, start: int = 1) -> str:
    """Pure-text build: head filler, FACT, body filler, QUESTION (space-joined)."""
    parts = [filler_doc(start + i) for i in range(head_docs)]
    parts.append(FACT)
    parts += [filler_doc(start + head_docs + i) for i in range(body_docs)]
    parts.append(QUESTION)
    return " ".join(parts)


def place(encode, fact_at: int = 100, question_at: int = 4000, tol: int = 48):
    """Token-accurate placement: grow the filler deterministically until the
    FACT starts at/after ``fact_at`` tokens and the QUESTION at/after
    ``question_at`` (overshoot < one document; acceptance +-``tol``).

    ``encode`` = callable str -> list[int] (the real tokenizer at run time, a
    whitespace stub in unit tests).  Positions are measured on the exact
    prefixes of the composed text (token-boundary merges can shift the count by
    +-2; the tolerance absorbs it).  Returns ``(text, meta)``; raises on drift
    past tolerance — placement drift is the needle class's silent killer.
    """
    def ntok(s: str) -> int:
        return len(encode(s))

    head = 0
    fact_start = 0
    while fact_start < fact_at:
        head += 1
        fact_start = ntok(" ".join(filler_doc(i + 1) for i in range(head)))
    prefix = " ".join([filler_doc(i + 1) for i in range(head)] + [FACT])
    body = 0
    question_start = 0
    while question_start < question_at:  # measured on the composed prefix (exact)
        body += 1
        question_start = ntok(prefix + " " +
                              " ".join(filler_doc(head + 1 + j) for j in range(body)))
    text = build_text(head, body)
    meta = dict(fact_start=fact_start, question_start=question_start,
                total=ntok(text), head_docs=head, body_docs=body,
                fact_at=fact_at, question_at=question_at,
                prompt_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    assert fact_at - tol <= meta["fact_start"] <= fact_at + tol, meta
    assert question_at - tol <= meta["question_start"] <= question_at + tol, meta
    return text, meta


def render_chat(text: str, template_path: str | None = None,
                enable_thinking: bool = False) -> str:
    """X1a: render ``text`` as ONE user turn through ``chat_template.jinja``
    (sha ``853650be…``) with the assistant generation prompt appended and
    thinking off (the served default).  HF jinja semantics (StrictUndefined —
    a template wanting an undefined variable fails loud here, not silently)."""
    import json

    import jinja2

    base = os.environ.get("MIMO26_WEIGHTS_DIR", LOCAL_WEIGHTS)
    path = template_path or f"{base}/chat_template.jinja"
    src = open(path, encoding="utf-8").read()
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.filters["tojson"] = lambda v, ensure_ascii=False: json.dumps(v, ensure_ascii=ensure_ascii)
    tmpl = env.from_string(src)
    return tmpl.render(
        messages=[{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def check_chat_output(out: str) -> list[str]:
    """T28 (ADVISOR-I4 §3.5) chat-path output checks: no ``REWARD:`` (RL-rollout
    artifact), no role-less assistant marker (raw-mode leak).  'Nothing after EOS'
    holds by construction (greedy stops on the EOS set).  Returns violations."""
    bad = []
    if "REWARD:" in out:
        bad.append("REWARD: (RL-rollout artifact — T28)")
    if IM_START in out:
        bad.append("role-less assistant marker in output (T28)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="greedy through real_loop; assert verbatim retrieval")
    ap.add_argument("--chat", action="store_true",
                    help="X1a: render via chat_template.jinja, honour the EOS set")
    ap.add_argument("--probe", action="store_true",
                    help="dry device-footprint probe (no run, no torch)")
    ap.add_argument("--memtrace", action="store_true",
                    help="512-token, 2-layer live-tensor trace at op-class boundaries "
                         "(I1b suspect naming; --device cpu = dry format)")
    ap.add_argument("--device", default="cuda", help="memtrace device")
    ap.add_argument("--steps", type=int, default=16, help="greedy decode steps")
    ap.add_argument("--fact-at", type=int, default=100)
    ap.add_argument("--question-at", type=int, default=4000)
    ap.add_argument("--tol", type=int, default=48)
    ap.add_argument("--dump-ids", action="store_true",
                    help="print the raw and chat prompt token ids as JSON (no GPU; "
                         "feeds the X1b vLLM anchor window script)")
    args = ap.parse_args()
    if args.probe:
        from spike import footprint_probe
        return footprint_probe.main([])

    from tokenizers import Tokenizer
    os.environ.setdefault("MIMO26_WEIGHTS_DIR", LOCAL_WEIGHTS)
    tok = Tokenizer.from_file(f"{os.environ['MIMO26_WEIGHTS_DIR']}/tokenizer.json")
    text, meta = place(lambda s: tok.encode(s).ids, args.fact_at, args.question_at, args.tol)
    print(f"[needle] tokens={meta['total']} fact_start={meta['fact_start']} "
          f"question_start={meta['question_start']} head_docs={meta['head_docs']} "
          f"body_docs={meta['body_docs']} chunk={os.environ.get('MIMO26_SPIKE_CHUNK', '0')} "
          f"sha256={meta['prompt_sha256']}")
    print(f"[needle] fact={FACT!r}")
    print(f"[needle] tail={text[-160:]!r}")
    if args.dump_ids:
        # X1b prep: the exact id lists the spike uses, for the vLLM anchor POST.
        import json as _json
        raw_ids = tok.encode(text).ids
        chat_ids = tok.encode(render_chat(text)).ids
        print(_json.dumps({"raw_ids": raw_ids, "chat_ids": chat_ids,
                           "raw_sha256": meta["prompt_sha256"],
                           "eos_ids": EOS_IDS}))
        return 0
    if args.memtrace:
        # I1b (1): EMPIRICAL TRACE — 512-token input, first 2 layers, one line
        # per op-class boundary naming the live set (fired by the captain).
        from spike import real_loop as RL  # torch import lives here (run env only)
        ids = tok.encode(text).ids[:512]
        model = RL.RealModel(RL.Reader(), device=args.device, memtrace=True, layer_cap=2)
        kv = RL.KV(model.n_layers, model.device)
        with RL.torch.inference_mode():
            model.forward(ids, kv, None)
        for line in RL.TRACE:
            print(line)
        print(f"[memtrace] done ids={len(ids)} layers=2 device={args.device} lines={len(RL.TRACE)}")
        return 0

    if not args.run:
        return 0

    from spike import real_loop as RL  # torch import lives here (run env only)
    if args.chat:
        rendered = render_chat(text)
        ids = tok.encode(rendered).ids
        print(f"[needle-chat] rendered_tokens={len(ids)} raw_tokens={meta['total']} "
              f"rendered_sha256={hashlib.sha256(rendered.encode('utf-8')).hexdigest()} "
              f"template_sha8=853650bee57b thinking=off eos={EOS_IDS}")
        model = RL.RealModel(RL.Reader())
        gen, n = RL.greedy(model, ids, args.steps, eos=set(EOS_IDS))
        out = tok.decode(gen)
        viol = check_chat_output(out)
        stopped = bool(gen) and gen[-1] in EOS_IDS
        immediate = bool(gen) and gen[0] in EOS_IDS
        ok = FACT_CODE in out and not viol and stopped and not immediate
        print(f"[needle-chat] prompt_tokens={len(ids)} gen_ids={gen} -> {out!r}")
        print(f"[needle-chat] stopped_on_eos={stopped} immediate_eos={immediate} "
              f"t28_violations={viol}")
        if ok:
            print(f"RESULT: PASS X1a chat needle retrieved verbatim: {FACT_CODE} (EOS honoured)")
            return 0
        print(f"RESULT: FAIL X1a chat needle (want {FACT_CODE} then EOS; viol={viol} "
              f"stopped={stopped} immediate={immediate}) — T28 red flag: STOP and diagnose "
              "(ADVISOR-I4 §3.0), do not rationalise.")
        return 1

    ids = tok.encode(text).ids
    model = RL.RealModel(RL.Reader())
    gen, n = RL.greedy(model, ids, args.steps)
    out = tok.decode(gen)
    ok = FACT_CODE in out
    print(f"[needle] prompt_tokens={len(ids)} gen_ids={gen} -> {out!r}")
    if ok:
        print(f"RESULT: PASS needle retrieved verbatim: {FACT_CODE} (raw mode — T28: a raw-mode probe, EOS ignored)")
        return 0
    print(f"RESULT: FAIL needle NOT retrieved (want verbatim {FACT_CODE}); "
          "salad? name the seam per real_loop.py:36 (T1/T2/T3/T5/T8/T9).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
