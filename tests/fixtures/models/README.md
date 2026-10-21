# Official checkpoint snapshots

These are configuration-only, offline fixtures, retrieved on 2026-10-21 from
the official Hugging Face repositories named in each `source.json`. The `sha`
pins the upstream revision; `config_sha256` protects the checked-in fixture.
Qwen JSON is reserialized without changing values; its original byte hash is
also recorded. No weights, tokenizer implementations or executable checkpoint
code are included.

Profiles describe metadata and prompt protocols, not verified GPU support.
DeepSeek V4.1 intentionally has no Jinja fixture: its official `README.md` and
`encoding/README.md` describe native encoding, spaced DSML tags, and numeric
reasoning budgets (with `low`/`high`/`max` aliases).

Parser spellings are checked against the installed image, not assumed from a
version number. Reference implementations used when adding the profiles:

- https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/function_call_parser.py
- https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/parser/reasoning_parser.py
- https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/__init__.py
- https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/__init__.py
