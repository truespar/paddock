"""Reference renders for tests/tojson_parity.rs: every fixture chat template
through transformers' own chat-template environment, on normalized.json.

    python render_reference.py      # writes <name>.txt beside this file

The context mirrors Paddock's render: messages, tools, add_generation_prompt
true, enable_thinking true, no special-token strings. `strftime_now` is pinned
to 2026-10-08 (gpt-oss prints the date; the Rust gate maps today onto it).
"""
import datetime
import json
import pathlib

import transformers
from transformers.utils.chat_template_utils import _compile_jinja_template

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE.parent
NAMES = ["kolibri", "laguna", "qwen35", "qwen36", "qwen38", "gemma4", "gemma4_qat",
         "granite", "granite_vision", "gptoss", "muse"]

conv = json.loads((HERE / "normalized.json").read_text())
for name in NAMES:
    src = (FIXTURES / f"{name}_chat_template.jinja").read_text()
    tmpl = _compile_jinja_template(src)
    tmpl.environment.globals["strftime_now"] = lambda fmt: datetime.date(2026, 10, 8).strftime(fmt)
    out = tmpl.render(
        messages=conv["messages"], tools=conv["tools"],
        add_generation_prompt=True, enable_thinking=True)
    (HERE / f"{name}.txt").write_text(out)
    print(f"{name}: {len(out)} chars (transformers {transformers.__version__})")
