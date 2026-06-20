"""Contract clause extractor.

Intentionally single-file to demonstrate that Overmind helps even for
tool-less, long-context prompt agents. Overmind has to earn its score
entirely from system-prompt / output-schema rewrites here.

Deliberate sub-optimalities:
- Prompt asks for fields but doesn't require evidence spans
- No date-format normalisation rule
- `red_flags` is open-ended; model may skip auto-renewal clauses
- No fallback for missing fields (sometimes emits "N/A" strings, sometimes null)
- Uses gpt-4o for all cases - could be tiered by document length
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from overmind import entry_point, init

load_dotenv()

init(service_name="contract-extractor")

_MODEL = os.environ.get("CONTRACT_EXTRACTOR_MODEL", "gpt-4o")


def _client() -> OpenAI:
    return OpenAI()


SYSTEM_PROMPT = """You are a paralegal. Extract structured information from the contract below.

Return a JSON object with exactly these 8 keys:
- parties (list of strings): all named parties to the contract
- effective_date (string or null): the date the contract takes effect; null if not stated
- term (string or null): the duration or end date of the contract; null if not stated
- termination (list of strings): all termination-related clauses and conditions
- liability_cap (string or null): the liability cap amount or description; null if absent
- auto_renewal (string or null): the auto-renewal clause description; null if no auto-renewal clause exists
- ip_assignment (string or null): the IP assignment clause description; null if no IP assignment clause exists
- red_flags (list of strings): MUST be non-empty — include at least one entry for each applicable concern below:
  (1) Auto-renewal risk: flag any short cancellation window or automatic rollover provision
  (2) Uncapped or absent liability: flag when liability_cap is null or unlimited — note the unlimited liability exposure
  (3) Broad or perpetual IP assignment: flag when ip_assignment transfers broad or perpetual rights to the counterparty
  (4) Unilateral termination rights: flag any clause allowing one party to terminate without cause or with minimal notice
  (5) Any other unusual or one-sided provisions
  If none of the above apply, include: "No significant red flags identified — manual review recommended"
  NEVER return an empty red_flags list.

For optional string fields (effective_date, term, liability_cap, auto_renewal, ip_assignment):
  - Use null if the clause is absent from the contract
  - NEVER use false, true, empty string, or "N/A" — only null or a descriptive string

Ground your extraction exclusively in the contract text provided. Do not infer or assume clauses that are not present.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.removeprefix("json")
        text = text.rsplit("```", 1)[0]

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        candidates.append(text[start : end + 1])

    for chunk in candidates:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue

    return {
        "parties": [],
        "effective_date": None,
        "term": None,
        "termination": [],
        "liability_cap": None,
        "auto_renewal": None,
        "ip_assignment": None,
        "red_flags": [text[:200]],
    }


_STRING_OPTIONAL_FIELDS = (
    "effective_date",
    "term",
    "liability_cap",
    "auto_renewal",
    "ip_assignment",
)
_LIST_FIELDS = ("parties", "termination", "red_flags")
_DEFAULT_KEYS: dict[str, Any] = {
    "parties": [],
    "effective_date": None,
    "term": None,
    "termination": [],
    "liability_cap": None,
    "auto_renewal": None,
    "ip_assignment": None,
    "red_flags": [],
}


def _post_process(data: dict[str, Any]) -> dict[str, Any]:
    result = dict(_DEFAULT_KEYS)
    result.update(data)

    for field in _STRING_OPTIONAL_FIELDS:
        val = result.get(field)
        if isinstance(val, bool):
            result[field] = None if not val else "Present (see contract)"
        elif val == "" or val == "N/A":
            result[field] = None

    for field in _LIST_FIELDS:
        val = result.get(field)
        if not isinstance(val, list):
            result[field] = [] if val is None else [str(val)]

    if not result["red_flags"]:
        result["red_flags"] = ["No explicit red flags identified by extractor — manual review recommended"]

    return result


@entry_point("Contract Extractor")
def run(input_data: dict[str, Any]) -> dict[str, Any]:
    contract_text = input_data.get("contract_text", "")
    clause_types = input_data.get("clause_types", [])

    user_msg = (
        f"Contract:\n{contract_text[:20000]}\n\n"
        f"Clause types of interest: {', '.join(clause_types) if clause_types else 'all standard'}\n"
        "IMPORTANT: Extract ALL fields from the contract text above. "
        "For red_flags, you MUST check and report on: auto-renewal risk, liability cap adequacy, "
        "IP assignment breadth, unilateral termination rights, and any other one-sided provisions. "
        "Do not return an empty red_flags list.\n"
        "Return the extraction JSON."
    )

    client = _client()
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    raw = _extract_json(resp.choices[0].message.content or "")
    return _post_process(raw)
