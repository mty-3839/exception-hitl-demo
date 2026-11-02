"""Exception ops copilot demo: classify → suggest → HITL → log/eval."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import streamlit as st

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - missing dep shown in UI
    OpenAI = None  # type: ignore[misc, assignment]


SAMPLE_EXCEPTIONS: list[dict[str, str]] = [
    {
        "id": "delay-refund",
        "title": "Delayed order + refund request",
        "text": (
            "Order #A1283 delayed 48h, customer requesting refund, "
            "warehouse claims carrier issue"
        ),
    },
    {
        "id": "inventory-mismatch",
        "title": "Inventory mismatch",
        "text": (
            "Inventory mismatch: system shows 12 units, physical count 7 units "
            "for SKU-XR99"
        ),
    },
    {
        "id": "payment-pending",
        "title": "Paid but still Pending",
        "text": (
            'Payment received but order status still "Pending", '
            "customer already charged"
        ),
    },
    {
        "id": "late-damage-return",
        "title": "Damage after return window",
        "text": (
            "Damaged goods reported by customer, photo uploaded, "
            "but return window expired 2 days ago"
        ),
    },
    {
        "id": "duplicate-pick",
        "title": "Duplicate order already picked",
        "text": (
            "Duplicate order created due to double-click, both orders already picked"
        ),
    },
]

CATEGORIES = [
    "Logistics / Delay",
    "Inventory Discrepancy",
    "Payment / Order Status Mismatch",
    "Returns / Damage / Policy",
    "Duplicate / Duplicate Fulfillment",
    "Other",
]

HIGH_RISK_FLAGS = {"money", "customer_commitment", "policy_gray_area"}

SYSTEM_PROMPT = """You are an operations exception copilot for an e-commerce / warehouse team.

You MUST follow this reasoning order:
1) Classify the exception into exactly one category.
2) Propose a concrete, executable suggested action (who does what, in what order, with checks).
3) Score confidence 0-100 and name the main uncertainty factors.
4) Decide whether to recommend escalate-to-human.

Escalation is REQUIRED when ANY of these are true:
- confidence < 70
- the case involves money movement (refund, charge, credit, write-off)
- the case involves a customer-facing promise (SLA, apology credit, guaranteed ship date)
- the case sits in a policy gray area (expired window, exception to SOP, conflicting claims)

Do not invent missing facts. If data is incomplete, lower confidence and say what to verify.

Return ONLY valid JSON with this exact schema (no markdown):
{
  "category": "one of: Logistics / Delay | Inventory Discrepancy | Payment / Order Status Mismatch | Returns / Damage / Policy | Duplicate / Duplicate Fulfillment | Other",
  "suggested_action": "numbered or bulleted executable steps as a single string",
  "confidence": 0,
  "rationale": "2-4 sentences",
  "uncertainty_factors": ["short strings"],
  "risk_flags": ["subset of: money, customer_commitment, policy_gray_area, operational"],
  "recommend_escalate": true,
  "escalate_reasons": ["short strings"]
}
"""

MOCK_BY_ID: dict[str, dict[str, Any]] = {
    "delay-refund": {
        "category": "Logistics / Delay",
        "suggested_action": (
            "1) Pull tracking + warehouse scan timestamps for Order #A1283.\n"
            "2) Confirm delay is carrier-side vs warehouse miss.\n"
            "3) Do NOT issue refund yet. Offer: expedite remaining shipment OR wait-and-credit.\n"
            "4) If customer insists on cancel/refund, route to CX lead with delay evidence.\n"
            "5) Log carrier claim ticket if scans support it."
        ),
        "confidence": 64,
        "rationale": (
            "Delay plus refund request mixes logistics root-cause with money. "
            "Warehouse blame on the carrier is unverified. Auto-refund would be premature."
        ),
        "uncertainty_factors": [
            "No tracking/scan evidence attached",
            "Unknown customer SLA / paid shipping tier",
            "Refund vs cancel vs wait preference not confirmed",
        ],
        "risk_flags": ["money", "customer_commitment"],
        "recommend_escalate": True,
        "escalate_reasons": [
            "confidence < 70",
            "money: refund requested",
            "customer_commitment: delay compensation",
        ],
    },
    "inventory-mismatch": {
        "category": "Inventory Discrepancy",
        "suggested_action": (
            "1) Freeze outbound allocation for SKU-XR99 until recount.\n"
            "2) Cycle-count the bin(s) and adjacent locations (mis-slot risk).\n"
            "3) Reconcile last 48h receipts, picks, returns, and adjustments.\n"
            "4) Post adjustment only after dual verification; note 5-unit shrink.\n"
            "5) If open orders depend on the 12-unit figure, re-promise or split-ship."
        ),
        "confidence": 78,
        "rationale": (
            "Classic system vs physical variance. Action path is standard, but "
            "root cause (theft, mis-pick, receiving error) is unknown."
        ),
        "uncertainty_factors": [
            "Which location/bin was counted",
            "Open order demand against SKU-XR99",
            "Whether variance is recurring",
        ],
        "risk_flags": ["operational"],
        "recommend_escalate": False,
        "escalate_reasons": [],
    },
    "payment-pending": {
        "category": "Payment / Order Status Mismatch",
        "suggested_action": (
            "1) Match payment id / last-4 / amount / timestamp to the order.\n"
            "2) If captured successfully, flip order from Pending to Paid and release to fulfillment.\n"
            "3) If capture succeeded but webhook failed, replay status sync; do not recapture.\n"
            "4) Message the customer that they will not be charged twice.\n"
            "5) If payment is pending/auth-only, do not fulfill; escalate to payments."
        ),
        "confidence": 61,
        "rationale": (
            "Customer already charged is a high-risk money state. Status stuck on "
            "Pending is often a webhook/sync bug, but double-charge risk must be ruled out."
        ),
        "uncertainty_factors": [
            "Capture vs auth-only not confirmed",
            "Possible duplicate payment attempt",
            "PSP vs OMS source of truth unknown",
        ],
        "risk_flags": ["money", "customer_commitment"],
        "recommend_escalate": True,
        "escalate_reasons": [
            "confidence < 70",
            "money: customer already charged",
            "customer_commitment: must not double-charge",
        ],
    },
    "late-damage-return": {
        "category": "Returns / Damage / Policy",
        "suggested_action": (
            "1) Review uploaded photos for in-transit vs customer-caused damage.\n"
            "2) Check order date, delivery date, and published return window.\n"
            "3) Do NOT auto-approve a full refund past window.\n"
            "4) Options for human: goodwill partial credit, replacement if in stock, or deny with policy cite.\n"
            "5) If photos show packing failure, file carrier/warehouse quality ticket regardless of window."
        ),
        "confidence": 58,
        "rationale": (
            "Return window expired (policy gray area) plus damage photos (possible warehouse/carrier fault). "
            "Goodwill vs strict policy is a human judgment, not an auto-decision."
        ),
        "uncertainty_factors": [
            "Photo quality / damage attribution",
            "Exact window policy and exceptions",
            "Customer value / prior goodwill",
        ],
        "risk_flags": ["money", "policy_gray_area", "customer_commitment"],
        "recommend_escalate": True,
        "escalate_reasons": [
            "confidence < 70",
            "policy_gray_area: window expired 2 days",
            "money: refund/replacement at stake",
        ],
    },
    "duplicate-pick": {
        "category": "Duplicate / Duplicate Fulfillment",
        "suggested_action": (
            "1) Identify both order ids and confirm they are true duplicates (same SKU/address/payment).\n"
            "2) Stop pack-out / handoff if either carton is still in warehouse.\n"
            "3) If both already picked: hold the second carton at pack; do not ship both.\n"
            "4) Cancel/void the duplicate order after inventory is put back.\n"
            "5) If the second already shipped, start intercept or prepaid return; do not charge twice."
        ),
        "confidence": 72,
        "rationale": (
            "Double-click duplicate with both already picked is operationally clear, "
            "but shipping both would create a customer/money incident. Hold-and-restock is the default."
        ),
        "uncertainty_factors": [
            "Whether either carton already left the building",
            "Whether payment was captured twice",
            "Same vs slightly different SKU mix",
        ],
        "risk_flags": ["money", "operational"],
        "recommend_escalate": True,
        "escalate_reasons": [
            "money: possible double charge / extra shipment cost",
            "customer_commitment: duplicate delivery risk",
        ],
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def get_model_name() -> str:
    env_model = os.environ.get("OPENAI_MODEL", "").strip()
    if env_model:
        return env_model
    try:
        secret_model = str(st.secrets.get("OPENAI_MODEL", "")).strip()
        if secret_model:
            return secret_model
    except Exception:
        pass
    return "gpt-4o-mini"


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output was not valid JSON.")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")
    return data


def apply_escalation_rules(parsed: dict[str, Any]) -> dict[str, Any]:
    """Enforce demo policy even if the model under-escalates."""
    confidence = int(parsed.get("confidence", 0) or 0)
    confidence = max(0, min(100, confidence))
    flags = {str(f).strip().lower() for f in (parsed.get("risk_flags") or [])}
    reasons = [str(r) for r in (parsed.get("escalate_reasons") or []) if str(r).strip()]

    must = False
    if confidence < 70:
        must = True
        tag = "confidence < 70"
        if tag not in reasons:
            reasons.append(tag)
    if flags & {"money", "customer_commitment", "policy_gray_area"}:
        must = True
        for flag in sorted(flags & HIGH_RISK_FLAGS):
            tag = f"high-risk: {flag}"
            if tag not in reasons and not any(flag in r.lower() for r in reasons):
                reasons.append(tag)

    parsed["confidence"] = confidence
    parsed["recommend_escalate"] = bool(parsed.get("recommend_escalate")) or must
    parsed["escalate_reasons"] = reasons
    parsed["risk_flags"] = sorted(flags) if flags else []
    if parsed.get("category") not in CATEGORIES:
        parsed["category"] = "Other"
    parsed["suggested_action"] = str(parsed.get("suggested_action") or "").strip()
    parsed["rationale"] = str(parsed.get("rationale") or "").strip()
    parsed["uncertainty_factors"] = [
        str(x).strip() for x in (parsed.get("uncertainty_factors") or []) if str(x).strip()
    ]
    return parsed


def mock_analyze(text: str, sample_id: str | None) -> dict[str, Any]:
    if sample_id and sample_id in MOCK_BY_ID:
        return apply_escalation_rules(dict(MOCK_BY_ID[sample_id]))
    lowered = text.lower()
    if "refund" in lowered or "delayed" in lowered:
        base = dict(MOCK_BY_ID["delay-refund"])
    elif "inventory" in lowered or "sku" in lowered:
        base = dict(MOCK_BY_ID["inventory-mismatch"])
    elif "payment" in lowered or "pending" in lowered:
        base = dict(MOCK_BY_ID["payment-pending"])
    elif "damaged" in lowered or "return window" in lowered:
        base = dict(MOCK_BY_ID["late-damage-return"])
    elif "duplicate" in lowered:
        base = dict(MOCK_BY_ID["duplicate-pick"])
    else:
        base = {
            "category": "Other",
            "suggested_action": (
                "1) Restate the exception and missing facts.\n"
                "2) Pull order, inventory, and payment records.\n"
                "3) Draft a recommended action only after verification.\n"
                "4) Escalate if money, customer promise, or policy exception is involved."
            ),
            "confidence": 45,
            "rationale": "Offline mock has no model context for this custom text.",
            "uncertainty_factors": ["Custom exception not in sample set"],
            "risk_flags": ["policy_gray_area"],
            "recommend_escalate": True,
            "escalate_reasons": ["confidence < 70", "unfamiliar case"],
        }
    return apply_escalation_rules(base)


def llm_analyze(text: str, model: str, api_key: str) -> dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify this operations exception and propose the next action.\n\n"
                    f"EXCEPTION:\n{text}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    parsed = extract_json(content)
    parsed = apply_escalation_rules(parsed)
    parsed["_model"] = model
    parsed["_raw"] = content
    return parsed


def init_state() -> None:
    defaults: dict[str, Any] = {
        "exception_text": SAMPLE_EXCEPTIONS[0]["text"],
        "active_sample_id": SAMPLE_EXCEPTIONS[0]["id"],
        "ai_result": None,
        "ai_error": "",
        "edited_action": "",
        "hitl_escalate_mode": "follow",
        "decision_log": [],
        "last_confirmed_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_sample(sample: dict[str, str]) -> None:
    st.session_state.exception_text = sample["text"]
    st.session_state.active_sample_id = sample["id"]
    st.session_state.ai_result = None
    st.session_state.ai_error = ""
    st.session_state.edited_action = ""
    st.session_state.hitl_escalate_mode = "follow"
    st.session_state.last_confirmed_id = None


def compute_stats(log: list[dict[str, Any]]) -> dict[str, Any]:
    if not log:
        return {"n": 0, "avg_confidence": None, "escalate_rate": None, "edit_rate": None}
    n = len(log)
    avg_conf = sum(int(row["ai_confidence"]) for row in log) / n
    escalate_rate = sum(1 for row in log if row["final_escalate"]) / n
    edit_rate = sum(1 for row in log if row["human_modified"]) / n
    return {
        "n": n,
        "avg_confidence": avg_conf,
        "escalate_rate": escalate_rate,
        "edit_rate": edit_rate,
    }


def confidence_label(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 70:
        return "Medium"
    return "Low — escalate"


def render_ai_result(result: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Category", result["category"])
    c2.metric("Confidence", f"{result['confidence']}/100", confidence_label(result["confidence"]))
    c3.metric(
        "Escalate?",
        "Yes" if result["recommend_escalate"] else "No",
        "Human review" if result["recommend_escalate"] else "AI can proceed",
    )
    st.markdown("**Suggested action**")
    st.info(result["suggested_action"])
    st.markdown("**Rationale**")
    st.write(result["rationale"])
    u1, u2 = st.columns(2)
    with u1:
        st.caption("Uncertainty factors")
        factors = result.get("uncertainty_factors") or []
        if factors:
            for item in factors:
                st.write(f"- {item}")
        else:
            st.write("- (none listed)")
    with u2:
        st.caption("Risk flags / escalate reasons")
        flags = result.get("risk_flags") or []
        reasons = result.get("escalate_reasons") or []
        if flags:
            st.write("Flags: " + ", ".join(flags))
        for reason in reasons:
            st.write(f"- {reason}")
        if not flags and not reasons:
            st.write("- (none)")


def main() -> None:
    st.set_page_config(
        page_title="Exception HITL Demo",
        page_icon="⚡",
        layout="wide",
    )
    init_state()

    st.title("Ops Exception Copilot")
    st.caption(
        "Demo loop: exception → classify + action + confidence → HITL confirm → in-memory log / eval. "
        "Not a production system."
    )

    api_key = get_api_key()
    default_model = get_model_name()

    with st.sidebar:
        st.subheader("Model")
        mock_mode = st.toggle(
            "Offline mock (no LLM)",
            value=not bool(api_key),
            help="Use canned responses so the HITL loop still runs without an API key.",
        )
        model = st.text_input("OpenAI model", value=default_model)
        if mock_mode:
            st.success("Mock mode on — no API call.")
        elif api_key:
            st.success("API key loaded from env / secrets.")
        else:
            st.error("No OPENAI_API_KEY in environment or st.secrets.")
        st.markdown(
            "Policy: escalate if **confidence < 70** or the case involves "
            "**money / customer promise / policy gray area**."
        )

    left, mid = st.columns([1, 1.35], gap="large")

    with left:
        st.subheader("1. Exception")
        st.caption("One-click samples")
        for sample in SAMPLE_EXCEPTIONS:
            if st.button(sample["title"], key=f"sample-{sample['id']}", use_container_width=True):
                load_sample(sample)
                st.rerun()

        text = st.text_area(
            "Exception text",
            key="exception_text",
            height=160,
        )
        if st.button("Analyze exception", type="primary", use_container_width=True):
            cleaned = (text or "").strip()
            if not cleaned:
                st.session_state.ai_error = "Paste or load an exception first."
                st.session_state.ai_result = None
            else:
                st.session_state.ai_error = ""
                known_ids = {s["text"]: s["id"] for s in SAMPLE_EXCEPTIONS}
                sample_id = known_ids.get(cleaned, st.session_state.get("active_sample_id"))
                if cleaned != next((s["text"] for s in SAMPLE_EXCEPTIONS if s["id"] == sample_id), None):
                    sample_id = known_ids.get(cleaned)
                    st.session_state.active_sample_id = sample_id
                try:
                    if mock_mode:
                        result = mock_analyze(cleaned, sample_id)
                        result["_model"] = "offline-mock"
                    else:
                        if not api_key:
                            raise RuntimeError(
                                "Missing OPENAI_API_KEY. Set the env var, add st.secrets, or enable Offline mock."
                            )
                        result = llm_analyze(cleaned, model=model, api_key=api_key)
                    result["_source_text"] = cleaned
                    result["_analyzed_at"] = utc_now_iso()
                    st.session_state.ai_result = result
                    st.session_state.edited_action = result["suggested_action"]
                    st.session_state.hitl_escalate_mode = (
                        "follow"
                    )
                    st.session_state.last_confirmed_id = None
                except Exception as exc:  # surface API/parse errors in the demo UI
                    st.session_state.ai_result = None
                    st.session_state.ai_error = str(exc)
            st.rerun()

    with mid:
        st.subheader("2. AI classification")
        if st.session_state.ai_error:
            st.error(st.session_state.ai_error)
        result = st.session_state.ai_result
        if not result:
            st.write("Load a sample and click **Analyze exception**.")
        else:
            st.caption(
                f"Analyzed {result.get('_analyzed_at', '')} · model `{result.get('_model', 'unknown')}`"
            )
            render_ai_result(result)

    st.divider()
    st.subheader("3. Human-in-the-loop")
    result = st.session_state.ai_result
    if not result:
        st.write("Analyze an exception before confirming a decision.")
    else:
        st.text_area(
            "Suggested action (editable)",
            key="edited_action",
            height=150,
        )
        mode = st.radio(
            "Escalation control",
            options=["follow", "force", "override"],
            format_func=lambda x: {
                "follow": "Follow AI recommendation",
                "force": "Force escalate to human",
                "override": "Override — do not escalate (auto-resolve)",
            }[x],
            key="hitl_escalate_mode",
            horizontal=True,
        )
        ai_escalate = bool(result["recommend_escalate"])
        if mode == "force":
            final_escalate = True
        elif mode == "override":
            final_escalate = False
        else:
            final_escalate = ai_escalate

        action_changed = (st.session_state.edited_action or "").strip() != (
            result.get("suggested_action") or ""
        ).strip()
        escalate_changed = final_escalate != ai_escalate
        human_modified = action_changed or escalate_changed

        preview = st.columns(3)
        preview[0].write(f"Final escalate: **{'Yes' if final_escalate else 'No'}**")
        preview[1].write(f"Action edited: **{'Yes' if action_changed else 'No'}**")
        preview[2].write(f"Counts as human modify: **{'Yes' if human_modified else 'No'}**")

        already = st.session_state.last_confirmed_id == result.get("_analyzed_at")
        confirm = st.button(
            "Confirm decision",
            type="primary",
            disabled=already,
        )
        if already:
            st.success("This analysis is already in the decision log. Analyze again to add another row.")
        if confirm and not already:
            log_row = {
                "ts": utc_now_iso(),
                "exception": result.get("_source_text", st.session_state.exception_text),
                "category": result["category"],
                "ai_action": result["suggested_action"],
                "final_action": (st.session_state.edited_action or "").strip(),
                "ai_confidence": int(result["confidence"]),
                "ai_escalate": ai_escalate,
                "final_escalate": final_escalate,
                "hitl_mode": mode,
                "human_modified": human_modified,
                "rationale": result.get("rationale", ""),
                "model": result.get("_model", ""),
            }
            st.session_state.decision_log = [log_row, *st.session_state.decision_log]
            st.session_state.last_confirmed_id = result.get("_analyzed_at")
            st.rerun()

    st.divider()
    st.subheader("4. Eval snapshot + decision log")
    log: list[dict[str, Any]] = st.session_state.decision_log
    stats = compute_stats(log)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decisions logged", stats["n"])
    m2.metric(
        "Avg confidence",
        "—" if stats["avg_confidence"] is None else f"{stats['avg_confidence']:.1f}",
    )
    m3.metric(
        "Escalate rate",
        "—" if stats["escalate_rate"] is None else f"{stats['escalate_rate'] * 100:.0f}%",
    )
    m4.metric(
        "Human modify rate",
        "—" if stats["edit_rate"] is None else f"{stats['edit_rate'] * 100:.0f}%",
    )

    if not log:
        st.write("No confirmed decisions yet. Confirm at least one HITL decision to populate the log.")
    else:
        show_n = min(10, len(log))
        st.caption(f"Showing the latest {show_n} of {len(log)} (session only — refresh clears).")
        for i, row in enumerate(log[:10], start=1):
            badge = "ESCALATE" if row["final_escalate"] else "AUTO"
            edited = " · edited" if row["human_modified"] else ""
            with st.expander(
                f"{i}. [{badge}] {row['category']} · conf {row['ai_confidence']}{edited} · {row['ts']}",
                expanded=(i == 1),
            ):
                st.write(row["exception"])
                st.markdown("**Final action**")
                st.write(row["final_action"])
                if row["human_modified"] and row["final_action"] != row["ai_action"]:
                    st.markdown("**Original AI action**")
                    st.write(row["ai_action"])
                st.caption(
                    f"HITL mode: {row['hitl_mode']} · AI escalate: {row['ai_escalate']} · "
                    f"final escalate: {row['final_escalate']} · model: {row['model']}"
                )

        if st.button("Clear session log"):
            st.session_state.decision_log = []
            st.session_state.last_confirmed_id = None
            st.rerun()


if __name__ == "__main__":
    main()
