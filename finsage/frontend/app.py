"""Gradio chat UI for FinSage AI."""

import os
import time
import uuid
from typing import Any

import gradio as gr
import httpx


API_BASE_URL = os.getenv("FINSAGE_API_URL", "http://localhost:8000")
MAX_QUERIES = 5

EXAMPLE_QUERIES = [
    ("📊 Stock Analysis", "Should I buy Reliance stock today? How are its fundamentals?"),
    ("💰 Salary & Budget", "My salary is ₹80,000. How to manage it properly?"),
    ("📉 Tax Calculation", "I made 50k profit in TCS after 8 months. What is my tax?"),
    ("📈 Intraday & Options", "Nifty option chain max pain and PCR today?"),
    ("🛡️ Term Insurance", "Should I buy a term plan or ULIP?"),
    ("🏦 Home Loan EMI", "Home loan of 50 lakhs for 15 years, what is EMI and tax benefit?"),
    ("💼 Mutual Fund SIP", "Is Parag Parikh Flexi Cap a good fund for monthly SIP?"),
    ("🥇 Gold Investment", "Should I buy physical gold or SGB for investment?"),
    ("👴 Retirement (NPS)", "What are the tax benefits of NPS under 80CCD?"),
]


def init_state() -> dict[str, Any]:
    return {
        "history": [],
        "query_count": 0,
        "user_id": f"gradio_{uuid.uuid4().hex[:12]}",
    }


def ensure_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return init_state()

    state.setdefault("history", [])
    state.setdefault("query_count", 0)
    state.setdefault("user_id", f"gradio_{uuid.uuid4().hex[:12]}")
    return state


def intent_badge(intent: str) -> str:
    intent_colors = {
        "stock": "🟢",
        "index": "🔵",
        "tax": "🟠",
        "salary": "🟣",
        "general": "⚪",
    }
    return f"**Intent:** {intent_colors.get(intent, '⚪')} {intent}"


def confidence_bar(confidence: int) -> str:
    bounded_confidence = max(0, min(100, int(confidence)))
    return (
        "<div style='margin-top:8px;'>"
        f"<div><b>Confidence Score:</b> {bounded_confidence}%</div>"
        "<div style='width:100%;height:12px;background:#e5e7eb;border-radius:999px;overflow:hidden;'>"
        f"<div style='width:{bounded_confidence}%;height:100%;background:#00d4aa;'></div>"
        "</div>"
        "</div>"
    )


def trace_block(trace: list[Any], elapsed: float) -> str:
    lines = ["<details><summary><b>🔍 See how this was analyzed</b></summary><ul>"]
    for step in trace:
        lines.append(f"<li><code>{str(step)}</code></li>")
    lines.append(f"<li>⏱️ Total time: {elapsed}s</li>")
    lines.append("</ul></details>")
    return "".join(lines)


def history_block(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""

    blocks = ["#### 📜 Recent Queries"]
    for index, item in enumerate(history):
        icon = "🕛" if index == 0 else "🕐"
        short_query = f"{item['query'][:80]}..." if len(item["query"]) > 80 else item["query"]
        short_answer = item["answer"][:500] + ("..." if len(item["answer"]) > 500 else "")
        blocks.append(
            "<details {open_flag}><summary>{icon} {query}</summary>"
            "<div style='margin:8px 0 0 0;'>{answer}</div>"
            "<div style='font-size:0.9rem;color:#6b7280;'>Intent: {intent} | Confidence: {confidence}% | Time: {time}s</div>"
            "</details>".format(
                open_flag="open" if index == 0 else "",
                icon=icon,
                query=short_query,
                answer=short_answer,
                intent=item["intent"],
                confidence=item["confidence"],
                time=item["time"],
            )
        )
    return "\n".join(blocks)


def status_text(query_count: int) -> str:
    remaining = max(0, MAX_QUERIES - query_count)
    if query_count >= MAX_QUERIES:
        return (
            f"Session query limit: {query_count}/{MAX_QUERIES} used ({remaining} left)\n\n"
            "🛑 You have reached the maximum limit of 5 queries per session. Refresh the page to start a new session."
        )
    return f"Session query limit: {query_count}/{MAX_QUERIES} used ({remaining} left)"


def analyze(query: str, state: dict[str, Any] | None):
    state = ensure_state(state)

    if state["query_count"] >= MAX_QUERIES:
        return (
            "",
            confidence_bar(0),
            "",
            "",
            history_block(state["history"]),
            status_text(state["query_count"]),
            "",
            state,
        )

    if not query or not query.strip():
        return (
            "⚠️ Please enter a question first.",
            confidence_bar(0),
            "",
            "",
            history_block(state["history"]),
            status_text(state["query_count"]),
            query,
            state,
        )

    try:
        start_time = time.time()
        response = httpx.post(
            f"{API_BASE_URL}/api/chat",
            json={"user_id": state["user_id"], "query": query.strip()},
            timeout=120.0,
        )
        elapsed = round(time.time() - start_time, 1)

        if response.status_code == 200:
            data = response.json()
            item = {
                "query": query,
                "answer": data.get("answer", "No answer generated."),
                "confidence": data.get("confidence", 0) or 0,
                "intent": data.get("intent", "general"),
                "trace": data.get("trace", []),
                "time": elapsed,
            }
            state["history"].insert(0, item)
            state["history"] = state["history"][:5]
            state["query_count"] += 1

            answer = f"✅ Analysis complete in {elapsed}s\n\n{item['answer']}"
            confidence = confidence_bar(item["confidence"])
            intent = intent_badge(item["intent"])
            trace = trace_block(item["trace"], elapsed)
            history = history_block(state["history"])
            status = status_text(state["query_count"])
            return answer, confidence, intent, trace, history, status, "", state

        if response.status_code == 400:
            detail = response.json().get("detail", "Unknown error")
            return (
                f"❌ Bad request: {detail}",
                confidence_bar(0),
                "",
                "",
                history_block(state["history"]),
                status_text(state["query_count"]),
                query,
                state,
            )

        return (
            f"❌ Server error ({response.status_code}): {response.text[:300]}",
            confidence_bar(0),
            "",
            "",
            history_block(state["history"]),
            status_text(state["query_count"]),
            query,
            state,
        )

    except httpx.ConnectError:
        return (
            "❌ Backend not running. Start it with: cd finsage ; python main.py",
            confidence_bar(0),
            "",
            "",
            history_block(state["history"]),
            status_text(state["query_count"]),
            query,
            state,
        )
    except httpx.ReadTimeout:
        return (
            "⏰ Request timed out (over 120 seconds). The backend might be overloaded. Try again.",
            confidence_bar(0),
            "",
            "",
            history_block(state["history"]),
            status_text(state["query_count"]),
            query,
            state,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return (
            f"❌ Unexpected error: {str(exc)[:300]}",
            confidence_bar(0),
            "",
            "",
            history_block(state["history"]),
            status_text(state["query_count"]),
            query,
            state,
        )


CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 0.95rem;
    margin-top: 0;
    margin-bottom: 1rem;
}
"""


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="FinSage AI", css=CUSTOM_CSS) as demo:
        state = gr.State(init_state())

        gr.Markdown("## 📈 FinSage AI — Indian Financial Assistant", elem_classes=["main-title"])
        gr.Markdown(
            "Powered by Groq + LangGraph | Free real-time market data | Educational use only",
            elem_classes=["subtitle"],
        )

        gr.Markdown("#### 💡 Example Questions")
        with gr.Row():
            with gr.Column():
                q1 = gr.Button(EXAMPLE_QUERIES[0][0], variant="secondary")
                q2 = gr.Button(EXAMPLE_QUERIES[1][0], variant="secondary")
                q3 = gr.Button(EXAMPLE_QUERIES[2][0], variant="secondary")
            with gr.Column():
                q4 = gr.Button(EXAMPLE_QUERIES[3][0], variant="secondary")
                q5 = gr.Button(EXAMPLE_QUERIES[4][0], variant="secondary")
                q6 = gr.Button(EXAMPLE_QUERIES[5][0], variant="secondary")
            with gr.Column():
                q7 = gr.Button(EXAMPLE_QUERIES[6][0], variant="secondary")
                q8 = gr.Button(EXAMPLE_QUERIES[7][0], variant="secondary")
                q9 = gr.Button(EXAMPLE_QUERIES[8][0], variant="secondary")

        status = gr.Markdown(status_text(0))
        query_input = gr.Textbox(
            label="Ask any financial question...",
            placeholder="e.g., I sold TCS after 8 months with ₹50,000 profit. What is my tax?",
            lines=2,
        )
        analyze_btn = gr.Button("🔍 Analyze", variant="primary")

        answer = gr.Markdown()
        confidence = gr.HTML(confidence_bar(0))
        intent = gr.Markdown()
        trace = gr.HTML()
        history = gr.Markdown()

        examples = [q1, q2, q3, q4, q5, q6, q7, q8, q9]
        for button, (_, query_text) in zip(examples, EXAMPLE_QUERIES):
            button.click(
                fn=lambda text=query_text: text,
                inputs=None,
                outputs=query_input,
                api_name=False,
                queue=False,
            )

        outputs = [answer, confidence, intent, trace, history, status, query_input, state]
        analyze_btn.click(fn=analyze, inputs=[query_input, state], outputs=outputs)
        query_input.submit(fn=analyze, inputs=[query_input, state], outputs=outputs)

        gr.Markdown(
            "⚠️ Disclaimer: FinSage AI provides financial information for educational purposes only. "
            "This is not SEBI-registered investment advice. Always consult a qualified financial advisor "
            "before making investment decisions."
        )

    return demo


demo = create_ui()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("FINSAGE_UI_PORT", "7860")))
