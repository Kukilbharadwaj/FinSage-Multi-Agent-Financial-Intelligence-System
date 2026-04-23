# frontend/app.py
# Streamlit chat UI for FinSage AI.

import streamlit as st
import httpx
import time

# ─── Page Config ───
st.set_page_config(
    page_title="FinSage AI",
    page_icon="📈",
    layout="centered",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #00d4aa;
    }
    .trace-step {
        padding: 4px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📈 FinSage AI — Indian Financial Assistant")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Powered by Groq + LangGraph | Free real-time market data | Educational use only</p>',
    unsafe_allow_html=True,
)

# ─── Session State Init ───
if "history" not in st.session_state:
    st.session_state.history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

def set_query(query_text):
    st.session_state.query_input = query_text

# ─── Example Questions Buttons ───
st.markdown("#### 💡 Example Questions:")
col1, col2, col3 = st.columns(3)

with col1:
    st.button("💰 Salary ₹50,000", use_container_width=True, on_click=set_query, args=("My salary is ₹50,000. How to manage it properly?",))
    st.button("📉 What is STCG?", use_container_width=True, on_click=set_query, args=("What is the new STCG and LTCG tax rate after the July budget?",))

with col2:
    st.button("📊 Buy Infosys?", use_container_width=True, on_click=set_query, args=("Should I buy Infosys stock today?",))
    st.button("🛡️ Section 80C", use_container_width=True, on_click=set_query, args=("How can I save tax using Section 80C?",))

with col3:
    st.button("📈 Option Trading", use_container_width=True, on_click=set_query, args=("In option trading, what should I do right now?",))
    st.button("💸 ITC Dividend", use_container_width=True, on_click=set_query, args=("Is ITC a good stock for dividend income?",))

st.divider()

# ─── Input Area ───
limit_reached = st.session_state.query_count >= 5

if limit_reached:
    st.error("🛑 You have reached the maximum limit of 5 queries per session. Please refresh the page to start a new session.")

query = st.text_input(
    "Ask any financial question...",
    placeholder="e.g., I sold TCS after 8 months with ₹50,000 profit. What is my tax?",
    key="query_input",
    disabled=limit_reached
)

# ─── Submit Button ───
if st.button("🔍 Analyze", type="primary", use_container_width=True, disabled=limit_reached):
    if not query or not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("🔄 Fetching market data and analyzing... This may take 30-60 seconds."):
            try:
                start_time = time.time()

                response = httpx.post(
                    "http://localhost:8000/api/chat",
                    json={"user_id": "streamlit_user", "query": query.strip()},
                    timeout=120.0,
                )

                elapsed = round(time.time() - start_time, 1)

                if response.status_code == 200:
                    data = response.json()

                    # Save to session history
                    st.session_state.history.insert(0, {
                        "query": query,
                        "answer": data["answer"],
                        "confidence": data["confidence"],
                        "intent": data["intent"],
                        "trace": data["trace"],
                        "time": elapsed,
                    })
                    # Keep only last 5
                    st.session_state.history = st.session_state.history[:5]
                    
                    st.session_state.query_count += 1

                    # ─── Display Results ───
                    st.success(f"✅ Analysis complete in {elapsed}s")

                    # Answer
                    st.markdown("---")
                    st.markdown(data["answer"])

                    # Confidence bar
                    st.markdown("---")
                    confidence = data.get("confidence", 0) or 0
                    st.markdown(f"**Confidence Score:** {confidence}%")
                    st.progress(confidence / 100)

                    # Intent badge
                    intent = data.get("intent", "general")
                    intent_colors = {
                        "stock": "🟢", "index": "🔵", "tax": "🟠",
                        "salary": "🟣", "general": "⚪",
                    }
                    st.markdown(f"**Intent:** {intent_colors.get(intent, '⚪')} `{intent}`")

                    # Agent trace
                    with st.expander("🔍 See how this was analyzed"):
                        for step in data.get("trace", []):
                            st.markdown(f"- `{step}`")
                        st.markdown(f"- ⏱️ Total time: {elapsed}s")

                elif response.status_code == 400:
                    st.error(f"❌ Bad request: {response.json().get('detail', 'Unknown error')}")
                else:
                    st.error(f"❌ Server error ({response.status_code}): {response.text[:300]}")

            except httpx.ConnectError:
                st.error(
                    "❌ **Backend not running.** Start it with:\n\n"
                    "```bash\ncd finsage\npython main.py\n```"
                )
            except httpx.ReadTimeout:
                st.error(
                    "⏰ **Request timed out** (over 120 seconds). "
                    "The backend might be overloaded. Try again."
                )
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)[:300]}")

# ─── Session History ───
if st.session_state.history:
    st.divider()
    st.markdown("#### 📜 Recent Queries")

    for i, item in enumerate(st.session_state.history):
        with st.expander(f"{'🕐' if i > 0 else '🕛'} {item['query'][:80]}...", expanded=(i == 0)):
            st.markdown(item["answer"][:500] + ("..." if len(item["answer"]) > 500 else ""))
            st.caption(f"Intent: {item['intent']} | Confidence: {item['confidence']}% | Time: {item['time']}s")

# ─── Footer ───
st.divider()
st.caption(
    "⚠️ **Disclaimer:** FinSage AI provides financial information for educational purposes only. "
    "This is not SEBI-registered investment advice. Always consult a qualified financial advisor "
    "before making investment decisions."
)
