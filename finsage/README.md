# 📈 FinSage AI — Indian Financial Assistant

**FinSage AI** is a multi-agent financial assistant for Indian users. Ask questions in plain English about stocks, market indices, salary planning, or taxes — and get real-time analysis with actionable recommendations.

> ⚠️ **Disclaimer:** This is an educational project. Not SEBI-registered investment advice. Always consult a qualified financial advisor before making investment decisions.

---

## 🎯 What It Does

| Query Type | Example | What Happens |
|---|---|---|
| **Salary Planning** | "My salary is ₹20,000. How should I manage?" | Budget breakdown with PPF, ELSS, SIP recommendations |
| **Stock Analysis** | "Should I buy Reliance now?" | Live price + Technical analysis + News sentiment + Recommendation |
| **Index Analysis** | "Nifty at 22,400 — buy or wait?" | Index data + EMA/RSI/MACD analysis + Trading signal |
| **Tax Calculation** | "Sold TCS after 8 months with ₹50K profit. Tax?" | STCG/LTCG calculation with optimization tips |
| **Market Overview** | "Current Indian market situation?" | News sentiment + Market summary |

---

## 🏗️ Architecture

```
User Query → Intent Agent (classify) → Route to specialist:
  ├── Stock/Index → Market Agent → News Agent → Technical Agent → Synthesis
  ├── Tax → Tax Agent (RAG + reasoning) → Synthesis
  ├── Salary → Salary Agent (RAG + planning) → Synthesis
  └── General → News Agent → Synthesis
```

**7 AI agents** orchestrated by **LangGraph**, powered by **3 Groq models**:
- `llama-3.1-8b-instant` — Fast classification & sentiment (under 200ms)
- `llama-3.3-70b-versatile` — Market analysis, planning, synthesis
- `qwen-qwq-32b` — Step-by-step math reasoning (tax, technical analysis)

---

## 📋 Prerequisites

Before starting, make sure you have:

1. **Python 3.10 or higher** installed
   ```bash
   python --version
   # Should show Python 3.10+ 
   ```

2. **pip** package manager (comes with Python)

3. **A free Groq API key**
   - Go to [https://console.groq.com](https://console.groq.com)
   - Sign up / log in
   - Navigate to **API Keys** → **Create API Key**
   - Copy the key (starts with `gsk_...`)

4. **Internet connection** — needed for:
   - Downloading Python packages
   - Downloading the embedding model (~80MB, first run only)
   - Fetching live market data and news

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Clone or Download the Project

If you have the project as a ZIP, extract it. If cloning:
```bash
git clone <your-repo-url>
cd finsage
```

### Step 2: Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install All Dependencies

```bash
pip install -r requirements.txt
```

This will install ~20 packages including LangGraph, Groq, FastAPI, Streamlit, yfinance, FAISS, and sentence-transformers.

> ⏰ **Note:** First install may take 3-5 minutes depending on your internet speed.

### Step 4: Set Up Your API Key

Create a `.env` file in the `finsage/` directory (or edit the existing one):

```bash
# Windows
echo GROQ_API_KEY=gsk_your_actual_key_here > .env

# Or manually edit .env file with any text editor
```

Replace `gsk_your_actual_key_here` with your actual Groq API key from Step 1 of Prerequisites.

> 🔐 **Security:** Never commit your `.env` file to git. It's already in `.gitignore`.

### Step 5: Build the Knowledge Base (Run Once)

This step downloads the embedding model (~80MB) and creates the FAISS vector index from the tax/investment/SEBI documents:

```bash
python scripts/ingest_docs.py
```

**Expected output:**
```
============================================================
FinSage AI — Knowledge Base Ingestion
============================================================
  📄 investment_rules.txt: 25 chunks
  📄 sebi_basics.txt: 18 chunks
  📄 tax_rules.txt: 22 chunks

  Total chunks: 65
  Embedding chunks using all-MiniLM-L6-v2...
  Embedding shape: (65, 384)
  FAISS index built with 65 vectors

  ✅ Index saved to: rag/faiss.index
  ✅ Chunks saved to: rag/chunks.pkl
============================================================
Knowledge base ready!
```

> ⏰ **First run:** The embedding model downloads ~80MB. Subsequent runs are instant.

### Step 6: Verify Tools Work (Optional but Recommended)

Test that the data tools work independently:

```bash
# Test Yahoo Finance
python -c "from tools.yahoo_tool import get_stock_data; print(get_stock_data('TCS'))"

# Test News fetching
python -c "from tools.news_tool import get_news; print(get_news('Reliance Industries')[:2])"
```

### Step 7: Run the Full Agent Test (Optional)

Test all 5 query paths without starting the server:

```bash
python scripts/test_query.py
```

This will test salary, stock, index, tax, and general queries. Each test takes 15-30 seconds because it calls Groq API and fetches live data.

### Step 8: Start the Backend Server

```bash
python main.py
```

**Expected output:**
```
✅ FinSage AI backend started
📊 API docs: http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to stop)
```

> 💡 **Tip:** Visit [http://localhost:8000/docs](http://localhost:8000/docs) to see the interactive API documentation (Swagger UI).

### Step 9: Start the Frontend (New Terminal)

Open a **new terminal window**, activate the virtual environment again, and run:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
streamlit run frontend/app.py
```

**macOS / Linux:**
```bash
source venv/bin/activate
streamlit run frontend/app.py
```

### Step 10: Open in Browser

Streamlit will automatically open your browser. If not, go to:

👉 **[http://localhost:8501](http://localhost:8501)**

You should see the FinSage AI chat interface with quick-start buttons.

---

## 🧪 Testing the System

### Quick Test via Browser
1. Open [http://localhost:8501](http://localhost:8501)
2. Click any quick-start button (e.g., "💰 Salary ₹25,000")
3. Click "🔍 Analyze"
4. Wait 15-30 seconds for the full analysis

### Test via API (curl / Postman)
```bash
# Health check
curl http://localhost:8000/api/health

# Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I buy Reliance now?", "user_id": "test"}'

# Get history
curl http://localhost:8000/api/history/test
```

---

## 📁 Project Structure

```
finsage/
├── .env                          ← Your Groq API key (DO NOT commit)
├── .env.example                  ← Template (safe to commit)
├── requirements.txt              ← All Python dependencies
├── main.py                       ← FastAPI server entry point
├── README.md                     ← This file
│
├── config/
│   ├── settings.py               ← Loads GROQ_API_KEY from .env
│   └── models.py                 ← 3 Groq model name constants
│
├── agents/
│   ├── state.py                  ← AgentState TypedDict (shared data)
│   ├── graph.py                  ← LangGraph routing & orchestration
│   ├── intent_agent.py           ← Classifies query intent
│   ├── market_agent.py           ← Fetches live market data
│   ├── news_agent.py             ← RSS news + sentiment scoring
│   ├── technical_agent.py        ← EMA, RSI, MACD, support/resistance
│   ├── tax_agent.py              ← Indian tax calculation (RAG + reasoning)
│   ├── salary_agent.py           ← Monthly budget planning
│   └── synthesis_agent.py        ← Final recommendation generator
│
├── tools/
│   ├── nse_tool.py               ← NSE India live price scraper
│   ├── yahoo_tool.py             ← Yahoo Finance fallback + OHLCV
│   ├── news_tool.py              ← Google News + MoneyControl RSS
│   └── technical_tool.py         ← Technical indicator math (ta library)
│
├── rag/
│   ├── embedder.py               ← sentence-transformers embedder
│   ├── knowledge_base.py         ← FAISS search functions
│   ├── ingest.py                 ← Build FAISS index (run once)
│   ├── faiss.index               ← Generated FAISS index (after ingestion)
│   ├── chunks.pkl                ← Generated chunk texts (after ingestion)
│   └── docs/
│       ├── tax_rules.txt         ← Indian tax slabs, STCG, LTCG, 80C
│       ├── investment_rules.txt  ← 50-30-20 rule, SIP, PPF, NPS, ELSS
│       └── sebi_basics.txt       ← SEBI guidelines, disclaimers
│
├── db/
│   ├── database.py               ← SQLite + SQLAlchemy setup
│   ├── models.py                 ← QueryLog table model
│   └── crud.py                   ← Save/retrieve query logs
│
├── api/
│   └── routes.py                 ← /chat, /health, /history endpoints
│
├── frontend/
│   └── app.py                    ← Streamlit chat interface
│
└── scripts/
    ├── ingest_docs.py            ← Build FAISS index from docs
    └── test_query.py             ← Test all 5 query paths
```

---

## 🔧 What Each Query Type Does

### 1. Salary Planning (`intent: salary`)
**Flow:** Intent → Salary Agent → Synthesis

The salary agent:
- Retrieves budgeting rules from RAG (50-30-20 rule, emergency fund, SIP basics)
- Uses `llama-3.3-70b-versatile` to create a detailed monthly plan
- Includes: rent, groceries, emergency fund, SIP, PPF/ELSS, insurance, discretionary

### 2. Stock Analysis (`intent: stock`)
**Flow:** Intent → Market → News → Technical → Synthesis

Full pipeline:
1. **Market Agent:** Fetches live price from NSE (or Yahoo Finance fallback)
2. **News Agent:** Gets 10 news headlines, scores sentiment (-1 to +1)
3. **Technical Agent:** Calculates EMA, RSI, MACD, support/resistance
4. **Synthesis:** Combines everything into BUY/SELL/HOLD recommendation

### 3. Index Analysis (`intent: index`)
**Flow:** Intent → Market → News → Technical → Synthesis

Same as stock analysis but for Nifty 50, BankNifty, or Sensex.

### 4. Tax Calculation (`intent: tax`)
**Flow:** Intent → Tax Agent → Synthesis

The tax agent:
- Retrieves relevant tax rules from FAISS (STCG/LTCG rates, 80C deductions)
- Uses `qwen-qwq-32b` (reasoning model) for step-by-step tax calculation
- Shows math, suggests optimizations, reminds to consult a CA

### 5. General Market Query (`intent: general`)
**Flow:** Intent → News → Synthesis

Light pipeline:
1. **News Agent:** Fetches latest Indian market news and scores sentiment
2. **Synthesis:** Provides market overview based on news

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated: `.\venv\Scripts\Activate.ps1` |
| `GROQ_API_KEY not set` | Check `.env` file exists with valid key |
| `Backend not running` error in Streamlit | Start backend first: `python main.py` |
| `FAISS index not found` | Run `python scripts/ingest_docs.py` |
| NSE data timeout | Normal — system auto-falls back to Yahoo Finance |
| Slow responses (>30s) | Normal for first request. Groq models need warm-up |
| `Port 8000 already in use` | Kill the other process or change port in `main.py` |
| `Port 8501 already in use` | Run `streamlit run frontend/app.py --server.port 8502` |

---

## 🔮 What to Build Next

These improvements are planned for future versions (do not build now):

1. **User Portfolio Storage** — Track holdings, buy price, P&L in SQLite
2. **NSE Options Chain** — Live Nifty options for PCR and max pain
3. **Macro Agent** — RBI repo rate, FII/DII data
4. **Redis Caching** — Cache market data (60s) and news (15min)
5. **Better LLM for Synthesis** — Claude Sonnet for higher quality output
6. **Interactive Charts** — OHLCV + indicator overlays with Plotly
7. **Telegram Alerts** — Price target and stop-loss notifications

---

## 📄 License

This project is for educational purposes only. Use at your own risk.

---

*Built with ❤️ using Groq, LangGraph, FastAPI, and Streamlit*
