# 🎬 Demo Guide — Smart City AI Agent

Use this script when showcasing the project in interviews, portfolio reviews, or video demos.

---

## Setup (Before the Demo)

1. Start both servers:
   ```
   # Terminal 1
   uvicorn app.main:app --reload

   # Terminal 2
   streamlit run frontend/app.py
   ```
2. Clear any old sessions: click "New Conversation" in the UI
3. Have the GitHub repo open in another tab
4. Have `http://localhost:8000/docs` open for the API docs view

---

## Demo Flow (5 minutes)

### 1. First Impression (30s)

Open the dashboard. Point out:
- Split-view layout: chat + insights
- Live traffic map with TomTom tiles (toggle on)
- "This is an autonomous AI agent for London city data"

### 2. Simple Query (45s)

Type: **"How's the tube today?"**

Watch the live reasoning chain appear:
- "See how the agent decides which tools to call..."
- "It selected only the tube status tool — minimum necessary"
- "The response uses real-time TfL data"

### 3. Multi-Source Correlation (60s)

Type: **"Full London overview — traffic, tube, weather, air quality"**

This is the showstopper. Point out:
- Reasoning chain shows 6 steps
- Multiple APIs called in parallel
- Health gauge populates with scores
- Anomaly alerts appear (if any)
- Correlation cards show detected patterns
- "The agent found that [rain/congestion/etc.] is correlated with [weather/incidents/etc.]"

### 4. Geocoding (45s)

Type: **"How's the traffic on Baker Street?"**

Point out:
- "Baker Street isn't one of the predefined points"
- "The agent geocodes it via Nominatim to get real coordinates"
- "Then passes those coordinates to TomTom for actual traffic data"

### 5. Follow-Up with Memory (30s)

Type: **"What about the weather there?"**

Point out:
- "I said 'there' — the agent remembers I was asking about Baker Street"
- "Session memory carries context across questions"

### 6. Architecture Overview (60s)

Switch to the GitHub README or API docs. Walk through:
- "7-node LangGraph state machine"
- "4 real-time APIs, all free tier"
- "Correlation engine runs before the LLM — gives it pre-computed insights"
- "250+ tests — unit, integration, and flow"
- "Response caching prevents redundant API calls"

### 7. Code Quality (30s)

Quickly show in the repo:
- `app/agent/graph.py` — the LangGraph definition
- `app/agent/correlation.py` — the correlation engine
- `tests/` — the test suite
- "Clean separation: data layer, intelligence layer, presentation layer"

---

## Key Talking Points

### If Asked "Why LangGraph?"
"LangGraph gives explicit control over the agent's reasoning flow — I can define exactly when tools run in parallel, add conditional routing for simple queries, and stream node-by-node for the UI. AgentExecutor is a black-box loop; LangGraph is a directed graph I designed."

### If Asked "What's the correlation engine?"
"It runs before the LLM sees the data. It extracts structured metrics from each API response — speeds, temperatures, PM2.5 values — then checks for known patterns: rain + congestion, low wind + poor air quality, tube disruptions pushing commuters onto roads. The LLM then builds on these pre-computed insights instead of discovering patterns from scratch."

### If Asked "How does it handle errors?"
"Three levels: (1) HTTP retry with backoff in the base tool class, (2) per-tool error isolation — if one API fails, others still run, (3) LLM fallback — if Gemini is down, the raw tool data and correlation insights are returned directly."

### If Asked "What would you add next?"
- Historical data storage (PostgreSQL) for trend analysis over time
- WebSocket streaming for real-time UI updates as each node completes
- More data sources: Citymapper, National Rail, flood warnings
- Deploy to cloud with proper auth and rate limiting

---

## Quick Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_correlation.py -v

# Test a single live query (uses 2 Gemini calls)
python -m scripts.test_agent_live

# Check API health
curl http://localhost:8000/health

# Check cache stats
curl http://localhost:8000/api/cache/stats
```
