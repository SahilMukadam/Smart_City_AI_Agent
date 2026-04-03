"""
Smart City AI Agent - Streamlit Dashboard
Split-view: Chat (left) + Map & Insights (right)
Glassmorphism iOS-style design.

Run: streamlit run frontend/app.py
(Make sure FastAPI server is running on port 8000)
"""

import time
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

API_BASE = "http://localhost:8000"
LONDON_CENTER = [51.5074, -0.1278]

# London locations for map markers
LONDON_POINTS = {
    "Central London": [51.5074, -0.1278],
    "City of London": [51.5155, -0.0922],
    "Westminster": [51.4975, -0.1357],
    "Camden": [51.5390, -0.1426],
    "Tower Bridge": [51.5055, -0.0754],
    "King's Cross": [51.5317, -0.1240],
    "Canary Wharf": [51.5054, -0.0235],
    "Shoreditch": [51.5274, -0.0777],
    "Brixton": [51.4613, -0.1156],
    "Hammersmith": [51.4927, -0.2248],
}


# ══════════════════════════════════════════════════════════════════
# Page Config & CSS
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart City AI Agent",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit defaults */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}

    /* ── Glassmorphism Cards ────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        color: #e0e0e0;
    }

    .glass-card-accent {
        background: rgba(99, 102, 241, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        color: #e0e0e0;
    }

    /* ── Header ─────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .main-header .subtitle {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        font-weight: 400;
    }

    /* ── Chat Messages ──────────────────────────────────── */
    .chat-user {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px 16px 4px 16px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #e8e8ff;
        max-width: 85%;
        margin-left: auto;
        text-align: right;
    }

    .chat-agent {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px 16px 16px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #e0e0e0;
        max-width: 90%;
        line-height: 1.6;
    }

    .chat-agent p { margin: 0.3rem 0; }

    /* ── Health Score ───────────────────────────────────── */
    .health-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .health-good { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .health-moderate { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
    .health-poor { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* ── Anomaly Alerts ─────────────────────────────────── */
    .anomaly-critical {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #ef4444;
        border-radius: 0 12px 12px 0;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        color: #fca5a5;
    }

    .anomaly-warning {
        background: rgba(234, 179, 8, 0.1);
        border-left: 3px solid #eab308;
        border-radius: 0 12px 12px 0;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        color: #fde047;
    }

    .anomaly-info {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        color: #93c5fd;
    }

    .anomaly-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.2rem; }
    .anomaly-desc { font-size: 0.8rem; opacity: 0.85; }
    .anomaly-rec { font-size: 0.75rem; opacity: 0.7; font-style: italic; margin-top: 0.3rem; }

    /* ── Source Pills ───────────────────────────────────── */
    .source-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 500;
        margin: 0.15rem;
    }

    .source-ok { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.2); }
    .source-cached { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.2); }
    .source-error { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }

    /* ── Section Headers ────────────────────────────────── */
    .section-header {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1rem 0 0.5rem 0;
    }

    /* ── Input styling ──────────────────────────────────── */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.35) !important;
    }

    /* ── Scrollable chat ────────────────────────────────── */
    .chat-container {
        max-height: 55vh;
        overflow-y: auto;
        padding-right: 0.5rem;
    }

    /* ── Map container ──────────────────────────────────── */
    .map-container {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Streamlit element overrides */
    .stMarkdown { color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "api_connected" not in st.session_state:
    st.session_state.api_connected = False


# ══════════════════════════════════════════════════════════════════
# API Functions
# ══════════════════════════════════════════════════════════════════

def check_api_health() -> dict | None:
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except requests.ConnectionError:
        pass
    return None


def send_message(message: str, session_id: str | None = None) -> dict | None:
    """Send a message to the agent and return the structured response."""
    try:
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        r = requests.post(f"{API_BASE}/api/agent/chat", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        return {"response": f"Error: {r.status_code}", "success": False}
    except requests.ConnectionError:
        return {"response": "Cannot connect to the agent server. Make sure it's running on port 8000.", "success": False}
    except requests.Timeout:
        return {"response": "Request timed out. The agent may be overloaded.", "success": False}


def get_city_insights() -> dict | None:
    """Fetch proactive city insights."""
    try:
        r = requests.get(f"{API_BASE}/api/city/insights", timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════
# UI Components
# ══════════════════════════════════════════════════════════════════

def render_header():
    """Render the main header bar."""
    health = check_api_health()
    st.session_state.api_connected = health is not None

    status_dot = "🟢" if health else "🔴"
    agent_status = "Online" if health and health.get("agent_ready") else "Offline"
    sessions = health.get("active_sessions", 0) if health else 0

    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1>🏙️ Smart City AI Agent</h1>
            <span class="subtitle">Real-time London city intelligence</span>
        </div>
        <div style="text-align: right;">
            <span style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">
                {status_dot} Agent: {agent_status} &nbsp;|&nbsp; Sessions: {sessions}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_health_gauge(health_data: dict | None):
    """Render the city health score gauge using Plotly."""
    if not health_data:
        st.markdown('<div class="glass-card" style="text-align:center; opacity:0.5;">No health data yet</div>', unsafe_allow_html=True)
        return

    overall = health_data.get("overall")
    if overall is None:
        return

    # Determine color
    if overall >= 70:
        color = "#4ade80"
        label = "Good"
    elif overall >= 40:
        color = "#facc15"
        label = "Moderate"
    else:
        color = "#f87171"
        label = "Poor"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall,
        title={"text": "City Health", "font": {"size": 14, "color": "rgba(255,255,255,0.7)"}},
        number={"font": {"size": 36, "color": color}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)", "dtick": 25},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.1)"},
                {"range": [40, 70], "color": "rgba(234,179,8,0.1)"},
                {"range": [70, 100], "color": "rgba(34,197,94,0.1)"},
            ],
        },
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Sub-scores row
    cols = st.columns(4)
    sub_labels = [("🚗", "Traffic"), ("🚇", "Tube"), ("🌤️", "Weather"), ("💨", "Air")]
    sub_keys = ["traffic", "tube", "weather", "air_quality"]

    for col, (emoji, label), key in zip(cols, sub_labels, sub_keys):
        val = health_data.get(key)
        if val is not None:
            css_class = "health-good" if val >= 70 else ("health-moderate" if val >= 40 else "health-poor")
            col.markdown(f'<div style="text-align:center;"><span class="health-badge {css_class}">{emoji} {val}</span><br><span style="font-size:0.7rem;color:rgba(255,255,255,0.5);">{label}</span></div>', unsafe_allow_html=True)


def render_anomalies(anomalies: list[dict]):
    """Render anomaly alert cards."""
    if not anomalies:
        st.markdown('<p class="section-header">✅ No anomalies detected</p>', unsafe_allow_html=True)
        return

    st.markdown(f'<p class="section-header">🔍 Anomalies ({len(anomalies)})</p>', unsafe_allow_html=True)

    for a in anomalies[:5]:  # Show top 5
        level = a.get("level", "info")
        css_class = f"anomaly-{level}"
        emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(level, "❓")
        title = a.get("title", "Unknown")
        desc = a.get("description", "")
        rec = a.get("recommendation", "")

        rec_html = f'<div class="anomaly-rec">→ {rec}</div>' if rec else ""

        st.markdown(f"""
        <div class="{css_class}">
            <div class="anomaly-title">{emoji} {title}</div>
            <div class="anomaly-desc">{desc}</div>
            {rec_html}
        </div>
        """, unsafe_allow_html=True)


def render_sources(sources: list[dict]):
    """Render source pills showing which tools were called."""
    if not sources:
        return

    pills_html = ""
    for s in sources:
        name = s.get("tool_name", "unknown").replace("get_", "").replace("_", " ").title()
        if s.get("cached"):
            pills_html += f'<span class="source-pill source-cached">📦 {name}</span>'
        elif s.get("success"):
            ms = s.get("response_time_ms", 0)
            pills_html += f'<span class="source-pill source-ok">✅ {name} ({ms:.0f}ms)</span>'
        else:
            pills_html += f'<span class="source-pill source-error">❌ {name}</span>'

    st.markdown(f'<div style="margin: 0.5rem 0;">{pills_html}</div>', unsafe_allow_html=True)


def render_map():
    """Render London map with monitoring points."""
    m = folium.Map(
        location=LONDON_CENTER,
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    for name, coords in LONDON_POINTS.items():
        folium.CircleMarker(
            location=coords,
            radius=6,
            color="#6366f1",
            fill=True,
            fill_color="#6366f1",
            fill_opacity=0.6,
            popup=folium.Popup(name, max_width=150),
            tooltip=name,
        ).add_to(m)

    st_folium(m, height=280, use_container_width=True, returned_objects=[])


def render_chat_message(role: str, content: str):
    """Render a single chat message."""
    if role == "user":
        st.markdown(f'<div class="chat-user">💬 {content}</div>', unsafe_allow_html=True)
    else:
        # Clean up markdown-like formatting for display
        display = content.replace("*", "").replace("📊", "📊 ")
        st.markdown(f'<div class="chat-agent">{display}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Main Layout
# ══════════════════════════════════════════════════════════════════

render_header()

# Split layout: Chat (left) | Insights (right)
left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: Chat Panel ──────────────────────────────────────────────
with left_col:
    st.markdown('<p class="section-header">💬 Chat with the Agent</p>', unsafe_allow_html=True)

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div class="glass-card-accent" style="text-align: center;">
                <p style="font-size: 1rem; margin-bottom: 0.5rem;">👋 Welcome!</p>
                <p style="font-size: 0.85rem; opacity: 0.8;">
                    Ask me about London's traffic, tube status, weather, or air quality.<br>
                    Try: <em>"Why is traffic bad in Central London?"</em>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                render_chat_message(msg["role"], msg["content"])

    # Source pills for last response
    if st.session_state.last_response:
        sources = st.session_state.last_response.get("sources", [])
        render_sources(sources)

        # Timing info
        total_time = st.session_state.last_response.get("total_time_ms", 0)
        if total_time > 0:
            st.markdown(
                f'<p style="font-size:0.7rem; color:rgba(255,255,255,0.35); margin-top:0.3rem;">⏱️ {total_time/1000:.1f}s total</p>',
                unsafe_allow_html=True,
            )

    # Chat input
    st.markdown("---")

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Ask about London",
            placeholder="e.g., How's the traffic near Canary Wharf?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_btn:
        send_clicked = st.button("Send", use_container_width=True, type="primary")

    # Quick query buttons
    st.markdown('<p class="section-header" style="margin-top:0.5rem;">Quick Queries</p>', unsafe_allow_html=True)
    qc1, qc2 = st.columns(2)
    with qc1:
        if st.button("🚇 Tube Status", use_container_width=True):
            user_input = "How's the tube today?"
            send_clicked = True
        if st.button("🌧️ Weather", use_container_width=True):
            user_input = "What's the weather like in London?"
            send_clicked = True
    with qc2:
        if st.button("🚗 Traffic Overview", use_container_width=True):
            user_input = "Give me a London traffic overview"
            send_clicked = True
        if st.button("🏙️ Full Report", use_container_width=True):
            user_input = "Full London city overview — traffic, tube, weather, and air quality"
            send_clicked = True

    # Handle send
    if send_clicked and user_input and st.session_state.api_connected:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Agent is analyzing..."):
            response = send_message(user_input, st.session_state.session_id)

        if response:
            agent_text = response.get("response", "No response received.")
            st.session_state.messages.append({"role": "assistant", "content": agent_text})
            st.session_state.session_id = response.get("session_id")
            st.session_state.last_response = response

        st.rerun()

    elif send_clicked and not st.session_state.api_connected:
        st.error("⚠️ Agent server not connected. Run: `uvicorn app.main:app --reload`")


# ── RIGHT: Insights Panel ────────────────────────────────────────
with right_col:
    # Health Score Gauge
    st.markdown('<p class="section-header">🏥 City Health Score</p>', unsafe_allow_html=True)

    health_data = None
    if st.session_state.last_response:
        health_data = st.session_state.last_response.get("health")

    render_health_gauge(health_data)

    # Anomaly Alerts
    anomalies = []
    if st.session_state.last_response:
        anomalies = st.session_state.last_response.get("anomalies", [])

    render_anomalies(anomalies)

    # Correlation Insights
    insights = []
    if st.session_state.last_response:
        insights = st.session_state.last_response.get("insights", [])

    if insights:
        st.markdown(f'<p class="section-header">🔗 Correlations ({len(insights)})</p>', unsafe_allow_html=True)
        for ins in insights[:4]:
            conf = ins.get("confidence", "medium")
            emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
            st.markdown(f"""
            <div class="glass-card" style="padding: 0.8rem;">
                <div style="font-size:0.85rem; font-weight:500;">{emoji} {ins.get('title', '')}</div>
                <div style="font-size:0.75rem; opacity:0.7; margin-top:0.2rem;">{ins.get('description', '')[:150]}</div>
            </div>
            """, unsafe_allow_html=True)

    # London Map
    st.markdown('<p class="section-header">🗺️ London Monitoring Points</p>', unsafe_allow_html=True)
    with st.container():
        render_map()

    # Session info
    if st.session_state.session_id:
        st.markdown(
            f'<p style="font-size:0.65rem; color:rgba(255,255,255,0.25); margin-top:1rem; text-align:center;">'
            f'Session: {st.session_state.session_id}</p>',
            unsafe_allow_html=True,
        )

    # New session button
    st.markdown("")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.last_response = None
        st.rerun()
