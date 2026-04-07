"""
Smart City AI Agent - Streamlit Dashboard (Day 13 - v4)
Fix: Map tile URLs fetched from backend API, not imported from config.

Run: streamlit run frontend/app.py
"""

import re
import time as _time
import requests
import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"
DEFAULT_LAT, DEFAULT_LON = 51.5074, -0.1278

LONDON_POINTS = {
    "Central London": [51.5074, -0.1278], "City of London": [51.5155, -0.0922],
    "Westminster": [51.4975, -0.1357], "Camden": [51.5390, -0.1426],
    "Tower Bridge": [51.5055, -0.0754], "King's Cross": [51.5317, -0.1240],
    "Canary Wharf": [51.5054, -0.0235], "Shoreditch": [51.5274, -0.0777],
    "Brixton": [51.4613, -0.1156], "Hammersmith": [51.4927, -0.2248],
}

st.set_page_config(page_title="Smart City AI Agent", page_icon="🏙️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}

    .glass-card {
        background: rgba(255,255,255,0.05); backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
        padding: 1.2rem; margin-bottom: 0.8rem; color: #e0e0e0;
    }
    .glass-card-accent {
        background: rgba(99,102,241,0.08); backdrop-filter: blur(20px);
        border: 1px solid rgba(99,102,241,0.2); border-radius: 16px;
        padding: 1.2rem; margin-bottom: 0.8rem; color: #e0e0e0;
    }
    .main-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
        backdrop-filter: blur(20px); border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px; padding: 1.2rem 1.8rem; margin-bottom: 1.2rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .main-header h1 { color: #fff; font-size: 1.5rem; font-weight: 700; margin: 0; }
    .main-header .subtitle { color: rgba(255,255,255,0.6); font-size: 0.85rem; }

    .chat-user {
        background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.25);
        border-radius: 16px 16px 4px 16px; padding: 0.8rem 1rem; margin: 0.5rem 0;
        color: #e8e8ff; max-width: 85%; margin-left: auto; text-align: right;
    }
    .chat-agent {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px 16px 16px 4px; padding: 0.8rem 1.2rem; margin: 0.5rem 0;
        color: #e0e0e0; max-width: 92%; line-height: 1.7;
    }
    .chat-agent h2 { color: #a5b4fc; font-size: 1rem; font-weight: 600; margin: 0.8rem 0 0.3rem 0; border-bottom: 1px solid rgba(165,180,252,0.15); padding-bottom: 0.3rem; }
    .chat-agent h3 { color: #c4b5fd; font-size: 0.9rem; font-weight: 600; margin: 0.6rem 0 0.2rem 0; }
    .chat-agent strong { color: #e0e0ff; }
    .chat-agent p { margin: 0.3rem 0; }
    .chat-agent ul, .chat-agent ol { margin: 0.3rem 0; padding-left: 1.2rem; }
    .chat-agent li { margin: 0.15rem 0; }
    .chat-agent hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.6rem 0; }

    .rate-limit-msg {
        background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25);
        border-radius: 16px 16px 16px 4px; padding: 0.8rem 1.2rem; margin: 0.5rem 0;
        color: #fca5a5; max-width: 92%;
    }

    .health-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .health-good { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
    .health-moderate { background: rgba(234,179,8,0.2); color: #facc15; border: 1px solid rgba(234,179,8,0.3); }
    .health-poor { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

    .anomaly-critical { background: rgba(239,68,68,0.1); border-left: 3px solid #ef4444; border-radius: 0 12px 12px 0; padding: 0.7rem 1rem; margin: 0.4rem 0; color: #fca5a5; }
    .anomaly-warning { background: rgba(234,179,8,0.1); border-left: 3px solid #eab308; border-radius: 0 12px 12px 0; padding: 0.7rem 1rem; margin: 0.4rem 0; color: #fde047; }
    .anomaly-info { background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; border-radius: 0 12px 12px 0; padding: 0.7rem 1rem; margin: 0.4rem 0; color: #93c5fd; }
    .anomaly-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.2rem; }
    .anomaly-desc { font-size: 0.8rem; opacity: 0.85; }
    .anomaly-rec { font-size: 0.75rem; opacity: 0.7; font-style: italic; margin-top: 0.3rem; }

    .source-pill { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.7rem; font-weight: 500; margin: 0.15rem; }
    .source-ok { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
    .source-cached { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.2); }
    .source-error { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

    .section-header { color: rgba(255,255,255,0.7); font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin: 1rem 0 0.5rem 0; }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important; color: #e0e0e0 !important;
    }
    .stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.35) !important; }

    .reasoning-container { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 0.8rem; margin: 0.5rem 0; }
    .reasoning-step { display: flex; align-items: flex-start; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
    .reasoning-step:last-child { border-bottom: none; }
    .reasoning-emoji { font-size: 1rem; margin-right: 0.6rem; min-width: 1.4rem; text-align: center; }
    .reasoning-content { flex: 1; }
    .reasoning-label { font-size: 0.8rem; font-weight: 600; color: rgba(255,255,255,0.8); }
    .reasoning-detail { font-size: 0.7rem; color: rgba(255,255,255,0.45); margin-top: 0.1rem; }
    .reasoning-time { font-size: 0.65rem; color: rgba(99,102,241,0.6); margin-left: 0.5rem; }

    .stMarkdown { color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
for key, default in [
    ("messages", []), ("session_id", None), ("last_response", None),
    ("api_connected", False), ("processing", False), ("pending_query", None),
    ("input_key_counter", 0), ("map_config", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── API Functions ─────────────────────────────────────────────────

def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except: return None

def fetch_map_config():
    """Fetch map tile URLs from the backend (keys embedded server-side)."""
    try:
        r = requests.get(f"{API_BASE}/api/config/map", timeout=5)
        if r.status_code == 200:
            return r.json()
    except: pass
    return None

def send_message_api(message, session_id=None):
    try:
        payload = {"message": message}
        if session_id: payload["session_id"] = session_id
        r = requests.post(f"{API_BASE}/api/agent/chat", json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 503:
            return {"response": "Agent not available. Check GEMINI_API_KEY.", "success": False}
        else:
            return {"response": f"Server error ({r.status_code}).", "success": False}
    except requests.ConnectionError:
        return {"response": "Cannot connect to agent server.", "success": False}
    except requests.Timeout:
        return {"response": "Request timed out.", "success": False}

def get_example_queries():
    try:
        r = requests.get(f"{API_BASE}/api/examples", timeout=5)
        return r.json() if r.status_code == 200 else []
    except: return []

def is_rate_limit_error(response_data: dict) -> bool:
    error = response_data.get("error", "") or ""
    response_text = response_data.get("response", "") or ""
    combined = (error + " " + response_text).lower()
    return any(kw in combined for kw in ["429", "rate limit", "quota", "resource exhausted", "too many requests", "resourceexhausted"])


# ── Submit handlers ───────────────────────────────────────────────

def handle_enter_submit():
    input_key = f"chat_input_{st.session_state.input_key_counter}"
    user_text = st.session_state.get(input_key, "").strip()
    if user_text and st.session_state.api_connected:
        st.session_state.pending_query = user_text
        st.session_state.input_key_counter += 1

def handle_button_submit():
    input_key = f"chat_input_{st.session_state.input_key_counter}"
    user_text = st.session_state.get(input_key, "").strip()
    if user_text and st.session_state.api_connected:
        st.session_state.pending_query = user_text
        st.session_state.input_key_counter += 1

def handle_quick_query(query: str):
    if st.session_state.api_connected:
        st.session_state.pending_query = query
        st.session_state.input_key_counter += 1


# ── Markdown to HTML ──────────────────────────────────────────────

def md_to_html(text: str) -> str:
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = text.replace('\n---\n', '\n<hr>\n')
    lines = text.split('\n')
    result = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('• '):
            if not in_list: result.append('<ul>'); in_list = True
            result.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list: result.append('</ul>'); in_list = False
            if stripped == '': result.append('<br>')
            elif not stripped.startswith('<'): result.append(f'<p>{stripped}</p>')
            else: result.append(stripped)
    if in_list: result.append('</ul>')
    return '\n'.join(result)


# ── UI Components ─────────────────────────────────────────────────

def render_header():
    health = check_api_health()
    st.session_state.api_connected = health is not None
    # Fetch map config once when connected
    if st.session_state.api_connected and st.session_state.map_config is None:
        st.session_state.map_config = fetch_map_config()
    status_dot = "🟢" if health else "🔴"
    agent_status = "Online" if health and health.get("agent_ready") else "Offline"
    st.markdown(f"""
    <div class="main-header">
        <div><h1>🏙️ Smart City AI Agent</h1><span class="subtitle">Real-time London city intelligence — any street, any area</span></div>
        <div style="text-align:right;"><span style="color:rgba(255,255,255,0.7);font-size:0.8rem;">{status_dot} {agent_status}</span></div>
    </div>""", unsafe_allow_html=True)

def render_reasoning_chain(steps):
    if not steps: return
    steps_html = ""
    for step in steps:
        time_html = f'<span class="reasoning-time">{step.get("duration_ms",0):.0f}ms</span>' if step.get("duration_ms", 0) > 1 else ""
        detail_html = f'<div class="reasoning-detail">{step.get("detail","")}</div>' if step.get("detail") else ""
        steps_html += f'<div class="reasoning-step"><span class="reasoning-emoji">{step.get("emoji","⚙️")}</span><div class="reasoning-content"><div class="reasoning-label">{step.get("label","")}{time_html}</div>{detail_html}</div></div>'
    st.markdown(f'<div class="reasoning-container">{steps_html}</div>', unsafe_allow_html=True)

def render_health_gauge(health_data):
    if not health_data:
        st.markdown('<div class="glass-card" style="text-align:center;opacity:0.5;">No health data yet — ask a question!</div>', unsafe_allow_html=True)
        return
    overall = health_data.get("overall")
    if overall is None: return
    color = "#4ade80" if overall >= 70 else ("#facc15" if overall >= 40 else "#f87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=overall,
        title={"text": "City Health", "font": {"size": 14, "color": "rgba(255,255,255,0.7)"}},
        number={"font": {"size": 36, "color": color}, "suffix": "/100"},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
               "bar": {"color": color, "thickness": 0.3}, "bgcolor": "rgba(255,255,255,0.05)", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(239,68,68,0.1)"}, {"range": [40, 70], "color": "rgba(234,179,8,0.1)"}, {"range": [70, 100], "color": "rgba(34,197,94,0.1)"}]},
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.7)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    cols = st.columns(4)
    for col, (emoji, label, key) in zip(cols, [("🚗","Traffic","traffic"),("🚇","Tube","tube"),("🌤️","Weather","weather"),("💨","Air","air_quality")]):
        val = health_data.get(key)
        if val is not None:
            css = "health-good" if val >= 70 else ("health-moderate" if val >= 40 else "health-poor")
            col.markdown(f'<div style="text-align:center;"><span class="health-badge {css}">{emoji} {val}</span><br><span style="font-size:0.7rem;color:rgba(255,255,255,0.5);">{label}</span></div>', unsafe_allow_html=True)

def render_anomalies(anomalies):
    if not anomalies:
        st.markdown('<p class="section-header">✅ No anomalies detected</p>', unsafe_allow_html=True)
        return
    st.markdown(f'<p class="section-header">🔍 Anomalies ({len(anomalies)})</p>', unsafe_allow_html=True)
    for a in anomalies[:5]:
        level = a.get("level", "info")
        emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(level, "❓")
        rec_html = f'<div class="anomaly-rec">→ {a.get("recommendation","")}</div>' if a.get("recommendation") else ""
        st.markdown(f'<div class="anomaly-{level}"><div class="anomaly-title">{emoji} {a.get("title","")}</div><div class="anomaly-desc">{a.get("description","")}</div>{rec_html}</div>', unsafe_allow_html=True)

def render_sources(sources):
    if not sources: return
    pills = ""
    for s in sources:
        name = s.get("tool_name", "").replace("get_", "").replace("_", " ").title()
        if s.get("cached"): pills += f'<span class="source-pill source-cached">📦 {name}</span>'
        elif s.get("success"): pills += f'<span class="source-pill source-ok">✅ {name} ({s.get("response_time_ms",0):.0f}ms)</span>'
        else: pills += f'<span class="source-pill source-error">❌ {name}</span>'
    st.markdown(f'<div style="margin:0.5rem 0;">{pills}</div>', unsafe_allow_html=True)

def render_map(show_traffic: bool = True):
    """Render London map with TomTom traffic tiles fetched from backend."""
    map_config = st.session_state.map_config
    traffic_url = map_config.get("traffic_flow_url") if map_config else None
    incidents_url = map_config.get("traffic_incidents_url") if map_config else None
    has_traffic = show_traffic and traffic_url

    if has_traffic:
        # Light basemap so traffic colors (red/yellow/green) are clearly visible
        m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=13, tiles="CartoDB positron")
        folium.TileLayer("CartoDB dark_matter", name="🌙 Dark Map").add_to(m)

        folium.TileLayer(
            tiles=traffic_url,
            attr="Traffic Flow © TomTom",
            name="🚗 Traffic Flow",
            overlay=True,
            control=True,
            opacity=0.85,
        ).add_to(m)

        if incidents_url:
            folium.TileLayer(
                tiles=incidents_url,
                attr="Incidents © TomTom",
                name="⚠️ Incidents",
                overlay=True,
                control=True,
                opacity=0.9,
            ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
    else:
        m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=12, tiles="CartoDB dark_matter")

    for name, coords in LONDON_POINTS.items():
        folium.CircleMarker(
            location=coords, radius=5, color="#6366f1", fill=True,
            fill_color="#6366f1", fill_opacity=0.7, popup=name, tooltip=name,
        ).add_to(m)

    Fullscreen().add_to(m)
    st_folium(m, height=450, use_container_width=True, returned_objects=[])

def render_rate_limit_message():
    st.markdown("""
    <div class="rate-limit-msg">
        <strong>⚠️ API Rate Limit Reached</strong><br>
        The Gemini API daily quota (250 requests) has been exhausted.
        The limit resets at midnight Pacific Time.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Process pending query BEFORE rendering
# ══════════════════════════════════════════════════════════════════

if st.session_state.pending_query and not st.session_state.processing:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.processing = True
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# Main Layout
# ══════════════════════════════════════════════════════════════════

render_header()
left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: Chat ────────────────────────────────────────────────────
with left_col:
    st.markdown('<p class="section-header">💬 Chat with the Agent</p>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="glass-card-accent" style="text-align:center;">
            <p style="font-size:1rem;margin-bottom:0.5rem;">👋 Welcome!</p>
            <p style="font-size:0.85rem;opacity:0.8;">Ask about <strong>any location</strong> in London — streets, areas, landmarks.<br>
            Try: <em>"How's traffic on Baker Street?"</em> or <em>"Air quality near Brick Lane"</em></p>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">💬 {msg["content"]}</div>', unsafe_allow_html=True)
            elif msg.get("is_rate_limit"):
                render_rate_limit_message()
            else:
                st.markdown(f'<div class="chat-agent">{md_to_html(msg["content"])}</div>', unsafe_allow_html=True)

    # Live reasoning chain while processing
    if st.session_state.processing:
        last_user_msg = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break

        if last_user_msg:
            with st.status("🧠 Agent is thinking...", expanded=True) as status:
                st.write("🔀 Analyzing your query...")
                response = send_message_api(last_user_msg, st.session_state.session_id)

                if response and not is_rate_limit_error(response):
                    for step in response.get("reasoning_steps", []):
                        emoji = step.get("emoji", "⚙️")
                        label = step.get("label", "Processing")
                        detail = step.get("detail", "")
                        dur = step.get("duration_ms", 0)
                        time_str = f" ({dur:.0f}ms)" if dur > 1 else ""
                        detail_str = f" — {detail}" if detail else ""
                        st.write(f"{emoji} **{label}**{time_str}{detail_str}")
                        _time.sleep(0.3)

                    total = response.get("total_time_ms", 0)
                    status.update(label=f"✅ Analysis complete ({total/1000:.1f}s)", state="complete", expanded=False)
                    st.session_state.messages.append({"role": "assistant", "content": response.get("response", "No response.")})
                    st.session_state.session_id = response.get("session_id")
                    st.session_state.last_response = response

                elif response and is_rate_limit_error(response):
                    status.update(label="⚠️ Rate limit reached", state="error", expanded=False)
                    st.session_state.messages.append({"role": "assistant", "content": "", "is_rate_limit": True})
                    st.session_state.last_response = None

                else:
                    status.update(label="❌ Request failed", state="error", expanded=False)
                    st.session_state.messages.append({"role": "assistant", "content": response.get("response", "Failed.") if response else "No response."})
                    st.session_state.last_response = None

        st.session_state.processing = False
        st.rerun()

    # Post-response: reasoning chain expander + sources
    if st.session_state.last_response and not st.session_state.processing:
        reasoning = st.session_state.last_response.get("reasoning_steps", [])
        if reasoning:
            with st.expander("🧠 Agent Reasoning Chain", expanded=False):
                render_reasoning_chain(reasoning)
        render_sources(st.session_state.last_response.get("sources", []))
        total_time = st.session_state.last_response.get("total_time_ms", 0)
        if total_time > 0:
            st.markdown(f'<p style="font-size:0.7rem;color:rgba(255,255,255,0.35);">⏱️ {total_time/1000:.1f}s</p>', unsafe_allow_html=True)

    # Input
    st.markdown("---")
    current_input_key = f"chat_input_{st.session_state.input_key_counter}"
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        st.text_input("Ask", placeholder="Any London location — traffic on Brick Lane, weather in Greenwich...", label_visibility="collapsed", key=current_input_key, on_change=handle_enter_submit)
    with col_btn:
        if st.button("Send", use_container_width=True, type="primary"):
            handle_button_submit(); st.rerun()

    # Quick queries
    st.markdown('<p class="section-header" style="margin-top:0.5rem;">Quick Queries</p>', unsafe_allow_html=True)
    qc1, qc2 = st.columns(2)
    with qc1:
        if st.button("🚇 Tube Status", use_container_width=True): handle_quick_query("How's the tube today?"); st.rerun()
        if st.button("🌧️ Weather", use_container_width=True): handle_quick_query("What's the weather like?"); st.rerun()
    with qc2:
        if st.button("🚗 Traffic Overview", use_container_width=True): handle_quick_query("London traffic overview"); st.rerun()
        if st.button("🏙️ Full Report", use_container_width=True): handle_quick_query("Full London overview — traffic, tube, weather, air quality"); st.rerun()

    with st.expander("📚 Example Query Library", expanded=False):
        examples = get_example_queries() if st.session_state.api_connected else []
        if examples:
            for cat in examples:
                st.markdown(f'**{cat.get("category", "")}**')
                for q in cat.get("queries", []):
                    if st.button(f"▸ {q['text']}", key=f"ex_{q['text'][:30]}", use_container_width=True, help=q.get("description", "")):
                        handle_quick_query(q["text"]); st.rerun()
        else:
            st.caption("Connect to agent to load examples")

    if not st.session_state.api_connected:
        st.error("⚠️ Agent not connected. Run: `uvicorn app.main:app --reload`")


# ── RIGHT: Insights ───────────────────────────────────────────────
with right_col:
    st.markdown('<p class="section-header">🗺️ London Live Traffic Map</p>', unsafe_allow_html=True)
    traffic_available = bool(st.session_state.map_config and st.session_state.map_config.get("traffic_flow_url"))
    if traffic_available:
        show_traffic = st.toggle("Show Traffic Flow", value=True, help="Toggle TomTom traffic overlay (green=fast, red=slow)")
    else:
        show_traffic = False
        st.caption("⚠️ Traffic tiles unavailable — check TomTom API key")
    render_map(show_traffic=show_traffic)

    st.markdown('<p class="section-header">🏥 City Health Score</p>', unsafe_allow_html=True)
    render_health_gauge(st.session_state.last_response.get("health") if st.session_state.last_response else None)

    render_anomalies(st.session_state.last_response.get("anomalies", []) if st.session_state.last_response else [])

    insights = st.session_state.last_response.get("insights", []) if st.session_state.last_response else []
    if insights:
        st.markdown(f'<p class="section-header">🔗 Correlations ({len(insights)})</p>', unsafe_allow_html=True)
        for ins in insights[:4]:
            emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(ins.get("confidence", ""), "⚪")
            st.markdown(f'<div class="glass-card" style="padding:0.8rem;"><div style="font-size:0.85rem;font-weight:500;">{emoji} {ins.get("title","")}</div><div style="font-size:0.75rem;opacity:0.7;margin-top:0.2rem;">{ins.get("description","")[:150]}</div></div>', unsafe_allow_html=True)

    if st.session_state.session_id:
        st.markdown(f'<p style="font-size:0.65rem;color:rgba(255,255,255,0.25);margin-top:1rem;text-align:center;">Session: {st.session_state.session_id}</p>', unsafe_allow_html=True)
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []; st.session_state.session_id = None
        st.session_state.last_response = None; st.session_state.processing = False
        st.rerun()
