import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from env.env_logic import CampusAssistantEnv, TASK_REGISTRY
from env.models import ActionType, CampusAction
from agent.baseline_agent import _rule_based_policy

st.set_page_config(page_title="Campus AI Assistant", page_icon="🎓", layout="wide")

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, body { font-family: 'Inter', sans-serif; }

/* background */
.stApp { background: #0f1117; color: #e2e8f0; }

/* hide streamlit default top-bar padding */
[data-testid="stAppViewBlockContainer"] { padding-top: 1rem !important; }

/* hero */
.hero {
    background: linear-gradient(120deg,#4f46e5,#7c3aed 55%,#0891b2);
    border-radius:16px; padding:28px 36px; margin-bottom:20px;
    box-shadow:0 8px 30px rgba(79,70,229,.35);
}
.hero h1 { margin:0 0 6px; font-size:1.9rem; font-weight:700; color:#fff; }
.hero p  { margin:0; color:rgba(255,255,255,.8); font-size:.95rem; }

/* card */
.card {
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09);
    border-radius:14px; padding:18px 22px; margin-bottom:14px;
}

/* step items */
.step-ok  { color:#34d399; font-size:.88rem; display:block; padding:2px 0; }
.step-todo{ color:#475569; font-size:.88rem; display:block; padding:2px 0; }

/* metric boxes */
.mrow { display:flex; gap:10px; margin-bottom:14px; }
.mbox {
    flex:1; text-align:center; background:rgba(99,102,241,.12);
    border:1px solid rgba(99,102,241,.25); border-radius:12px; padding:14px 10px;
}
.mbox .v { font-size:1.5rem; font-weight:700; color:#818cf8; }
.mbox .l { font-size:.68rem; color:#64748b; text-transform:uppercase; letter-spacing:.06em; margin-top:3px; }

/* progress bar */
.pbar-wrap { background:rgba(255,255,255,.07); border-radius:999px; height:9px; overflow:hidden; margin:6px 0 12px; }
.pbar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#6366f1,#06b6d4); }

/* transcript rows */
.tx { background:rgba(99,102,241,.07); border-left:3px solid #6366f1;
      border-radius:8px; padding:9px 14px; margin-bottom:9px; font-size:.83rem; }
.tx .ta { color:#a5b4fc; font-weight:600; }
.tx .tr { float:right; color:#34d399; font-weight:600; }
.tx .tm { display:block; color:#64748b; margin-top:2px; }

/* content box */
.cbox {
    background:rgba(0,0,0,.3); border:1px solid rgba(255,255,255,.07);
    border-radius:10px; padding:14px 16px; font-size:.82rem;
    white-space:pre-wrap; line-height:1.75; color:#cbd5e1;
    max-height:270px; overflow-y:auto;
}

/* badge */
.b-easy   {background:rgba(16,185,129,.18);color:#10b981;border:1px solid rgba(16,185,129,.4);border-radius:999px;padding:2px 12px;font-size:.76rem;font-weight:600;}
.b-medium {background:rgba(245,158,11,.18);color:#f59e0b;border:1px solid rgba(245,158,11,.4);border-radius:999px;padding:2px 12px;font-size:.76rem;font-weight:600;}
.b-hard   {background:rgba(239,68,68,.18);color:#ef4444;border:1px solid rgba(239,68,68,.4);border-radius:999px;padding:2px 12px;font-size:.76rem;font-weight:600;}

/* buttons */
.stButton>button {
    background:linear-gradient(135deg,#6366f1,#7c3aed)!important;
    color:#fff!important; border:none!important; border-radius:10px!important;
    font-weight:600!important; font-size:.88rem!important;
    transition:all .2s!important;
}
.stButton>button:hover { transform:translateY(-1px)!important; box-shadow:0 5px 18px rgba(99,102,241,.45)!important; }
.stButton>button:disabled { background:rgba(255,255,255,.08)!important; color:#475569!important; transform:none!important; }

/* sidebar */
[data-testid="stSidebar"] { background:#0a0c18!important; border-right:1px solid rgba(255,255,255,.06)!important; }
hr { border-color:rgba(255,255,255,.07)!important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
for k, v in [
    ("env", None), ("difficulty", "easy"), ("episode_active", False),
    ("transcript", []), ("benchmark", None)
]:
    if k not in st.session_state:
        st.session_state[k] = v


def get_env():
    if st.session_state.env is None:
        st.session_state.env = CampusAssistantEnv(seed=42)
    return st.session_state.env


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Campus AI")
    st.markdown("---")

    diff = st.selectbox("Difficulty", ["easy", "medium", "hard"],
                        index=["easy","medium","hard"].index(st.session_state.difficulty))
    if diff != st.session_state.difficulty:
        st.session_state.difficulty = diff
        st.session_state.episode_active = False
        st.session_state.transcript = []
        st.session_state.env = None

    ti = TASK_REGISTRY[diff]
    st.markdown(f"**{ti.name}**")
    st.markdown(f'<span class="b-{diff}">{diff.upper()}</span>', unsafe_allow_html=True)
    st.markdown(f"<small style='color:#64748b'>{ti.description}</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Steps required:**")
    for i, s in enumerate(ti.required_steps, 1):
        st.markdown(f"`{i}.` `{s.value}`")
    st.markdown("---")
    st.caption("OpenEnv v1.0 · CampusAssistantEnv v1.0.0")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎓 Autonomous AI Campus Assistant</h1>
  <p>Search notes · Build study plans · Generate quizzes · Set reminders · Create checklists</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: 2 main cols  ─────────────────────────────────────────────────────
left, right = st.columns([1, 1.4], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT — Controls + State
# ════════════════════════════════════════════════════════════════════════════
with left:

    # ── Start / Reset always at top ──────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        start_clicked = st.button("▶️ Start Episode", use_container_width=True)
    with c2:
        reset_clicked = st.button("🔄 Reset", use_container_width=True)

    if start_clicked:
        env = get_env()
        env.reset(st.session_state.difficulty)
        st.session_state.episode_active = True
        st.session_state.transcript = []
        st.session_state.benchmark = None
        st.rerun()

    if reset_clicked:
        st.session_state.env = None
        st.session_state.episode_active = False
        st.session_state.transcript = []
        st.session_state.benchmark = None
        st.rerun()

    st.markdown("---")

    # ── State panel ──────────────────────────────────────────────────────────
    env = get_env()
    sd  = env.state()

    reward   = sd.get("current_reward", 0.0)
    time_rem = sd.get("time_remaining", 0)
    done     = sd.get("done", False)
    comp     = sd.get("completed_steps", [])
    rem      = sd.get("tasks_remaining", [])
    n_done   = len(comp)
    n_total  = n_done + len(rem)

    if not st.session_state.episode_active:
        st.info("👆 Select a difficulty, then click **▶️ Start Episode**.")
    else:
        # metrics
        pct = int(reward * 100)
        st.markdown(f"""
<div class="mrow">
  <div class="mbox"><div class="v">{reward:.2f}</div><div class="l">Reward</div></div>
  <div class="mbox"><div class="v">{n_done}/{n_total}</div><div class="l">Steps</div></div>
  <div class="mbox"><div class="v">{time_rem}s</div><div class="l">Time Left</div></div>
</div>
<div class="pbar-wrap"><div class="pbar-fill" style="width:{pct}%"></div></div>
<small style="color:#64748b">Reward: <b style="color:#818cf8">{pct}%</b></small>
""", unsafe_allow_html=True)

        # task progress
        if n_total > 0:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📋 Task Progress**")
            for s in comp:
                st.markdown(f'<span class="step-ok">✅ {s}</span>', unsafe_allow_html=True)
            for s in rem:
                st.markdown(f'<span class="step-todo">⬜ {s}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # goal
        goal = sd.get("user_goal","")
        if goal:
            st.markdown(f"<small style='color:#475569'>🎯 {goal}</small>", unsafe_allow_html=True)

        # last message
        msg = sd.get("last_message","")
        if msg:
            st.markdown(f"> {msg}")

        # done banner
        if done:
            if reward >= 0.85:
                st.success(f"🎉 Complete! Reward: **{reward:.4f}**")
            elif reward >= 0.5:
                st.warning(f"✅ Finished. Reward: **{reward:.4f}**")
            else:
                st.error(f"⚠️ Ended. Reward: **{reward:.4f}**")

    st.markdown("---")

    # ── Action selector ───────────────────────────────────────────────────────
    st.markdown("### ⚡ Take Action")

    can_act = st.session_state.episode_active and not done
    action_options = [a.value for a in ActionType]

    sel = st.selectbox("Choose action", action_options, disabled=not can_act)

    # parameters
    params: dict = {}
    with st.expander("⚙️ Parameters", expanded=False):
        if sel in ("search_notes", "summarize_notes", "create_checklist"):
            params["topic"] = st.text_input("Topic", value="DBMS")
        elif sel == "create_quiz":
            params["topic"] = st.text_input("Topic", value="DBMS & SQL")
            params["num_questions"] = st.number_input("Questions", 1, 7, 5)
        elif sel == "create_study_plan":
            params["subject"] = st.text_input("Subject", value="DBMS & SQL")
            params["hours"]   = st.number_input("Hours", 1, 8, 2)
        elif sel == "generate_reminder":
            params["event"] = st.text_input("Event", value="SQL Viva")
            params["time"]  = st.text_input("Time",  value="8:00 AM")
        else:
            st.caption("No parameters needed.")

    if st.button("➤ Execute Action", use_container_width=True, disabled=not can_act):
        try:
            env = get_env()
            action = CampusAction(action_type=ActionType(sel), parameters=params)
            obs, rw, dn = env.step(action)
            st.session_state.transcript.append({
                "step":    len(st.session_state.transcript) + 1,
                "action":  sel,
                "reward":  round(rw, 4),
                "total":   round(obs.current_reward, 4),
                "message": obs.last_message,
            })
        except Exception as e:
            st.error(f"Error: {e}")
        st.rerun()

    st.markdown("---")

    # ── Auto Benchmark ────────────────────────────────────────────────────────
    st.markdown("### 🤖 Auto Benchmark")
    st.caption("Runs AI agent on all 3 tasks and shows scores.")

    if st.button("🚀 Run Benchmark", use_container_width=True):
        results = {}
        with st.spinner("Running agent on Easy → Medium → Hard…"):
            for d in ["easy", "medium", "hard"]:
                try:
                    e2 = CampusAssistantEnv(seed=42)
                    e2.reset(d)
                    for _ in range(20):
                        s2  = e2.state()
                        act = _rule_based_policy(s2)
                        _, _, dn2 = e2.step(act)
                        if dn2:
                            break
                    results[d] = e2.state()["current_reward"]
                except Exception as ex:
                    results[d] = 0.0
                    st.warning(f"{d}: error — {ex}")
        st.session_state.benchmark = results
        st.rerun()

    bm = st.session_state.benchmark
    if bm:
        for d in ["easy", "medium", "hard"]:
            sc = bm.get(d, 0.0)
            col_name = {"easy":"#10b981","medium":"#f59e0b","hard":"#ef4444"}[d]
            filled = int(sc * 12)
            bar = "█"*filled + "░"*(12-filled)
            st.markdown(
                f'<span class="b-{d}">{d.upper()}</span> `{bar}` '
                f'<b style="color:{col_name}">{sc:.4f}</b>',
                unsafe_allow_html=True,
            )
        avg = sum(bm.values()) / len(bm)
        st.markdown(f"**Average: `{avg:.4f}`**")

# ════════════════════════════════════════════════════════════════════════════
# RIGHT — Generated Content + Transcript
# ════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown("### 📄 Generated Content")

    # always get fresh state
    sd2       = get_env().state()
    generated = sd2.get("generated_content", {})
    order     = ["summary","study_plan","quiz","checklist","reminder","raw_notes"]
    keys      = [k for k in order if k in generated]

    if keys:
        tabs = st.tabs([k.replace("_"," ").title() for k in keys])
        for tab, key in zip(tabs, keys):
            with tab:
                raw = generated[key]
                # safely escape HTML
                safe = raw.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                st.markdown(f'<div class="cbox">{safe}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="card" style="text-align:center;padding:40px;">
  <div style="font-size:2.5rem">📂</div>
  <div style="color:#475569;margin-top:10px;">No content yet</div>
  <div style="color:#334155;font-size:.82rem;margin-top:4px;">
    Start an episode and execute actions to generate notes, quizzes, study plans and more.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📜 Episode Transcript")

    tx = st.session_state.transcript
    if tx:
        for entry in reversed(tx[-15:]):
            msg_safe = entry["message"].replace("<","&lt;").replace(">","&gt;")
            st.markdown(f"""
<div class="tx">
  <span class="ta">Step {entry['step']} · {entry['action']}</span>
  <span class="tr">+{entry['reward']:.4f} (∑{entry['total']:.4f})</span>
  <span class="tm">{msg_safe}</span>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("<small style='color:#334155'>No steps yet — start an episode and execute actions.</small>",
                    unsafe_allow_html=True)

    with st.expander("🔍 Raw state() JSON"):
        try:
            st.json(get_env().state())
        except Exception as e:
            st.error(str(e))

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#334155;font-size:.78rem'>🎓 Campus AI Assistant · OpenEnv v1.0 · Powered by Streamlit</div>",
            unsafe_allow_html=True)
