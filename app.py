"""Streamlit chat over the airport agent: `uv run streamlit run app.py`.

Each answer shows the data first, rendered by code from the tool envelopes (src/render.py), then
the model's text labeled as commentary, then the raw tool calls. Streamlit reruns this script on
every interaction, so the agent (and its in-memory conversation store) lives in st.cache_resource
and the thread_id and messages live in st.session_state. Tools open SQLite read-only per call.
Voice input (src/voice.py) turns a recording into text that then takes the typed-text path.
"""
from __future__ import annotations

import uuid
from contextlib import closing

import streamlit as st

from src import agent as agent_mod
from src import census, render, store, tools, voice
from src.config import ConfigError, Settings, get_settings, load_scoring
from src.llm import llm_error_message

SAMPLE_QUESTIONS = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long-haul flights out of Anchorage?",
    "What is the unmet flight demand at SFO and why?",
]


@st.cache_resource(show_spinner="Loading the agent...")
def get_agent() -> agent_mod.AirportAgent:
    return agent_mod.build_agent(get_settings())


def data_status(settings: Settings) -> dict:
    with closing(store.connect_ro(settings.db_path)) as conn:
        return tools.data_status(conn, load_scoring(settings.weights_path))


def new_chat() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def ask_sample(question: str) -> None:
    st.session_state.pending = question


def sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.header("Airport Investment Agent")
        try:
            status = data_status(settings)
            st.markdown(f"**Data period:** {status['data_period']}  \n**Freshness:** {status['freshness']}")
        except Exception as exc:  # noqa: BLE001 - the chat shows the actionable error
            st.markdown(f"**Data:** not available ({type(exc).__name__})")
        st.markdown(f"**Model:** {settings.llm_provider} / {settings.llm_model}")
        st.button("New chat", on_click=new_chat, width="stretch")
        st.subheader("Sample questions")
        for q in SAMPLE_QUESTIONS:
            st.button(q, on_click=ask_sample, args=(q,), width="stretch")
        st.caption(census.ATTRIBUTION)


def show_data(tool_calls: list[dict], envelopes: list) -> None:
    st.markdown("##### Data (from tools)")
    if not envelopes:
        st.caption("No tool was called, so this answer has no figures from the data.")
    for i, env in enumerate(envelopes):
        title = tool_calls[i]["name"] if i < len(tool_calls) else None
        if not isinstance(env, dict) or env.get("error"):
            st.markdown(render.render_envelope(env, title))
            continue
        st.markdown("\n\n".join(p for p in (f"**{title}**" if title else "", render.render_headline(env),
                                            render.render_comparison(env), render.render_facts(env),
                                            render.render_why(env), render.render_ranking(env),
                                            render.render_drivers(env)) if p))
        if excluded := render.render_excluded(env):
            with st.expander(render.excluded_title(env)):
                st.markdown(excluded)
        if assumptions := render.render_assumptions(env):
            st.markdown("Assumptions:\n\n" + assumptions)


def show_assistant(msg: dict) -> None:
    if "error" in msg:
        st.error(msg["error"])
        return
    show_data(msg["tool_calls"], msg["tool_results"])
    check = msg.get("figure_check", {})
    if not check.get("hidden"):  # rankings and comparisons: no commentary block when the model wrote nothing
        st.markdown("##### Commentary (model)")
        st.caption("Written by the language model about the data above; the figures come from the data card.")
        st.markdown(msg["commentary"] or "_(the model returned no text)_")
        if note := check.get("note"):
            st.caption(note)
    with st.expander(f"Tools called ({len(msg['tool_calls'])})"):
        st.json([{**c, "result": r} for c, r in zip(msg["tool_calls"], msg["tool_results"])], expanded=False)


def show_user(msg: dict) -> None:
    st.markdown(f"You said: {msg['content']}" if msg.get("voice") else msg["content"])


def voice_input(settings: Settings) -> str | None:
    """Microphone above the chat input; the transcript of a new recording, or None.

    Each recording is read once: the widget key changes after use, so the next run shows an empty
    microphone and Streamlit drops the old widget and its audio from the session state."""
    if reason := voice.unavailable_reason(settings):
        st.caption(reason)
        return None
    n = st.session_state.setdefault("voice_n", 0)
    recording = st.audio_input("Or ask by voice", key=f"voice_{n}")
    if recording is None:
        return None
    st.session_state.voice_n = n + 1
    try:
        with st.spinner("Transcribing..."):
            return voice.transcribe(settings, recording.getvalue())
    except voice.VoiceError as exc:
        st.error(str(exc))
        return None


def answer(agent: agent_mod.AirportAgent, settings: Settings, question: str) -> dict:
    try:
        reply = agent.ask(question, st.session_state.thread_id)
    except agent_mod.InputRejected as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - e.g. the model server is not running
        return {"error": llm_error_message(settings, exc)}
    return {"tool_calls": reply.tool_calls, "tool_results": reply.tool_results, "commentary": reply.commentary,
            "figure_check": reply.figure_check}


def main() -> None:
    st.set_page_config(page_title="Airport Investment Agent", layout="wide")
    if "thread_id" not in st.session_state:
        new_chat()
    try:
        settings = get_settings()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    sidebar(settings)
    try:
        agent = get_agent()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()

    st.title("Airport Investment Agent")
    st.caption("Figures and rankings come from the data tools; the model only adds commentary.")
    chat = st.container()  # the conversation, kept above the microphone and the chat input
    with chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    show_assistant(msg)
                else:
                    show_user(msg)

    spoken = voice_input(settings)
    typed = st.chat_input("Ask about US airports")
    question = typed or spoken or st.session_state.pop("pending", None)
    if not question:
        return
    user_msg = {"role": "user", "content": question, "voice": bool(spoken) and not typed}
    st.session_state.messages.append(user_msg)
    with chat:
        with st.chat_message("user"):
            show_user(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                msg = {"role": "assistant", **answer(agent, settings, question)}
            show_assistant(msg)
    st.session_state.messages.append(msg)


main()
