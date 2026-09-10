"""Gradio chat GUI for the ReAct agent defined in main.py.

Run with: python gradio_app.py

This is a thin UI layer — all the actual agent logic (tools, system prompt,
the Thought/Action/Observation loop) lives in main.py and is reused as-is via
run_react_turn(), so the CLI (main.py) and this GUI always run the exact same
agent, not two diverging copies of it.
"""

import math
import os
import re
import sys
import threading
import time

import gradio as gr
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Backend loading: importing main.py embeds ~106k severe injury narratives and
# two regulatory PDFs on a cold cache (or loads them from disk on a warm one),
# reporting progress via plain print() calls. That import is kicked off here
# on a background thread — with its stdout captured line-by-line instead of
# going to the terminal — so the Gradio page can show a live loading log
# instead of the terminal owning startup and the GUI just appearing frozen.
# ---------------------------------------------------------------------------

_backend_log_lines: list[str] = []
_backend_log_lock = threading.Lock()
_backend_ready = threading.Event()
_backend_error: list[Exception] = []
_main = None  # set to the imported main module once loading finishes


class _LineCapture:
    """File-like object that splits writes into lines and appends each to
    _backend_log_lines — used as sys.stdout while main.py's import runs, so
    its print() progress lands in the GUI's loading log."""

    def __init__(self):
        self._buffer = ""

    def write(self, text: str) -> None:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            with _backend_log_lock:
                _backend_log_lines.append(line)

    def flush(self) -> None:
        pass


def _load_backend() -> None:
    global _main
    real_stdout = sys.stdout
    sys.stdout = _LineCapture()
    try:
        import main as main_module

        # Normally built lazily on the first relevant tool call — warmed here
        # instead so its own print progress lands in this same loading log
        # rather than surfacing mid-conversation on a random later turn.
        main_module._load_power_plant_locations()
        _main = main_module
    except Exception as exc:  # surfaced in the GUI's loading log, not a crash
        _backend_error.append(exc)
    finally:
        sys.stdout = real_stdout
        _backend_ready.set()


def _backend_log_text() -> str:
    with _backend_log_lock:
        lines = list(_backend_log_lines)
    return "\n".join(lines) if lines else "Starting up..."


# Matches embedding_cache.py's per-batch progress print, e.g.
# "Embedding progress: 44500/106001 (42%) — 210s elapsed, ~290s remaining"
_PROGRESS_LINE_RE = re.compile(r"^Embedding progress: ([\d,]+)/([\d,]+) \((\d+)%\) — (.+)$")
# Matches the print emitted once per dataset when its cache is missing, e.g.
# "No cached embeddings found for severe injury narratives — embedding 106001 chunks ..."
_START_EMBEDDING_RE = re.compile(r"^No cached embeddings found for (.+?) —")


def _current_progress_status() -> str:
    """Scan the captured startup log for the most recent embedding-progress
    line and which dataset it belongs to, so the loading header can show a
    live percentage instead of a static message the whole time."""
    with _backend_log_lock:
        lines = list(_backend_log_lines)

    label = None
    progress = None
    for line in lines:
        start_match = _START_EMBEDDING_RE.match(line)
        if start_match:
            label = start_match.group(1)
            progress = None  # a new dataset started embedding — reset

        progress_match = _PROGRESS_LINE_RE.match(line)
        if progress_match:
            progress = progress_match.groups()

    if progress and label:
        done, total, pct, timing = progress
        return f"Embedding {label} — {pct}% ({done}/{total}), {timing}"
    return ""


def await_backend_ready():
    """Generator for demo.load(): streams the loading log to the page until
    the background import finishes, then hides the loading screen and
    reveals the app (or, on failure, leaves the log up with an error). Also
    seeds lc_state with the real conversation once the backend is ready —
    lc_state itself starts out as a plain empty list (see the gr.State
    definition below) rather than a callable, specifically so building the
    UI never blocks on _backend_ready; gr.skip() leaves it untouched on every
    yield here until that final one."""
    base_header = "### ⏳ Starting up — loading models and reference data (first run can take a few minutes)..."
    while not _backend_ready.is_set():
        status = _current_progress_status()
        header = f"### ⏳ {status}" if status else base_header
        yield header, _backend_log_text(), gr.update(visible=True), gr.update(visible=False), gr.skip()
        time.sleep(0.25)

    if _backend_error:
        header = "### ⚠️ Startup failed — see the log below"
        log_text = f"{_backend_log_text()}\n\n{_backend_error[0]!r}"
        yield header, log_text, gr.update(visible=True), gr.update(visible=False), gr.skip()
        return

    yield header, _backend_log_text(), gr.update(visible=False), gr.update(visible=True), _new_conversation()


def quit_app() -> str:
    """Stops the Gradio process shortly after this returns (giving the
    response time to reach the browser first) — the click handler also runs
    JS to best-effort close the tab, though most browsers only allow that for
    a script-opened window, so the status message doubles as the fallback."""

    def _shutdown():
        time.sleep(0.75)
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()
    return "🛑 **Server stopped.** You can close this browser tab now."


# Tools whose result lines describe a power plant in the fixed format PLANT_LINE_RE
# parses below (query_power_plant_database's own filter results, or plants found
# by searching outward from an incident's location).
PLANT_TOOLS = {"query_power_plant_database", "search_power_plants_near_location"}

# Matches a power-plant result line, e.g.:
# [USA0006045] St Lucie | Hutchinson Island South, Florida, ... | Nuclear | 2160.0 MW | lat/long: 27.3486, -80.2464
# — with an optional trailing "| N.N mi away" from search_power_plants_near_location.
PLANT_LINE_RE = re.compile(
    r"^\[(?P<id>[^\]]+)\]\s*(?P<name>[^|]+?)\s*\|\s*"
    r"(?P<location>[^|]+?)\s*\|\s*"
    r"(?P<fuel>[^|]+?)\s*\|\s*"
    r"(?P<capacity>[\d.]+)\s*MW\s*\|.*?"
    r"lat/long:\s*(?P<lat>-?[\d.]+),\s*(?P<lon>-?[\d.]+)"
)

# Tools whose result lines describe a severe injury incident in the fixed format
# INJURY_LINE_RE parses below.
INJURY_TOOLS = {
    "search_severe_injury_lessons_learned",
    "search_severe_injury_near_location",
    "search_severe_injury_by_vendor",
}

# Matches a severe-injury result line, e.g.:
# [2016109551] 10/10/2016 | Duke Energy Florida, LLC | CRYSTAL RIVER, FLORIDA | Gunshot wounds | lat/long: 28.94, -82.59 — ...
# — with an optional leading "N.N mi away |" from search_severe_injury_near_location.
INJURY_LINE_RE = re.compile(
    r"^\[(?P<id>[^\]]+)\]\s*"
    r"(?:(?P<distance>[\d.]+)\s*mi away\s*\|\s*)?"
    r"(?P<date>[^|]+?)\s*\|\s*"
    r"(?P<employer>[^|]+?)\s*\|\s*"
    r"(?P<location>[^|]+?)\s*\|\s*"
    r"(?P<nature>[^|]+?)\s*\|\s*"
    r"lat/long:\s*(?P<lat>-?[\d.]+),\s*(?P<lon>-?[\d.]+)"
)

# A small, fixed taxonomy the raw OSHA "NatureTitle" field (350+ free-text
# variants) is bucketed into for map coloring — an open-ended category count
# isn't something a legend (or a colorblind-safe palette) can represent, so
# everything not matching one of these folds into "Other" rather than getting
# its own generated color.
SAFETY_CATEGORY_COLORS = {
    "Fractures": "#2a78d6",
    "Amputations": "#eb6834",
    "Burns / Electrical": "#1baf7a",
}
DEFAULT_SAFETY_CATEGORY_COLOR = "#898781"  # "Other" — muted, not a competing hue
SAFETY_CATEGORY_ORDER = [*SAFETY_CATEGORY_COLORS, "Other"]


def _safety_category(nature_title: str) -> str:
    text = (nature_title or "").lower()
    if "amputat" in text or "avulsion" in text or "enucleat" in text:
        return "Amputations"
    if "fracture" in text:
        return "Fractures"
    if "burn" in text or "electr" in text:
        return "Burns / Electrical"
    return "Other"

# Free MapLibre tile style — no API key needed, unlike Mapbox-based traces.
MAP_STYLE = "open-street-map"

FUEL_COLORS = {
    "coal": "#4d4d4d",
    "oil": "#7f7f7f",
    "gas": "#e67e22",
    "nuclear": "#8e44ad",
    "hydro": "#2980b9",
    "wind": "#16a085",
    "solar": "#f1c40f",
    "geothermal": "#c0392b",
    "biomass": "#27ae60",
    "waste": "#95a5a6",
    "storage": "#34495e",
    "cogeneration": "#d35400",
    "petcoke": "#2c3e50",
}
DEFAULT_FUEL_COLOR = "#e74c3c"


def _new_conversation() -> list:
    # Safe to block here: called from clear_conversation() (a button handler,
    # always after startup) and from await_backend_ready()'s final yield
    # (only reached once _backend_ready is already set, so the wait below is
    # immediate). Never pass this as gr.State's callable `value=` directly —
    # Gradio calls that eagerly, synchronously, while the UI is being built
    # (before the backend-loading thread even starts), and blocking there
    # deadlocks startup and delays the local URL right along with it.
    _backend_ready.wait()
    if _main is None:
        return []
    return [SystemMessage(content=_main.REACT_SYSTEM_PROMPT)]


def _extract_plants_from_steps(steps: list[dict]) -> list[dict]:
    """Pull plant details out of any query_power_plant_database or
    search_power_plants_near_location result this turn, by parsing the fixed
    line format those tools always return."""
    plants = []
    for step in steps:
        if step.get("type") != "action" or step.get("tool") not in PLANT_TOOLS:
            continue
        for line in step["observation"].splitlines():
            match = PLANT_LINE_RE.match(line.strip())
            if match:
                plants.append({
                    "id": match.group("id"),
                    "name": match.group("name").strip(),
                    "location": match.group("location").strip(),
                    "fuel": match.group("fuel").strip(),
                    "capacity_mw": float(match.group("capacity")),
                    "lat": float(match.group("lat")),
                    "lon": float(match.group("lon")),
                })
    return plants


def _extract_injuries_from_steps(steps: list[dict]) -> list[dict]:
    """Pull incident details out of any severe-injury location result this
    turn (near-location, vendor lookup, or plain lessons-learned search), by
    parsing the fixed line format those tools always return."""
    injuries = []
    for step in steps:
        if step.get("type") != "action" or step.get("tool") not in INJURY_TOOLS:
            continue
        for line in step["observation"].splitlines():
            match = INJURY_LINE_RE.match(line.strip())
            if match:
                nature = match.group("nature").strip()
                injuries.append({
                    "id": match.group("id"),
                    "date": match.group("date").strip(),
                    "employer": match.group("employer").strip(),
                    "location": match.group("location").strip(),
                    "nature": nature,
                    "category": _safety_category(nature),
                    "lat": float(match.group("lat")),
                    "lon": float(match.group("lon")),
                })
    return injuries


def _marker_size(capacity_mw: float) -> float:
    # Square-root scale so a handful of huge plants (2000+ MW) don't dwarf
    # small ones into invisibility, clamped to a sane pixel range.
    return max(8, min(40, 6 + capacity_mw ** 0.5))


def _center_and_zoom(plants: list[dict]) -> tuple[dict, float]:
    """Compute an explicit center/zoom that frames all plotted points.

    layout.map.fitbounds="locations" only reliably auto-fits on the map's
    very first render — Plotly's tile-based maps treat it as a one-time
    computation, not something re-applied on every Plotly.react() update, so
    it doesn't re-focus when gr.Plot pushes a new figure each turn. Setting
    center/zoom explicitly forces the view every time instead.
    """
    lats = [p["lat"] for p in plants]
    lons = [p["lon"] for p in plants]
    center = {"lat": (min(lats) + max(lats)) / 2, "lon": (min(lons) + max(lons)) / 2}

    if len(plants) == 1:
        return center, 9

    # Web-mercator-ish heuristic: zoom out as the bounding box widens, with
    # some padding so outer points aren't flush against the map's edge.
    span = max(max(lats) - min(lats), max(lons) - min(lons), 0.001) * 1.4
    zoom = math.log2(360 / span)
    return center, max(1.5, min(11, zoom))


def build_map_figure(plants: list[dict]) -> go.Figure:
    fig = go.Figure()

    if not plants:
        fig.update_layout(
            map=dict(style=MAP_STYLE, center=dict(lat=39.8, lon=-98.6), zoom=3),
            title="No power plants referenced yet",
            margin=dict(l=0, r=0, t=30, b=0),
            height=550,
        )
        return fig

    by_fuel: dict[str, list[dict]] = {}
    for plant in plants:
        by_fuel.setdefault(plant["fuel"], []).append(plant)

    # One trace per fuel type: gives a legend, a distinct color per fuel, and
    # lets the user click a legend entry to toggle that fuel type on/off.
    for fuel in sorted(by_fuel):
        group = by_fuel[fuel]
        fig.add_trace(go.Scattermap(
            lat=[p["lat"] for p in group],
            lon=[p["lon"] for p in group],
            mode="markers",
            name=fuel,
            marker=dict(
                size=[_marker_size(p["capacity_mw"]) for p in group],
                color=FUEL_COLORS.get(fuel.lower(), DEFAULT_FUEL_COLOR),
            ),
            text=[
                f"<b>{p['name']}</b><br>{p['location']}<br>"
                f"{p['fuel']} — {p['capacity_mw']:.0f} MW"
                for p in group
            ],
            hoverinfo="text",
        ))

    center, zoom = _center_and_zoom(plants)
    fig.update_layout(
        map=dict(style=MAP_STYLE, center=center, zoom=zoom),
        title=f"{len(plants)} power plant(s) in this response",
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
        legend=dict(title=dict(text="Fuel type")),
    )
    return fig


def build_injury_map_figure(injuries: list[dict]) -> go.Figure:
    fig = go.Figure()

    if not injuries:
        fig.update_layout(
            map=dict(style=MAP_STYLE, center=dict(lat=39.8, lon=-98.6), zoom=3),
            title="No safety incidents referenced yet",
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
        )
        return fig

    by_category: dict[str, list[dict]] = {}
    for injury in injuries:
        by_category.setdefault(injury["category"], []).append(injury)

    # One trace per safety category, in the same fixed order every render so
    # a color never gets reassigned to a different category as new incidents
    # come in — SAFETY_CATEGORY_ORDER puts "Other" last regardless of volume.
    for category in SAFETY_CATEGORY_ORDER:
        group = by_category.get(category)
        if not group:
            continue
        fig.add_trace(go.Scattermap(
            lat=[i["lat"] for i in group],
            lon=[i["lon"] for i in group],
            mode="markers",
            name=category,
            marker=dict(
                size=11,
                color=SAFETY_CATEGORY_COLORS.get(category, DEFAULT_SAFETY_CATEGORY_COLOR),
            ),
            text=[
                f"<b>{i['employer']}</b><br>{i['location']}<br>"
                f"{i['nature']} — {i['date']}"
                for i in group
            ],
            hoverinfo="text",
        ))

    center, zoom = _center_and_zoom(injuries)
    fig.update_layout(
        map=dict(style=MAP_STYLE, center=center, zoom=zoom),
        title=f"{len(injuries)} safety incident(s) in this response",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        legend=dict(title=dict(text="Safety category")),
    )
    return fig


def _extract_usage(steps: list[dict]) -> dict:
    for step in steps:
        if step["type"] == "usage":
            return step
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _format_final_text(text: str) -> str:
    # The system prompt has the model prefix its answer with "Final Answer:"
    # so a human reading the raw ReAct transcript can find it — redundant in
    # a chat bubble that's already clearly the assistant's answer, so drop it.
    stripped = text.strip()
    if stripped.lower().startswith("final answer:"):
        return stripped.split(":", 1)[1].strip()
    return stripped


def format_turn_as_markdown(steps: list[dict]) -> str:
    """Render run_react_turn()'s steps for the Gradio chat: intermediate
    reasoning and tool calls collapsed into a <details> block so the bubble
    isn't clumped together, with the final answer shown prominently below it."""
    reasoning_parts = []
    final_text = None
    warning_text = None

    for step in steps:
        if step["type"] == "thought":
            reasoning_parts.append(step["text"])
        elif step["type"] == "action":
            args_str = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
            reasoning_parts.append(
                f"**Tool call:** `{step['tool']}({args_str})`\n\n"
                f"```text\n{step['observation']}\n```"
            )
        elif step["type"] == "final":
            final_text = step["text"]
        elif step["type"] == "warning":
            warning_text = step["text"]

    parts = []

    if reasoning_parts:
        tool_calls = sum(1 for step in steps if step["type"] == "action")
        label = f"Reasoning ({tool_calls} tool call{'s' if tool_calls != 1 else ''})"
        body = "\n\n---\n\n".join(reasoning_parts)
        parts.append(f"<details>\n<summary>{label}</summary>\n\n{body}\n\n</details>")

    if final_text:
        parts.append(_format_final_text(final_text))

    if warning_text:
        parts.append(f"⚠️ {warning_text}")

    usage = _extract_usage(steps)
    if usage["total_tokens"]:
        parts.append(
            f"<sub>🔢 {usage['total_tokens']:,} tokens "
            f"(in: {usage['input_tokens']:,}, out: {usage['output_tokens']:,})</sub>"
        )

    return "\n\n".join(parts)


def _format_session_tokens(total: int) -> str:
    return f"**Session tokens used:** {total:,}"


def _dedupe_by_id(items: list[dict]) -> list[dict]:
    seen_ids = set()
    deduped = []
    for item in items:
        if item["id"] not in seen_ids:
            deduped.append(item)
            seen_ids.add(item["id"])
    return deduped


def _map_outputs(turn_plants: list, turn_injuries: list):
    """The maps reflect only the latest response, not the whole conversation
    — each turn replaces what's plotted rather than adding to it, so a map
    is only built and shown when this turn actually referenced that kind of
    location; otherwise it (and the panel, if neither did) hides again."""
    plant_update = (
        gr.update(value=build_map_figure(turn_plants), visible=True)
        if turn_plants else gr.update(visible=False)
    )
    injury_update = (
        gr.update(value=build_injury_map_figure(turn_injuries), visible=True)
        if turn_injuries else gr.update(visible=False)
    )
    panel_update = gr.update(visible=bool(turn_plants or turn_injuries))
    return panel_update, plant_update, injury_update


def respond(
    user_message: str,
    display_history: list,
    lc_messages: list,
    session_tokens: int,
):
    if not user_message.strip():
        return (
            display_history, lc_messages, "",
            gr.update(), gr.update(), gr.update(),
            session_tokens, _format_session_tokens(session_tokens),
        )

    lc_messages.append(HumanMessage(content=user_message))

    if _main is None:
        steps = []
        turn_output = "⚠️ **Backend failed to load** — check the startup log for details."
    else:
        try:
            steps = _main.run_react_turn(lc_messages)
            turn_output = format_turn_as_markdown(steps)
        except Exception as exc:  # keep the GUI alive even if a turn errors out
            steps = []
            turn_output = f"⚠️ **Error:** {exc}"

    turn_plants = _dedupe_by_id(_extract_plants_from_steps(steps))
    turn_injuries = _dedupe_by_id(_extract_injuries_from_steps(steps))

    session_tokens += _extract_usage(steps)["total_tokens"]

    display_history = display_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": turn_output},
    ]
    panel_update, plant_update, injury_update = _map_outputs(turn_plants, turn_injuries)
    return (
        display_history, lc_messages, "",
        panel_update, plant_update, injury_update,
        session_tokens, _format_session_tokens(session_tokens),
    )


def clear_conversation():
    panel_update, plant_update, injury_update = _map_outputs([], [])
    return (
        [], _new_conversation(),
        panel_update, plant_update, injury_update,
        0, _format_session_tokens(0),
    )


THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    radius_size="lg",
)

CUSTOM_CSS = """
.gradio-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
}
#header {
    text-align: center;
    padding-bottom: 4px;
}
#header h1 {
    margin-bottom: 2px;
}
#session-tokens {
    text-align: right;
}
#chat-panel, #map-panel {
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
#send-btn {
    border-radius: 999px !important;
}
#quit-btn {
    border-radius: 999px !important;
}
#quit-status {
    text-align: right;
}
#loading-log textarea {
    font-family: ui-monospace, "SF Mono", monospace;
    font-size: 0.85em;
}
"""

EMPTY_CHAT_PLACEHOLDER = (
    "### 👋 Ask me anything\n"
    "Try: *\"List nuclear power plants in Florida\"*, "
    "*\"What safety guidelines cover lockout/tagout?\"*, or "
    "*\"What's the root cause of hand injuries near conveyor belts?\"*"
)

with gr.Blocks(title="Power Plant Safety Research Assistant") as demo:
    with gr.Row(elem_id="header"):
        with gr.Column(scale=8):
            gr.Markdown("# ⚡ Power Plant Safety Research Assistant")
            gr.Markdown(
                "Ask about power plants, nearby severe injury incidents, OSHA safety "
                "guidelines, or request a root cause analysis."
            )
        with gr.Column(scale=1, min_width=120):
            quit_btn = gr.Button("🛑 Quit", size="sm", elem_id="quit-btn")
    quit_status = gr.Markdown(elem_id="quit-status")

    # Shown until the background import of main.py (embedding the injury/PDF
    # vector stores, warming the plant geocoding cache) finishes; see
    # _load_backend()/await_backend_ready() above.
    with gr.Column(visible=True) as loading_col:
        loading_header = gr.Markdown()
        loading_log = gr.Textbox(
            label="Startup log",
            lines=14,
            max_lines=20,
            interactive=False,
            autoscroll=True,
            elem_id="loading-log",
        )

    with gr.Column(visible=False) as app_col:
        # Starts empty rather than gr.State(value=_new_conversation) — see
        # the comment on _new_conversation() for why a callable value here
        # would block startup. await_backend_ready() seeds the real
        # conversation into this state once the backend is actually ready.
        lc_state = gr.State(value=[])
        tokens_state = gr.State(value=0)

        session_tokens_display = gr.Markdown(_format_session_tokens(0), elem_id="session-tokens")

        with gr.Row():
            with gr.Column(scale=3, elem_id="chat-panel"):
                chatbot = gr.Chatbot(
                    height=550,
                    placeholder=EMPTY_CHAT_PLACEHOLDER,
                    show_label=False,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question...",
                        scale=8,
                        show_label=False,
                        container=False,
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary", elem_id="send-btn")

                clear_btn = gr.Button("🗑️ Clear conversation", size="sm")

            # Hidden unless the latest response referenced a plant or incident
            # — the maps track only the most recent turn, not the whole
            # conversation, so this collapses again on a turn that didn't.
            # plant_map/injury_map each independently stay hidden further
            # still, until their own kind of data shows up in that turn.
            with gr.Column(scale=2, elem_id="map-panel", visible=False) as map_panel:
                plant_map = gr.Plot(visible=False, label="🏭 Power plants in this response")
                injury_map = gr.Plot(visible=False, label="🚑 Safety incidents in this response")

        inputs = [msg, chatbot, lc_state, tokens_state]
        outputs = [
            chatbot, lc_state, msg,
            map_panel, plant_map, injury_map, tokens_state, session_tokens_display,
        ]
        clear_outputs = [
            chatbot, lc_state,
            map_panel, plant_map, injury_map, tokens_state, session_tokens_display,
        ]

        submit_btn.click(respond, inputs, outputs)
        msg.submit(respond, inputs, outputs)
        clear_btn.click(clear_conversation, None, clear_outputs)

    demo.load(
        fn=await_backend_ready,
        inputs=None,
        outputs=[loading_header, loading_log, loading_col, app_col, lc_state],
    )
    quit_btn.click(
        fn=quit_app,
        inputs=None,
        outputs=quit_status,
        js="() => { setTimeout(() => window.close(), 300); }",
    )


if __name__ == "__main__":
    # prevent_thread_lock=True makes launch() print "Running on local URL" and
    # return immediately instead of blocking here, so the URL is on screen
    # right away and the user knows to open it and watch the loading log,
    # rather than the terminal looking silent for as long as the embedding
    # job underneath takes. The heavy backend-loading thread is started only
    # after that — safe now that lc_state no longer runs _new_conversation()
    # (which blocks on the backend) while the UI is being built; see the
    # comments on _new_conversation() and lc_state above.
    demo.launch(theme=THEME, css=CUSTOM_CSS, prevent_thread_lock=True)
    threading.Thread(target=_load_backend, daemon=True).start()
    demo.block_thread()
