"""Streamlit UI for the notebook-derived Meeting → Action AI Agent."""
import os
from datetime import date
import streamlit as st

st.set_page_config(page_title="Meeting Action AI", page_icon="🗒️", layout="wide")

# Streamlit Cloud exposes TOML secrets here; local runs can use environment variables.
try:
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "STORAGE_DIR"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = str(st.secrets[key])
except (FileNotFoundError, OSError):
    pass

from backend import agent as backend

st.title("🗒️ Meeting → Action AI Agent")
st.caption("Gemini · LangGraph · Human Approval · SQLite")
st.warning("Demo only: this app has no login or user isolation, and Streamlit Community Cloud local SQLite storage may reset. Don't upload confidential meeting transcripts.")

# The SQL checkpoint connection must be reused across Streamlit reruns in one process.
# The backend module initializes it when imported.
if "workflow_id" not in st.session_state:
    st.session_state.workflow_id = None


def get_config():
    return backend.new_config(st.session_state.workflow_id)


def current_state():
    if not st.session_state.workflow_id:
        return {}, None
    config = get_config()
    snapshot = backend.graph.get_state(config)
    return snapshot.values or {}, backend.get_pending_review(backend.graph, config)


def render_analysis(analysis):
    st.subheader("Meeting Summary")
    st.write(analysis.get("summary", ""))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Decisions**")
        for item in analysis.get("decisions", []):
            st.markdown(f"- {item}")
        if not analysis.get("decisions"):
            st.caption("No confirmed decisions.")
    with c2:
        st.markdown("**Suggestions (not confirmed tasks)**")
        for item in analysis.get("suggestions", []):
            st.markdown(f"- {item}")
        if not analysis.get("suggestions"):
            st.caption("No unapproved suggestions.")


new_tab, saved_tab = st.tabs(["Analyze & Review", "Saved Tasks"])
with new_tab:
    with st.form("meeting_form"):
        title = st.text_input("Meeting title", placeholder="Website Development Meeting")
        meeting_date = st.date_input("Meeting date", value=date.today())
        transcript = st.text_area("Meeting transcript", height=230, placeholder="أحمد: أنا هخلص التصميم بكرة.\nسارة: هراجع الـ Login API يوم الخميس.")
        start = st.form_submit_button("Analyze Meeting", type="primary")

    if start:
        if not title.strip() or not transcript.strip():
            st.error("Please enter a title and transcript.")
        elif len(transcript) > backend.MAX_TRANSCRIPT_CHARS:
            st.error("Transcript exceeds the maximum length.")
        elif not os.environ.get("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY is missing. Configure it in Streamlit Secrets.")
        else:
            new_config = backend.new_config()
            try:
                with st.spinner("Analyzing meeting…"):
                    backend.graph.invoke(
                        backend.initial_state(title, meeting_date.isoformat(), transcript),
                        config=new_config,
                    )
                st.session_state.workflow_id = new_config["configurable"]["thread_id"]
                st.rerun()
            except Exception as exc:
                st.error(f"Analysis failed ({type(exc).__name__}). Check your Gemini key/model and logs.")

    workflow_id = st.session_state.workflow_id
    if workflow_id:
        st.caption(f"Workflow ID: {workflow_id}")
        state, pending = current_state()
        status = state.get("status")
        if state.get("analysis"):
            render_analysis(state["analysis"])

        if pending:
            st.subheader("Human Approval")
            st.info("Approve, reject or skip each task. Event-based and unknown deadlines require attention; blocked tasks cannot be approved. Decisions are not saved until you submit.")
            with st.form("approval_form"):
                decisions = {}
                for item in pending.get("tasks", []):
                    idx = str(item["task_id"])
                    valid = item["validation"]
                    with st.container(border=True):
                        st.markdown(f"**Task {idx}: {item['title']}**")
                        st.write(f"Owner: {item['owner'] or '—'}")
                        st.write(f"Deadline: {item['deadline'] or '—'}  ·  Type: {item['deadline_type'] or 'unknown'}")
                        if item.get("deadline_text"):
                            st.caption(f"Original deadline: {item['deadline_text']}")
                        st.caption(f"Evidence: {item['evidence']}")
                        st.write(f"Validation: **{valid['status']}**")
                        for err in valid.get("errors", []):
                            st.error(err)
                        for warning in valid.get("warnings", []):
                            st.warning(warning)
                        options = ["skip", "reject"] if valid["status"] == "blocked" else ["approve", "reject", "skip"]
                        decisions[idx] = st.radio(
                            f"Decision for task {idx}", options,
                            index=0 if options[0] == "approve" else 0,
                            horizontal=True, key=f"{workflow_id}_task_{idx}"
                        )
                        if valid["status"] == "needs_review":
                            st.caption("Approving a task with missing/event-based deadline is allowed by the notebook's policy; it will be stored without a fabricated date.")
                submit = st.form_submit_button("Submit Review & Save Approved Tasks", type="primary")
            if submit:
                try:
                    with st.spinner("Saving approved tasks…"):
                        backend.graph.invoke(backend.Command(resume=decisions), config=get_config())
                    st.rerun()
                except Exception as exc:
                    st.error(f"Review failed ({type(exc).__name__}). Inspect logs before retrying.")
        elif status == "saved":
            stats = state.get("save_result", {})
            st.success(f"Saved {stats.get('saved', 0)} new task(s); skipped {stats.get('skipped', 0)} duplicate(s).")
            if state.get("saved_meeting_id"):
                st.dataframe(backend.get_tasks(state["saved_meeting_id"]), use_container_width=True, hide_index=True)
        elif status == "review_completed":
            st.info("Review completed. No tasks were approved; nothing was saved.")
        elif status == "security_review_required":
            st.error("The transcript was flagged for possible prompt injection. Processing stopped.")
            st.write("Security flags:", ", ".join(state.get("security_flags", [])))
        elif status == "no_tasks_found":
            st.info("No confirmed tasks were found.")
        elif status == "invalid_review_decision":
            st.error("Invalid review decisions; nothing was saved.")
        elif status:
            st.write("Workflow status:", status)

        if st.button("Clear current workflow (start new)"):
            st.session_state.workflow_id = None
            st.rerun()

with saved_tab:
    st.subheader("Saved Action Items")
    st.caption("This demo displays all local records; there is no account-level data isolation.")
    records = backend.get_tasks()
    if records:
        st.dataframe(records, use_container_width=True, hide_index=True)
    else:
        st.info("No tasks saved yet.")
    with st.expander("Ask Gemini about saved tasks"):
        question = st.text_input("Question", placeholder="وريني المهام اللي على أحمد")
        if st.button("Ask saved-task assistant", disabled=not bool(question.strip())):
            try:
                with st.spinner("Looking up tasks…"):
                    st.write(backend.ask_about_saved_tasks(question))
            except Exception as exc:
                st.error(f"Could not answer ({type(exc).__name__}).")
