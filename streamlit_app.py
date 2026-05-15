import io
import json
import time
import traceback
from contextlib import redirect_stdout

import cv2
import streamlit as st

from main import execute_plan, get_llm_plan
from obj_det import set_detection_profile


st.set_page_config(
    page_title="RE-ACT Console",
    page_icon="R",
    layout="wide",
)


def _run_and_capture(func, *args, **kwargs):
    """Run a function and capture printed logs for UI display."""
    buffer = io.StringIO()
    result = None
    err = None
    with redirect_stdout(buffer):
        try:
            result = func(*args, **kwargs)
        except Exception:
            err = traceback.format_exc()
    return result, buffer.getvalue(), err


def _validate_plan(plan_obj):
    if not isinstance(plan_obj, list):
        return False, "Plan must be a JSON array of steps."

    allowed_actions = {"grasp", "move", "drop"}
    for idx, step in enumerate(plan_obj, start=1):
        if not isinstance(step, dict):
            return False, f"Step {idx} must be a JSON object."

        action = step.get("action")
        if action not in allowed_actions:
            return False, f"Step {idx} has invalid action: {action}."

        if action in {"grasp", "move"} and not step.get("target"):
            return False, f"Step {idx} action {action} requires target."

        if action == "drop" and not (step.get("location") or step.get("target")):
            return False, f"Step {idx} action drop requires location."

    return True, ""


def _parse_plan_text(plan_text):
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"

    ok, msg = _validate_plan(plan)
    if not ok:
        return None, msg

    # Normalize drop action so execute_plan can read either target or location.
    for step in plan:
        if step.get("action") == "drop" and "target" in step and "location" not in step:
            step["location"] = step["target"]

    return plan, ""


if "task_text" not in st.session_state:
    st.session_state.task_text = ""
if "plan_text" not in st.session_state:
    st.session_state.plan_text = "[]"
if "llm_logs" not in st.session_state:
    st.session_state.llm_logs = ""
if "exec_logs" not in st.session_state:
    st.session_state.exec_logs = ""
if "speed_mode" not in st.session_state:
    st.session_state.speed_mode = "balanced"


st.title("RE-ACT Console")
st.caption("Task planning and execution")

left_col, right_col = st.columns([3, 1.8])

with left_col:
    st.subheader("Task")
    st.session_state.task_text = st.text_area(
        "Task description",
        value=st.session_state.task_text,
        height=130,
        placeholder="Example: pick the green tape and drop it in the box.",
    )

    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

    with action_col1:
        if st.button("Generate", use_container_width=True, type="primary"):
            task = st.session_state.task_text.strip()
            if not task:
                st.warning("Enter a task description first.")
            else:
                with st.spinner("Generating plan..."):
                    plan, logs, err = _run_and_capture(get_llm_plan, task)

                st.session_state.llm_logs = logs

                if err:
                    st.error("Planner error. See logs.")
                    st.session_state.plan_text = "[]"
                elif not plan:
                    st.warning("No valid plan returned. Check logs and API key.")
                    st.session_state.plan_text = "[]"
                else:
                    st.session_state.plan_text = json.dumps(plan, indent=2)
                    st.success("Plan ready.")

    with action_col2:
        if st.button("Validate", use_container_width=True):
            _, parse_err = _parse_plan_text(st.session_state.plan_text)
            if parse_err:
                st.error(parse_err)
            else:
                st.success("Plan is valid")

    with action_col3:
        if st.button("Run", use_container_width=True):
            plan, parse_err = _parse_plan_text(st.session_state.plan_text)
            if parse_err:
                st.error(parse_err)
            else:
                with st.spinner("Running plan..."):
                    _, logs, err = _run_and_capture(execute_plan, plan)

                st.session_state.exec_logs = logs

                if err:
                    st.error("Run failed.")
                    st.code(err, language="text")
                else:
                    st.success("Run completed.")

    st.subheader("Plan")
    st.session_state.plan_text = st.text_area(
        "Review or edit JSON plan",
        value=st.session_state.plan_text,
        height=380,
    )

with right_col:
    st.subheader("Settings")
    st.session_state.speed_mode = st.selectbox(
        "Detection mode",
        options=["fast", "balanced", "precise"],
        index=["fast", "balanced", "precise"].index(st.session_state.speed_mode),
        help="Fast is quicker. Precise is stricter.",
    )
    set_detection_profile(st.session_state.speed_mode)

    st.subheader("Camera Check")
    camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    preview_seconds = st.slider("Preview seconds", min_value=2, max_value=12, value=5)
    if st.button("Preview", use_container_width=True):
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            st.error("Could not open camera for preview.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            preview_slot = st.empty()
            end_time = time.time() + int(preview_seconds)
            while time.time() < end_time:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame from camera.")
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_slot.image(frame_rgb, channels="RGB", use_container_width=True)
            cap.release()
            st.success("Preview finished.")

    st.subheader("Planner Output")
    st.code(st.session_state.llm_logs or "No planner logs yet.", language="text")

    st.subheader("Run Output")
    st.code(st.session_state.exec_logs or "No execution logs yet.", language="text")


