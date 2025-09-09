import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads   # if file is backend.py -> from backend import ...
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from datetime import datetime

#Utilities
def generate_thread_id():
    return str(uuid.uuid4())

def build_title_from_text(text: str, max_len: int = 40) -> str:
    if not text:
        return "New chat"
    t = text.strip().splitlines()[0] 
    if len(t) > max_len:
        t = t[:max_len - 1] + "…"
    return t

def get_or_build_title(thread_id: str) -> str:
    if "chat_titles" in st.session_state and thread_id in st.session_state["chat_titles"]:
        return st.session_state["chat_titles"][thread_id]

    try:
        msgs = chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values.get("messages", [])
        first_user = next((m for m in msgs if isinstance(m, HumanMessage)), None)
        title = build_title_from_text(first_user.content if first_user else "")
    except Exception:
        title = "New chat"

    st.session_state.setdefault("chat_titles", {})
    st.session_state["chat_titles"][thread_id] = title
    return title

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state.setdefault("chat_titles", {})
    st.session_state["chat_titles"][thread_id] = "New chat"

def add_thread(thread_id):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state or not st.session_state["chat_threads"]:
    try:
        st.session_state["chat_threads"] = retrieve_all_threads() or []
    except Exception:
        st.session_state["chat_threads"] = []

st.session_state.setdefault("chat_titles", {})

add_thread(st.session_state["thread_id"])
st.session_state["chat_titles"].setdefault(st.session_state["thread_id"], "New chat")

for tid in list(st.session_state["chat_threads"]):
    get_or_build_title(tid)

# Sidebar
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat", key="new_chat_btn"):
    reset_chat()

st.sidebar.header("My Conversations")

threads_desc = []
for tid in st.session_state["chat_threads"]:
    title = get_or_build_title(tid)
    threads_desc.append((tid, title))

for idx, (thread_id, title) in enumerate(threads_desc[::-1]):
    if st.sidebar.button(title, key=f"thread_btn_{idx}_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                temp_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                compact = f"[tool: {tool_name}] {msg.content}"
                temp_messages.append({"role": "assistant", "content": compact})
        st.session_state["message_history"] = temp_messages

# Main UI

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    cur_tid = st.session_state["thread_id"]
    first_title = build_title_from_text(user_input)

    if not st.session_state["chat_titles"].get(cur_tid) or st.session_state["chat_titles"][cur_tid] == "New chat":
        st.session_state["chat_titles"][cur_tid] = first_title

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, list):
                        content = " ".join(map(str, content))
                    if content is None:
                        content = ""
                    yield content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
