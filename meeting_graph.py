import os
import re

from typing import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Add it to .env locally or to Streamlit secrets when deployed."
    )

os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    model="gpt-6-luna",
    temperature=0,
    reasoning_effort="none"
)


MAX_TRANSCRIPT_CHARS = 40000
NO_ACTION_MESSAGE = "No action items identified in this meeting."


class MeetingState(TypedDict):
    transcript: str
    key_topics: list[str]
    meeting_summary: str
    action_items: list[dict]
    final_report: str


class TopicList(BaseModel):
    topics: list[str]


class ActionItem(BaseModel):
    task: str
    owner: str
    deadline: str


class ActionItemList(BaseModel):
    items: list[ActionItem]


def input_node(state: MeetingState) -> dict:
    text = state["transcript"]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"\n\s*\n", "\n\n", text)

    if text == "":
        raise ValueError("The transcript is empty. Please paste or upload a meeting transcript.")

    if len(text) > MAX_TRANSCRIPT_CHARS:
        raise ValueError(f"The transcript has {len(text)} characters, which is more than the limit of {MAX_TRANSCRIPT_CHARS}.")

    return {"transcript": text}


TOPIC_PROMPT = """You are a meeting analyst. Read the meeting transcript below and identify the main topics that were discussed.

Rules:
- Write each topic as a short noun phrase of no more than six words.
- Include only topics that are actually discussed in the transcript. Do not add topics that are not mentioned.
- Each topic must be distinct. Do not repeat the same topic in different words.
- List only the main topics, usually 3 to 6. A short meeting may have fewer.
- Do not list greetings, thanks, or opening and closing remarks as topics.
- Do not list deadlines, dates or scheduling details as topics.
- If one candidate topic is a symptom, example, subtopic or supporting detail of another broader topic, keep only the broader main topic.
- When one candidate topic is the stated problem or symptom and another is the corresponding improvement goal for the same issue, combine them into one broader topic rather than listing them separately. Keep different pieces of work discussed to address the issue as separate topics.

Transcript:
{transcript}"""

topic_llm = llm.with_structured_output(TopicList)


def topic_extraction_node(state: MeetingState) -> dict:
    prompt = TOPIC_PROMPT.format(transcript=state["transcript"])
    result = topic_llm.invoke(prompt)
    return {"key_topics": result.topics}


SUMMARY_PROMPT = """You are a meeting analyst. Read the meeting transcript below and write a concise summary of the meeting.

Rules:
- Write between 3 and 5 complete sentences as a single paragraph.
- Explain what the meeting was about, the main issues discussed, and the next steps that were agreed, if any.
- Use only information stated in the transcript. Do not add facts, names, numbers or dates that are not in the transcript.
- Do not use bullet points, headings or lists.
- Return only the summary text.

Transcript:
{transcript}"""


def meeting_summary_node(state: MeetingState) -> dict:
    prompt = SUMMARY_PROMPT.format(transcript=state["transcript"])
    summary = llm.invoke(prompt).content.strip()
    return {"meeting_summary": summary}


ACTION_PROMPT = """You are a meeting analyst. Read the meeting transcript below and extract the action items.

An action item is a specific task that someone is expected to do after the meeting. Include only:
- Commitments or clear offers to take responsibility for a specific future task, for example "I will ...", "Let me ..." or "I can ...". Do not include statements that merely describe what the speaker is capable of doing.
- Assignments, where a task is given to a named person or role.
- Directives, where a specific task is requested, for example "Please ...".
- Agreed next steps, where a specific, concrete action is proposed with "Let's ..." or "We should ..." and is not rejected, or a proposal that another participant accepts.
- Unassigned tasks, where the meeting explicitly states that someone is needed to perform a specific future task but nobody is assigned, for example "We need someone to ...". Use "Not specified" as the owner.

Do not include:
- Broad goals or general "we need to ..." statements are not action items. Do not extract them unless the transcript explicitly indicates that someone is needed to take responsibility for the specific task.
- Questions without an agreed answer, suggestions nobody agreed to, or speculation such as "could be" or "might be".
- Observations, status updates, or work already done before the meeting.
- Statements that only set a deadline for tasks already mentioned. Use that timing as the deadline of those tasks instead.
- Requests to prioritise, focus on or treat an issue as urgent when they express urgency rather than a separate concrete task or deliverable.

For each action item, provide:
- task: a short description of the task, using only what the transcript says.
- owner: if a speaker commits to the task, use their speaker label exactly as written in the transcript. If the task is assigned to a named person or role, use that name or role exactly as written. If a speaker agrees to take on the same commitment as another speaker, list it as a separate action item for that speaker. If nobody is responsible, write "Not specified". Never guess an owner.
- deadline: copy the time phrase exactly as written in the transcript. If a time phrase applies to several tasks, use it for each of them. If there is more than one time phrase for a task, separate them with a semicolon. If no time is stated, write "Not specified". Never convert a phrase into a date and never invent a deadline.
- If the transcript explicitly states that a specific task is urgent, for example "urgent", "urgently" or "ASAP", also include that urgency word in the deadline, separated by a semicolon from any time phrase. If urgency is stated but no time, use the urgency word as the deadline. Do this only when the urgency is clearly linked to that task.
- If a deadline is conditional, preserve the condition together with the time phrase so the deadline is not presented as unconditional.
- A deadline may be inherited from the immediately preceding context only when the speaker clearly adopts the same timing, for example by explicitly agreeing to the same commitment or timing. Do not infer timing from distant context.

Additional rules:
- Do not treat a general meeting-level urgency or timing statement as a deadline for a specific task unless the transcript clearly links that timing to that task. Do not carry timing forward to later tasks merely because they appear after the statement.
- A first-person commitment to perform a specific task can still count when phrased cautiously, such as "should be able to", if the intended action is clear. Preserve any stated timing and condition together.

List each action once. If the same owner restates or confirms the same task, for example by accepting an assignment, list it as one item and keep all of its time phrases. Never combine actions that have different owners, never combine an action without an owner with an action that has an owner, and never give an action an owner because of a related action. Keep different pieces of work as separate items.

If there are no action items, return an empty list.

Transcript:
{transcript}"""

action_llm = ChatOpenAI(
    model="gpt-6-luna",
    reasoning_effort="medium"
).with_structured_output(ActionItemList)


def action_extraction_node(state: MeetingState) -> dict:
    prompt = ACTION_PROMPT.format(transcript=state["transcript"])
    result = action_llm.invoke(prompt)
    action_items = []
    seen = []
    for item in result.items:
        task_key = " ".join(item.task.lower().split())
        owner_key = " ".join(item.owner.lower().split())
        deadline_key = " ".join(item.deadline.lower().split())
        key = (task_key, owner_key, deadline_key)
        if key in seen:
            continue
        seen.append(key)
        action_items.append({
            "task_id": len(action_items) + 1,
            "task": item.task,
            "owner": item.owner,
            "deadline": item.deadline
        })
    return {"action_items": action_items}


def action_check_node(state: MeetingState) -> dict:
    return {}


def route_actions(state: MeetingState) -> str:
    if len(state["action_items"]) > 0:
        return "prioritise"
    return "skip"


HIGH_PRIORITY_PHRASES = [
    "urgent", "urgently", "asap", "immediately", "right away",
    "today", "tonight", "end of day", "end of the day", "tomorrow",
    "within the hour", "within the next hour", "within an hour", "this week"
]

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

MEDIUM_PRIORITY_PHRASES = ["next week", "next month", "soon", "later", "once", "after", "when", "before"]

NEGATED_URGENCY_PHRASES = ["not urgently", "not urgent", "no rush"]


def contains_phrase(text, phrase):
    return re.search(r"\b" + phrase + r"\b", text) is not None


def classify_priority(task, deadline):
    text = (task + " " + deadline).lower()

    for phrase in NEGATED_URGENCY_PHRASES:
        text = re.sub(r"\b" + phrase + r"\b", " ", text)

    for phrase in HIGH_PRIORITY_PHRASES:
        if contains_phrase(text, phrase):
            return "High Priority"

    for day in WEEKDAYS:
        if contains_phrase(text, "next " + day):
            return "Medium Priority"
        if contains_phrase(text, day):
            return "High Priority"

    for phrase in MEDIUM_PRIORITY_PHRASES:
        if contains_phrase(text, phrase):
            return "Medium Priority"

    return "Low Priority"


def priority_classification_node(state: MeetingState) -> dict:
    prioritised_items = []
    for item in state["action_items"]:
        prioritised_item = dict(item)
        prioritised_item["priority"] = classify_priority(item["task"], item["deadline"])
        prioritised_items.append(prioritised_item)
    return {"action_items": prioritised_items}


def final_output_node(state: MeetingState) -> dict:
    report_lines = []

    report_lines.append("1. Meeting Summary")
    report_lines.append(state["meeting_summary"])
    report_lines.append("")

    report_lines.append("2. Key Topics")
    if len(state["key_topics"]) == 0:
        report_lines.append("No key topics identified.")
    for number, topic in enumerate(state["key_topics"], start=1):
        report_lines.append(f"{number}. {topic}")
    report_lines.append("")

    action_items = state["action_items"]

    if len(action_items) == 0:
        for heading in ["3. Action Items", "4. Task Owners", "5. Priority Level"]:
            report_lines.append(heading)
            report_lines.append(NO_ACTION_MESSAGE)
            report_lines.append("")
        return {"final_report": "\n".join(report_lines).strip()}

    report_lines.append("3. Action Items")
    for item in action_items:
        report_lines.append(f"{item['task_id']}. Task: {item['task']} | Owner: {item['owner']} | Deadline: {item['deadline']} | Priority: {item['priority']}")
    report_lines.append("")

    report_lines.append("4. Task Owners")
    owners = []
    for item in action_items:
        if item["owner"] not in owners:
            owners.append(item["owner"])
    for owner in owners:
        report_lines.append(f"Owner: {owner}")
        for item in action_items:
            if item["owner"] == owner:
                report_lines.append(f"- {item['task']}")
    report_lines.append("")

    report_lines.append("5. Priority Level")
    for level in ["High Priority", "Medium Priority", "Low Priority"]:
        level_tasks = []
        for item in action_items:
            if item["priority"] == level:
                level_tasks.append(item["task"])
        report_lines.append(f"{level} ({len(level_tasks)})")
        if len(level_tasks) == 0:
            report_lines.append("- None")
        for task in level_tasks:
            report_lines.append(f"- {task}")

    return {"final_report": "\n".join(report_lines)}


def build_graph():
    builder = StateGraph(MeetingState)

    builder.add_node("input", input_node)
    builder.add_node("topic_extraction", topic_extraction_node)
    builder.add_node("meeting_summary", meeting_summary_node)
    builder.add_node("action_extraction", action_extraction_node)
    builder.add_node("action_check", action_check_node)
    builder.add_node("priority_classification", priority_classification_node)
    builder.add_node("final_output", final_output_node)

    builder.add_edge(START, "input")
    builder.add_edge("input", "topic_extraction")
    builder.add_edge("input", "meeting_summary")
    builder.add_edge("input", "action_extraction")
    builder.add_edge(["topic_extraction", "meeting_summary", "action_extraction"], "action_check")
    builder.add_conditional_edges(
        "action_check",
        route_actions,
        {"prioritise": "priority_classification", "skip": "final_output"}
    )
    builder.add_edge("priority_classification", "final_output")
    builder.add_edge("final_output", END)

    return builder.compile()
