from typing import Annotated, List, TypedDict
import operator
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import streamlit as st

# Define LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# ----- SCHEMA DEFINITIONS -----
class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of the main topics and concepts to be covered in this section.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

planner = llm.with_structured_output(Sections)

# ----- GRAPH STATE -----
class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

# ----- GRAPH NODES -----
def orchestrator(state: State):
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}"),
        ]
    )
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}

def assign_workers(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

# ----- BUILD WORKFLOW -----
def build_workflow():
    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("llm_call", llm_call)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
    builder.add_edge("llm_call", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()

# ----- STREAMLIT UI -----
st.set_page_config(page_title="LangGraph Report Generator", layout="wide")
st.title("📘 LangGraph Report Generator")
st.markdown("Generate a structured markdown report using LangGraph and OpenAI.")

topic = st.text_input("Enter a topic for the report:", value="Create a report on Agentic AI RAGs")

if st.button("Generate Report"):
    with st.spinner("Generating your report..."):
        try:
            orchestrator_worker = build_workflow()
            state = orchestrator_worker.invoke({"topic": topic})
            st.markdown(state["final_report"], unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
