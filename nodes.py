from utils import extraction_node, analyzer_node, evaluator_node
from langgraph.graph import StateGraph, END
from typing import List,TypedDict
from agent_state import AgentState

def create_analyzer_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("extractor", extraction_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("evaluator", evaluator_node)

    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "analyzer")
    workflow.add_edge("analyzer", "evaluator")
    workflow.add_edge("evaluator", END)
    
    return workflow.compile()