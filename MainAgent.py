from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
# from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, TypedDict
from langgraph.types import Send

from MemoryManager import build_memory_manager


class SuperGraphState(TypedDict):
    user_id: str
    thread_id: str
    space_id: str
    query: str
    output_message: str
    intents: List[str]
    merged_context: str

class InputState():
    query: str

class OutputState():
    result: str

def memory_manager(state):
    build_memory_manager().compile()
    return state

def intent_selector(state:SuperGraphState):
    newstate = state.copy()
    newstate["intents"] = ["chat", "generate_insight"]
    return newstate

def chat(state):
    return state

def insight_generator(state):
    return state

def notion_documentor(state):
    return state

def notion_scheduler(state):
    return state

def aggregator(state):
    return state


# Add conditional edges from memory_manager based on intent
def route_from_memory(state:SuperGraphState):
    # This is a placeholder for actual intent-based routing logic
    # In a real implementation, you would examine the state to determine the next node
    print("-x"*100,state)
    intents = state["intents"]
    router_list = []
    if "chat" in intents:
        router_list.append(Send("chat", {"intent": ""}))
    if "generate_insight" in intents:
        router_list.append(Send("insight_generator", {"intent": ""}))
    if "notion_documentor" in intents:
        router_list.append(Send("notion_documentor", {"intent": ""}))
    if "notion_aggregator" in intents:
        router_list.append(Send("notion_aggregator", {"intent": ""}))

    return router_list


graph = StateGraph(SuperGraphState, input=InputState, output=OutputState)

graph.add_node("memory_manager", build_memory_manager().compile())
graph.add_node("intent_selector", intent_selector)
graph.add_node("chat", chat)
graph.add_node("insight_generator", insight_generator)
graph.add_node("notion_documentor", notion_documentor)
graph.add_node("notion_scheduler", notion_scheduler)
graph.add_node("result_aggregator", aggregator)

# Define the edges between nodes
graph.add_edge(START, "memory_manager")
graph.add_edge("memory_manager", "intent_selector")


graph.add_conditional_edges(
    "intent_selector",
    route_from_memory,
    {
        "chat": "chat",
        "insight_generator": "insight_generator",
        "notion_documentor": "notion_documentor",
        "notion_scheduler": "notion_scheduler"
    }
)

graph.add_edge("chat", "result_aggregator")
graph.add_edge("insight_generator", "result_aggregator")
graph.add_edge("notion_documentor", "result_aggregator")
graph.add_edge("notion_scheduler", "result_aggregator")
graph.add_edge("result_aggregator", END)

# Compile the graph
memory_path = "./agent_memory"
# memory_saver = MemorySaver()
# compiled_graph = graph.compile(checkpointer=memory_saver)
compiled_graph = graph.compile()
compiled_graph




