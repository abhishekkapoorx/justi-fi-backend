from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
# from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, TypedDict
from langgraph.types import Send
from lib.llm import llm
from typing import List
from pydantic import BaseModel, Field
from MemoryManager import build_memory_manager, string_reducer


class SuperGraphState(TypedDict):
    user_id: str
    thread_id: str
    space_id: str
    input: str
    user_id: str
    space_id: str
    thread_id: str
    output_message: str
    intents: List[str]
    memory_summary: Annotated[str, string_reducer]

class InputState():
    input: str
    user_id: str
    space_id: str
    thread_id: str

class OutputState():
    result: str

def memory_manager(state):
    build_memory_manager().compile()
    return state

def intent_selector(state:SuperGraphState):
    """Analyze user input to determine their intentions and route accordingly.
    Uses LangChain's structured output for clean, type-safe parsing."""
    from lib.llm import llm
    from typing import List
    from pydantic import BaseModel, Field
    from langchain_core.messages import SystemMessage, HumanMessage
    
    newstate = state.copy()
    user_input = state.get("input", "")
    
    # Define Pydantic model for the expected structured output
    class IntentResponse(BaseModel):
        """Response schema for intent classification."""
        intents: List[str] = Field(
            description="List of identified intents from the user message. Should include one or more of: 'chat', 'generate_insight', 'notion_documentor', 'notion_scheduler'."
        )
        
        def __str__(self) -> str:
            """String representation of the intents for logging."""
            return f"Detected intents: {', '.join(self.intents)}"
    
    # Define system message for structured intent extraction
    system_message = """
    You are an intent classification system for a legal assistant AI.
    Your task is to analyze the user's input and identify what they want to do.

    POSSIBLE INTENTS:
    - "chat": For general conversation, questions, or clarifications
    - "generate_insight": When the user wants analysis, insights, or interpretations of legal information
    - "notion_documentor": When the user wants to create, update, or manage legal documents
    - "notion_scheduler": When the user wants to manage schedule, appointments, or deadlines

    Analyze the user's message and return the appropriate intents. Each message may have multiple intents.
    """
    
    user_message = f"User input: {user_input}\n\nPlease classify the intents in this message."
    
    try:
        # Use structured output directly with our Pydantic model
        structured_llm = llm.with_structured_output(IntentResponse)
        
        # Create a list of messages as input
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        # Pass the messages list as the input
        response = structured_llm.invoke(messages)
        
        # The response is already a Pydantic object - no need for json parsing
        newstate["intents"] = response.intents
        
        # Log the detected intents for debugging
        print(f"Detected intents: {response.intents}")
            
    except Exception as e:
        # Fallback to default intent if any error occurs
        print(f"Intent detection error: {e}. Using default intents.")
        newstate["intents"] = ["chat"]
    
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
    if "notion_scheduler" in intents:
        router_list.append(Send("notion_scheduler", {"intent": ""}))
    # return [Send(intent, {"hello": "world"}) for intent in intents if intent in ["chat", "generate_insight", "notion_documentor", "notion_aggregator"]]
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




