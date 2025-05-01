from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
# from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, TypedDict
from langgraph.types import Send
from ChatSubGraph import build_chat_subgraph
from lib.llm import llm
from typing import List
from pydantic import BaseModel, Field
from MemoryManager import build_memory_manager, string_reducer


class SuperGraphState(TypedDict):
    user_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer]
    output_message: Annotated[str, string_reducer]
    intents: List[str]
    chat_results: Annotated[str, string_reducer]
    memory_summary: Annotated[str, string_reducer]

class InputState():
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer] = "uid"
    space_id: Annotated[str, string_reducer] = "sid"
    thread_id: Annotated[str, string_reducer] = "tid"

class OutputState():
    result: Annotated[str, string_reducer]

class IntentResponse(BaseModel):
    """Response schema for intent classification."""
    intents: List[str] = Field(
        description="List of identified intents from the user message. Should include one or more of: 'chat', 'generate_insight', 'notion_documentor', 'notion_scheduler'."
    )
    
    def __str__(self) -> str:
        """String representation of the intents for logging."""
        return f"Detected intents: {', '.join(self.intents)}"

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

def aggregator(state: SuperGraphState) -> OutputState:
    """Aggregate results from all graph nodes, prioritizing chat responses.
    
    This function:
    1. Prioritizes chat_results if available
    2. Summarizes the work done by other nodes (insight_generator, notion_documentor, notion_scheduler)
    3. Creates a structured response that helps the user understand all system activities
    4. Returns a unified result in the OutputState format
    """
    from lib.llm import llm
    from langchain_core.messages import SystemMessage, HumanMessage
    import re
    
    # Initialize the output state
    output = {"result": ""}
    
    # Check which intents were identified and processed
    intents = state.get("intents", [])
    resolved_intents = []
    
    # Priority 1: Chat response (if available)
    chat_results = state.get("chat_results", "")
    has_chat = "chat" in intents and chat_results
    if has_chat:
        resolved_intents.append("chat")
        
    # Check for other intent results
    insight_results = state.get("insight_results", "")
    has_insights = "generate_insight" in intents and insight_results
    if has_insights:
        resolved_intents.append("generate_insight")
    
    document_results = state.get("document_results", "")
    has_documents = "notion_documentor" in intents and document_results
    if has_documents:
        resolved_intents.append("notion_documentor")
    
    scheduler_results = state.get("scheduler_results", "")
    has_scheduling = "notion_scheduler" in intents and scheduler_results
    if has_scheduling:
        resolved_intents.append("notion_scheduler")
    
    # If we have chat results, prioritize them
    if has_chat:
        main_response = chat_results
    # Otherwise, use whatever result we have available
    elif has_insights:
        main_response = insight_results
    elif has_documents:
        main_response = document_results
    elif has_scheduling:
        main_response = scheduler_results
    else:
        # If no specific results, but intents were detected, explain what was attempted
        if intents:
            operations_attempted = []
            if "chat" in intents:
                operations_attempted.append("answering your query")
            if "generate_insight" in intents:
                operations_attempted.append("generating legal insights")
            if "notion_documentor" in intents:
                operations_attempted.append("managing legal documents")
            if "notion_scheduler" in intents:
                operations_attempted.append("handling scheduling tasks")
                
            operations_text = ", ".join(operations_attempted[:-1])
            if len(operations_attempted) > 1:
                operations_text += f", and {operations_attempted[-1]}"
            else:
                operations_text = operations_attempted[0]
                
            main_response = f"I processed your request by {operations_text}, but wasn't able to generate specific results. Could you please provide more details or rephrase your request?"
        else:
            # Complete fallback if no intents detected
            main_response = "I'm sorry, but I wasn't able to process your request effectively. Could you please rephrase or provide more details?"
    
    # If we need to summarize additional operations beyond the main response
    additional_operations = []
    
    # Generate LLM-enhanced descriptions of operations if needed
    if len(resolved_intents) > 1 or (intents and not resolved_intents):
        # Only generate descriptions for operations that weren't the main response
        operations_to_describe = []
        
        if has_insights and "generate_insight" not in resolved_intents[:1]:
            operations_to_describe.append(("Legal Analysis", insight_results))
            
        if has_documents and "notion_documentor" not in resolved_intents[:1]:
            operations_to_describe.append(("Document Management", document_results))
            
        if has_scheduling and "notion_scheduler" not in resolved_intents[:1]:
            operations_to_describe.append(("Scheduling", scheduler_results))
            
        # If we have operations to describe, use LLM to summarize them
        if operations_to_describe:
            try:
                # For each operation, get a concise summary using LLM
                for op_type, op_result in operations_to_describe:
                    # Generate icon based on operation type
                    icon = "📊" if op_type == "Legal Analysis" else "📄" if op_type == "Document Management" else "📅"
                    
                    # Use LLM to generate a concise summary
                    messages = [
                        SystemMessage(content=f"""
                        You are a legal assistant summarization expert. Your task is to create a very concise 
                        summary (1-2 sentences) of the {op_type.lower()} operations that were performed.
                        Focus only on what was accomplished, not how it was done.
                        Start directly with the key outcome, no preamble needed.
                        """),
                        HumanMessage(content=f"Please summarize this {op_type.lower()} result concisely: {op_result[:500]}...")
                    ]
                    
                    summary = llm.invoke(messages).content
                    # Clean up common verbosity patterns
                    summary = re.sub(r'^(I |In summary, |To summarize, |Overall, |The system )', '', summary)
                    summary = summary.strip()
                    
                    additional_operations.append(f"{icon} **{op_type}**: {summary}")
            except Exception as e:
                # Fall back to simple summaries if LLM fails
                print(f"Error generating operation summaries: {e}")
                for op_type, op_result in operations_to_describe:
                    icon = "📊" if op_type == "Legal Analysis" else "📄" if op_type == "Document Management" else "📅"
                    summary = summarize_result(op_result)
                    additional_operations.append(f"{icon} **{op_type}**: {summary}")
                    
        # If we attempted operations that didn't produce results, mention them
        attempted_but_failed = [intent for intent in intents if intent not in resolved_intents]
        if attempted_but_failed:
            failed_operations = []
            if "chat" in attempted_but_failed:
                failed_operations.append("answering your query")
            if "generate_insight" in attempted_but_failed:
                failed_operations.append("generating legal insights")
            if "notion_documentor" in attempted_but_failed:
                failed_operations.append("processing legal documents")
            if "notion_scheduler" in attempted_but_failed:
                failed_operations.append("managing scheduling tasks")
                
            if failed_operations:
                failed_text = ", ".join(failed_operations[:-1])
                if len(failed_operations) > 1:
                    failed_text += f", and {failed_operations[-1]}"
                else:
                    failed_text = failed_operations[0]
                    
                additional_operations.append(f"ℹ️ **Note**: I also attempted {failed_text}, but couldn't complete those operations with the provided information.")
    
    # Construct the final response
    final_response = main_response
    
    # Add summaries of additional operations if they exist
    if additional_operations:
        final_response += "\n\n---\n\n**Additional operations performed:**\n\n"
        final_response += "\n\n".join(additional_operations)
    
    output["result"] = final_response
    return output

def summarize_result(result_text: str, max_length: int = 100) -> str:
    """Helper function to create a brief summary of operation results.
    
    Args:
        result_text: The full result text to summarize
        max_length: Maximum length of the summary
        
    Returns:
        A brief summary of the operation result
    """
    # If the result is short enough, just return it
    if len(result_text) <= max_length:
        return result_text
        
    # Try to find the first sentence or meaningful chunk
    sentences = result_text.split('.')
    if sentences and len(sentences[0]) + 1 <= max_length:
        return sentences[0] + '...'
        
    # If that doesn't work, just truncate
    return result_text[:max_length - 3] + '...'

# Add conditional edges from memory_manager based on intent
def route_from_memory(state:SuperGraphState):
    # This is a placeholder for actual intent-based routing logic
    # In a real implementation, you would examine the state to determine the next node
    print("-x"*100,state)
    intents = state["intents"]
    router_list = []
    # if "chat" in intents:
    #     router_list.append(Send("chat", {"input": state["input"], "user_id": state["user_id"], "space_id": state["space_id"], "thread_id": state["thread_id"], "memory_summary": state["memory_summary"]}))
    router_list.append(Send("chat", {"input": state["input"], "user_id": state["user_id"], "space_id": state["space_id"], "thread_id": state["thread_id"], "memory_summary": state["memory_summary"]}))
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
graph.add_node("chat", build_chat_subgraph().compile())
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




