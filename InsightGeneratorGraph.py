import operator
from typing import Annotated, TypedDict, List, Dict, Any, Callable
from langgraph.graph import StateGraph, START, END

from MemoryManager import string_reducer

# -- State Schema Definition --
class InsightState(TypedDict, total=False):
    # Input instruction or context for insight generation
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer]
    memory_summary: Annotated[str, string_reducer]

    
    # Outputs
    summary: Annotated[str, string_reducer]
    positives: Annotated[List[str], operator.add]
    negatives: Annotated[List[Dict[str, str]], operator.add]

class InsightStateInput(TypedDict):
    # Input instruction or context for insight generation
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer]
    memory_summary: Annotated[str, string_reducer]

class InsightStateOutput(TypedDict):
    # Outputs
    summary: Annotated[str, string_reducer]
    positives: Annotated[List[str], operator.add]
    negatives: Annotated[List[Dict[str, str]], operator.add]

# -- Node action functions --

def summarize_case(state: InsightState) -> InsightState:
    """
    Summarize the entire case based on merged_context
    """
    context = "\n".join(state.get('merged_context', []))
    # Call summarization LLM
    return state


def extract_positives(state: InsightState) -> InsightState:
    """
    Extract laws and judgements supporting the user
    """
    summary = state.get('summary', '')
    # Call LLM or rule-based extractor
    return state


def extract_negatives(state: InsightState) -> InsightState:
    """
    Identify negative arguments and relevant laws
    """
    summary = state.get('summary', '')
    return state


def generate_rebuttals(state: InsightState) -> InsightState:
    """
    Generate rebuttal technique for each negative argument
    """
    negatives = state.get('negatives', [])
    for item in negatives:
        argument = item['argument']
    state['negatives'] = negatives
    return state

def merged_context(state: InsightState) -> InsightState:
    """
    Placeholder for merged context state
    """
    # This state can be used to merge or process the context further if needed
    return state


# -- Build Insight Generator StateGraph --
def build_insight_generator():
    sg = StateGraph(InsightState)

    # Register states
    sg.add_node('SummarizeCase', summarize_case)
    sg.add_node('ExtractPositives', extract_positives)
    sg.add_node('ExtractNegatives', extract_negatives)
    sg.add_node('GenerateRebuttals', generate_rebuttals)
    sg.add_node("MergedContext", merged_context)  # Placeholder for merged context

    # Define transitions
    sg.add_edge(START, 'SummarizeCase')
    sg.add_edge('SummarizeCase', 'ExtractPositives')
    sg.add_edge('SummarizeCase', 'ExtractNegatives')
    sg.add_edge('ExtractPositives', 'MergedContext')
    sg.add_edge('ExtractNegatives', 'GenerateRebuttals')
    sg.add_edge('GenerateRebuttals', "MergedContext")
    sg.add_edge('MergedContext', END)

    # Compile and return
    return sg
