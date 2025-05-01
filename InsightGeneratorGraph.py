import operator
from typing import Annotated, TypedDict, List, Dict, Any, Callable
from langgraph.graph import StateGraph, START, END
from tools import getMessages, short_term_retrieval, long_term_retrieval, rag_law, rag_previous_cases
from MemoryManager import string_reducer

# -- State Schema Definition --
class InsightState(TypedDict, total=False):
    # Input fields
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer]
    memory_summary: Annotated[str, string_reducer]
    
    # Output fields
    summary: Annotated[str, string_reducer]
    positives: Annotated[List[str], operator.add]
    negatives: Annotated[List[Dict[str, Any]], operator.add]

class InsightStateInput(TypedDict):
    # Input fields for the graph
    input: Annotated[str, string_reducer]
    user_id: Annotated[str, string_reducer]
    space_id: Annotated[str, string_reducer]
    thread_id: Annotated[str, string_reducer]
    memory_summary: Annotated[str, string_reducer]

class InsightStateOutput(TypedDict):
    # Output fields returned by the graph
    summary: Annotated[str, string_reducer]
    positives: Annotated[List[str], operator.add]
    negatives: Annotated[List[Dict[str, Any]], operator.add]

# -- Node action functions --

def summarize_case(state: InsightStateInput) -> InsightState:
    """
    Summarize the entire case based only on chat messages from short-term memory.
    
    This function:
    1. Retrieves relevant conversation history from short-term memory
    2. Generates a comprehensive summary with the LLM using only chat context
    
    Returns the state updated with a detailed case summary based on chat messages
    """
    from tools import short_term_retrieval
    from lib.llm import llm
    from langchain_core.messages import SystemMessage, HumanMessage
    import re
    
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    # Extract key information
    user_input = new_state.get("input", "")
    user_id = new_state.get("user_id", "")
    space_id = new_state.get("space_id", "")
    memory_summary = new_state.get("memory_summary", "")
    
    # Build a context-rich query from the user's input and memory summary
    query = user_input
    query = f"{user_input} {memory_summary}"
    
    try:
        print(f"Generating case summary for query: {query}")
        
        # Retrieve only chat messages from short-term memory
        short_term_history = getMessages(user_id=user_id, space_id=space_id)
        
        # Format the retrieved chat messages for the LLM
        chat_context = short_term_history
        
        # Create a prompt for the LLM to generate a comprehensive case summary
        system_message = """
        You are a legal insight generator specialized in creating comprehensive case summaries.
        Your task is to analyze conversation history about a legal case and create a clear,
        structured summary that highlights the key aspects, legal issues, and relevant context.
        
        Focus on:
        1. The core legal questions and issues at stake
        2. Relevant facts and timeline of events
        3. Key legal principles mentioned in the conversation
        4. Client's goals and concerns
        5. Important dates, deadlines, or procedural steps mentioned
        6. Any advice or strategies discussed
        
        Format your summary in a clear, structured manner with appropriate headings.
        Only use information directly mentioned in the conversation history.
        Be objective and balanced in your assessment.
        """
        
        user_message = f"""
        Please generate a comprehensive summary of this legal case based on the following chat history:
        
        USER QUERY: {query}
        
        CONVERSATION HISTORY:
        {chat_context}
        
        Provide a well-structured, comprehensive summary that captures the essence of this legal matter
        based solely on the information in these conversations.
        """
        
        # Generate case summary using LLM
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        # Store the summary in the state
        new_state["summary"] = summary
        print("Case summary generated successfully.")
        
    except Exception as e:
        print(f"Error generating case summary: {e}")
        new_state["summary"] = f"Unable to generate case summary due to an error: {str(e)}"
    
    return new_state


def extract_positives(state: InsightState) -> InsightState:
    """
    Extract laws, precedents, and arguments that support the user's position.
    
    This function:
    1. Analyzes the case summary to identify favorable legal elements
    2. Extracts statutes, precedents, and legal principles that strengthen the case
    3. Structures these elements in a clear, actionable format
    
    Returns the state updated with a list of positive legal arguments
    """
    from lib.llm import llm
    from langchain_core.messages import SystemMessage, HumanMessage
    from typing import List
    from pydantic import BaseModel, Field
    
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    # Extract the case summary
    summary = new_state.get('summary', '')
    if not summary:
        print("No summary available to extract positive elements from")
        new_state['positives'] = []
        return new_state
    
    try:
        # Define a structured output format for positive elements
        class PositiveElement(BaseModel):
            """Model for a positive legal element that supports the case."""
            element: str = Field(description="The supporting statute, precedent, or legal principle")
            explanation: str = Field(description="Brief explanation of why this element is supportive")
            strength: str = Field(description="The relative strength of this support (Strong, Moderate, or Limited)")
        
        class PositiveElements(BaseModel):
            """Collection of positive legal elements."""
            elements: List[PositiveElement] = Field(description="List of elements that support the user's position")
        
        # Create a prompt for the LLM to extract positive elements
        system_message = """
        You are a legal insight extractor specialized in identifying favorable legal elements.
        Your task is to analyze a legal case summary and extract statutes, precedents, legal principles,
        and arguments that support the client's position or could be used to strengthen their case.
        
        For each positive element you identify, provide:
        1. The specific statute, precedent, or legal principle
        2. A brief explanation of why it's supportive
        3. An assessment of its strength (Strong, Moderate, or Limited)
        
        Be specific and precise. Focus only on genuinely supportive elements with legal merit.
        """
        
        user_message = f"""
        Please analyze the following legal case summary and identify all elements that support the client's position:
        
        {summary}
        
        Extract all statutes, precedents, legal principles, and arguments that could be used to strengthen the case.
        """
        
        # Extract positive elements using LLM with structured output
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        response = llm.with_structured_output(PositiveElements).invoke(messages)
        positive_elements = response.elements
        
        # Convert structured output to the format expected by the state
        formatted_positives = []
        for element in positive_elements:
            formatted_text = f"**{element.element}** ({element.strength})\n{element.explanation}"
            formatted_positives.append(formatted_text)
        
        # Store the positive elements in the state
        new_state["positives"] = formatted_positives
        print(f"Successfully extracted {len(formatted_positives)} positive elements")
        
    except Exception as e:
        print(f"Error extracting positive elements: {e}")
        new_state["positives"] = []
    
    return new_state


def extract_negatives(state: InsightState) -> InsightState:
    """
    Identify negative arguments, counterarguments, and legal obstacles to the user's position.
    
    This function:
    1. Analyzes the case summary to identify unfavorable legal elements
    2. Extracts opposing statutes, precedents, and legal principles that may weaken the case
    3. Structures these elements in a clear, actionable format with severity ratings
    
    Returns the state updated with a list of negative legal arguments
    """
    from lib.llm import llm
    from langchain_core.messages import SystemMessage, HumanMessage
    from typing import List
    from pydantic import BaseModel, Field
    
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    # Extract the case summary
    summary = new_state.get('summary', '')
    if not summary:
        print("No summary available to extract negative elements from")
        new_state['negatives'] = []
        return new_state
    
    try:
        # Define a structured output format for negative elements
        class NegativeElement(BaseModel):
            """Model for a negative legal element that challenges the case."""
            argument: str = Field(description="The opposing statute, precedent, or legal principle")
            explanation: str = Field(description="Brief explanation of why this element is challenging")
            severity: str = Field(description="The relative severity of this challenge (Critical, Significant, or Minor)")
            area: str = Field(description="The legal area or aspect of the case this affects")
        
        class NegativeElements(BaseModel):
            """Collection of negative legal elements."""
            elements: List[NegativeElement] = Field(description="List of elements that challenge the user's position")
        
        # Create a prompt for the LLM to extract negative elements
        system_message = """
        You are a legal risk assessor specialized in identifying challenging legal elements.
        Your task is to analyze a legal case summary and extract statutes, precedents, legal principles,
        and arguments that oppose the client's position or could be used against their case.
        
        For each negative element you identify, provide:
        1. The specific opposing statute, precedent, or legal principle
        2. A brief explanation of why it's challenging for the client's position
        3. An assessment of its severity (Critical, Significant, or Minor)
        4. The legal area or aspect of the case it affects (e.g., Procedural, Substantive, Jurisdictional)
        
        Be thorough and realistic. Your analysis will help the legal team prepare counterarguments.
        """
        
        user_message = f"""
        Please analyze the following legal case summary and identify all elements that challenge or oppose the client's position:
        
        {summary}
        
        Extract all statutes, precedents, legal principles, and arguments that could be used against the case.
        """
        
        # Extract negative elements using LLM with structured output
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        response = llm.with_structured_output(NegativeElements).invoke(messages)
        negative_elements = response.elements
        
        # Convert structured output to the format expected by the state
        formatted_negatives = []
        for element in negative_elements:
            formatted_negative = {
                "argument": element.argument,
                "explanation": element.explanation,
                "severity": element.severity,
                "area": element.area,
                "rebuttal": ""  # Empty placeholder for rebuttals to be filled later
            }
            formatted_negatives.append(formatted_negative)
        
        # Store the negative elements in the state
        new_state["negatives"] = formatted_negatives
        print(f"Successfully extracted {len(formatted_negatives)} negative elements")
        
    except Exception as e:
        print(f"Error extracting negative elements: {e}")
        new_state["negatives"] = []
    
    return new_state


def generate_rebuttals(state: InsightState) -> InsightState:
    """
    Generate rebuttal strategies for each negative argument identified in the case.
    
    This function:
    1. Analyzes each negative argument extracted from the case summary
    2. Develops strategic counterarguments and rebuttals for each challenge
    3. Considers the legal context, precedents, and facts of the case
    4. Provides actionable strategies to overcome or minimize each negative element
    
    Returns the state updated with rebuttals for each negative element
    """
    from lib.llm import llm
    from langchain_core.messages import SystemMessage, HumanMessage
    from typing import List
    from pydantic import BaseModel, Field
    import time
    
    # Make a copy of the state to avoid mutating it
    new_state = state.copy()
    
    # Extract the case summary and negative elements
    summary = new_state.get('summary', '')
    negatives = new_state.get('negatives', [])
    
    if not negatives:
        print("No negative elements to generate rebuttals for")
        return new_state
    
    updated_negatives = []
    
    try:
        # For each negative element, generate a specific rebuttal
        for negative in negatives:
            argument = negative.get("argument", "")
            explanation = negative.get("explanation", "")
            severity = negative.get("severity", "")
            area = negative.get("area", "")
            
            # Skip if missing essential information
            if not argument:
                updated_negatives.append(negative)
                continue
            
            # Create a structured output for rebuttals
            class Rebuttal(BaseModel):
                """Model for a rebuttal to a negative legal element."""
                counterargument: str = Field(description="The main counterargument or defensive strategy")
                legal_basis: str = Field(description="Legal basis or precedent supporting the counterargument")
                tactical_approach: str = Field(description="Tactical approach for presenting this rebuttal effectively")
                alternative_positions: str = Field(description="Alternative positions or fallback arguments if primary rebuttal fails")
            
            # Create a prompt for the LLM to generate a rebuttal
            system_message = """
            You are a legal defense strategist specialized in developing rebuttals to challenging legal arguments.
            Your task is to analyze a negative legal element in a case and develop effective counterarguments
            and strategies to overcome or minimize its impact.
            
            For each rebuttal you develop, provide:
            1. A main counterargument or defensive strategy
            2. The legal basis or precedent supporting your counterargument
            3. A tactical approach for presenting this rebuttal effectively
            4. Alternative positions or fallback arguments if the primary rebuttal fails
            
            Be creative but realistic. Your rebuttals should have solid legal foundations.
            """
            
            user_message = f"""
            Please develop a rebuttal strategy for the following negative legal element:
            
            CASE CONTEXT:
            {summary[:1000]}...
            
            NEGATIVE ELEMENT:
            Argument: {argument}
            Explanation: {explanation}
            Severity: {severity}
            Legal Area: {area}
            
            Develop a comprehensive rebuttal strategy to counter this negative element.
            """
            
            # Generate rebuttal using structured output
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ]
            
            response = llm.with_structured_output(Rebuttal).invoke(messages)
            
            # Update the negative element with the rebuttal
            negative_with_rebuttal = negative.copy()
            negative_with_rebuttal["rebuttal"] = {
                "counterargument": response.counterargument,
                "legal_basis": response.legal_basis,
                "tactical_approach": response.tactical_approach,
                "alternative_positions": response.alternative_positions
            }
            
            updated_negatives.append(negative_with_rebuttal)
            
            # Sleep briefly to avoid rate limiting
            time.sleep(0.5)
        
        # Update the state with rebuttals
        new_state["negatives"] = updated_negatives
        print(f"Successfully generated rebuttals for {len(updated_negatives)} negative elements")
        
    except Exception as e:
        print(f"Error generating rebuttals: {e}")
        new_state["negatives"] = negatives  # Return original negatives if an error occurs
    
    return new_state


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

    # Define transitions
    sg.add_edge(START, 'SummarizeCase')
    sg.add_edge('SummarizeCase', 'ExtractPositives')
    sg.add_edge('SummarizeCase', 'ExtractNegatives')
    sg.add_edge('ExtractPositives', END)
    sg.add_edge('ExtractNegatives', 'GenerateRebuttals')
    sg.add_edge('GenerateRebuttals', END)

    # Compile and return
    return sg
