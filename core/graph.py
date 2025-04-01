from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, List, Callable, Tuple, TypedDict, Any, Type, Union

def create_agent_graph(
    state_type: Type[TypedDict],
    nodes: Dict[str, Callable],
    edges: List[Tuple[str, str]],
    conditional_edges: Dict[str, Tuple[Callable, Dict[str, str]]] = None
):
    """
    Create a standard agent graph with the given nodes and edges
    
    Args:
        state_type: Type for the graph state
        nodes: Dictionary mapping node names to node functions
        edges: List of (source, target) edge tuples
        conditional_edges: Optional dict mapping source nodes to (router_func, target_dict) pairs
    
    Returns:
        Compiled StateGraph
    """
    builder = StateGraph(state_type)
    
    # Add nodes
    for name, func in nodes.items():
        builder.add_node(name, func)
    
    # Add edges
    for source, target in edges:
        # Handle START and END special cases
        source_node = START if source == "START" else source
        target_node = END if target == "END" else target
        builder.add_edge(source_node, target_node)
    
    # Add conditional edges
    if conditional_edges:
        for source, (router_func, targets) in conditional_edges.items():
            builder.add_conditional_edges(source, router_func, targets)
    
    # Compile with memory checkpointer
    return builder.compile(checkpointer=MemorySaver())

def create_interrupt_node(field_name: str):
    """
    Create a node that interrupts the graph to get user input
    
    Args:
        field_name: Field in the state to store the user's input
    
    Returns:
        A node function that can be used in a LangGraph
    """
    from langgraph.types import interrupt
    
    def get_user_input(state):
        value = interrupt({})
        return {field_name: value}
    
    return get_user_input 