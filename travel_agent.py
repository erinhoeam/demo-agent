#!/usr/bin/env python
"""
Travel Planning Agent with LangGraph and Azure OpenAI

This script demonstrates how to build a travel planning agent using LangGraph and Azure OpenAI.
The agent coordinates between specialized sub-agents to plan a trip based on user input.

Sub-agents:
1. Attractions Agent - Finds tourist attractions for a destination
2. Route Planner Agent - Plans a route between attractions
3. Finalizer Agent - Consolidates information into a final travel plan
"""

import os
import json
import logging
import argparse
from typing import Annotated, Dict, List, Sequence, TypedDict
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

# LangGraph imports
from langgraph.graph import END, StateGraph

# Load .env file from the correct path - ensure it's only loaded once at the start
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define our state typings
class AgentState(TypedDict):
    """Type definition for the agent state"""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation so far"]
    destination: Annotated[str, "The travel destination"]
    attractions: Annotated[List[Dict], "List of tourist attractions"]
    route_plan: Annotated[Dict, "Route planning information"]
    final_plan: Annotated[Dict, "Final consolidated travel plan"]

def create_azure_llm(temperature: float = 0.7):
    """
    Create an Azure OpenAI model with API key authentication and error handling
    
    Args:
        temperature: Temperature for text generation (0.0 to 1.0)
        
    Returns:
        Configured AzureChatOpenAI instance
    """
    try:
        # Get configuration from environment variables - don't load_dotenv again here
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        
        if not endpoint or not deployment or not api_key:
            raise ValueError("Missing required environment variables: "
                           "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, and AZURE_OPENAI_API_KEY must be set")
        
        logger.info(f"Initializing Azure OpenAI client (endpoint={endpoint}, deployment={deployment})")
        
        # Initialize Azure OpenAI with API key authentication
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            api_key=api_key,
            temperature=temperature
        )
        
        return llm
    except Exception as e:
        logger.error(f"Error creating Azure OpenAI client: {e}")
        raise

# Create different specialized agents

# 1. Router agent to coordinate
def create_router_agent():
    """Create a router agent that decides which specialized agent to call"""
    router_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Travel Planning Assistant that helps coordinate the travel planning process.
        Your job is to:
        1. Analyze the current state of the travel planning
        2. Determine what needs to be done next
        3. Route to the appropriate specialized agent
        
        Return your decision as JSON with this format:
        {"next_agent": "attractions" | "route_planner" | "finalizer"}
        
        Choose:
        - "attractions" if we need to find tourist attractions
        - "route_planner" if we have attractions and need to plan a route
        - "finalizer" if we have both attractions and route, and need to create a final plan
        """),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessage(content="""Current Planning State:
        Destination: {destination}
        Attractions found: {attractions_status}
        Route planned: {route_status}
        
        Decide which agent should be called next and respond in JSON format.
        """)
    ])
    
    llm = create_azure_llm(temperature=0)
    parser = JsonOutputParser()
    
    return router_prompt | llm | parser

# 2. Attractions agent
def create_attractions_agent():
    """Create an agent that finds tourist attractions for a destination"""
    attractions_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Tourist Attractions Expert.
        Your job is to identify the top attractions for the given travel destination.
        For each attraction, provide:
        - Name
        - Brief description
        - Why it's worth visiting
        - Estimated time to spend there
        
        Return a list of 5-10 attractions in JSON format:
        {"attractions": [
          {
            "name": "Attraction Name",
            "description": "Brief description",
            "highlights": "Why it's worth visiting",
            "recommended_time": "Estimated time to spend"
          }
          ...additional attractions...
        ]}
        """),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="Find the best tourist attractions for {destination}")
    ])
    
    llm = create_azure_llm(temperature=0.7)
    parser = JsonOutputParser()
    
    return attractions_prompt | llm | parser

# 3. Route planner agent
def create_route_planner_agent():
    """Create an agent that plans a route between attractions"""
    route_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Travel Route Planning Expert.
        Your job is to create an efficient route to visit the attractions provided.
        
        Consider these factors when planning:
        - Logical order of attractions based on proximity
        - Time needed at each location
        - Transportation options between locations
        - Suggested schedule for each day
        - Grouping nearby attractions together
        
        DO NOT ASK FOR MORE INFORMATION. Use the attractions already provided.
        
        ALWAYS return your plan in this exact JSON format:
        {"route_plan": {
          "day_by_day_itinerary": [
            {
              "day": 1,
              "activities": [
                {"time": "9:00 AM", "activity": "Visit X", "duration": "2 hours", "notes": "Additional tips"}
              ]
            }
          ],
          "transportation_tips": "Suggestions for getting around",
          "estimated_costs": "Rough cost estimates for transportation"
        }}
        """),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessage(content="""Below are the attractions to include in your route plan for {destination}:
        
        {attractions_details}
        
        CREATE A LOGICAL AND EFFICIENT ROUTE PLAN IN THE EXACT JSON FORMAT SPECIFIED.
        DO NOT ASK QUESTIONS OR REQUEST MORE INFORMATION.
        RETURN ONLY THE JSON RESPONSE.
        """)
    ])
    
    llm = create_azure_llm(temperature=0.7)
    parser = JsonOutputParser()
    
    return route_prompt | llm | parser

# 4. Finalizer agent
def create_finalizer_agent():
    """Create an agent that consolidates all information into a final plan"""
    finalizer_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Travel Plan Consolidation Expert.
        Your job is to create a comprehensive travel plan by consolidating information about:
        1. The destination
        2. Key attractions
        3. The planned route
        
        Create a well-organized, easy to follow travel plan with:
        - Introduction to the destination
        - Detailed day-by-day itinerary
        - Practical tips for the traveler
        - Cost estimates
        - Packing suggestions based on the destination
        
        Return the final plan in JSON format:
        {"final_plan": {
          "destination_overview": "Brief overview of the destination",
          "itinerary": [Day by day plan],
          "practical_tips": "Useful tips for the traveler",
          "cost_estimate": "Overall cost estimate",
          "packing_suggestions": "What to pack for this trip"
        }}
        """),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessage(content="""Create a final travel plan for {destination} based on:
        
        Attractions:
        {attractions_details}
        
        Route Plan:
        {route_details}
        
        Generate a comprehensive final travel plan in JSON format.
        """)
    ])
    
    llm = create_azure_llm(temperature=0.7)
    parser = JsonOutputParser()
    
    return finalizer_prompt | llm | parser

# Main router logic
def route_agent(state: AgentState) -> str:
    """Determine which agent should be called next based on the current state"""
    # Check the actual state values to determine status, not just existence
    has_attractions = bool(state["attractions"] and len(state["attractions"]) > 0)
    has_route = bool(state["route_plan"] and len(state["route_plan"]) > 0)
    
    # Prepare state information for the router
    attractions_status = "Completed" if has_attractions else "Not started"
    route_status = "Completed" if has_route else "Not started"
    
    logger.info(f"Routing decision: has_attractions={has_attractions}, has_route={has_route}")
    
    # Manual routing logic as a backup if the LLM router fails
    next_agent_backup = "attractions_agent"
    if has_attractions and not has_route:
        next_agent_backup = "route_planner"
    elif has_attractions and has_route:
        next_agent_backup = "finalizer"
    
    # Call the router agent
    try:
        router_agent = create_router_agent()
        result = router_agent.invoke({
            "messages": state["messages"],
            "destination": state["destination"],
            "attractions_status": attractions_status,
            "route_status": route_status
        })
        
        # Get the next agent from the result, with fallback to our backup logic
        next_agent = result.get("next_agent", next_agent_backup)
        logger.info(f"Router agent decided: {next_agent} (backup would be {next_agent_backup})")
        
        # Map "attractions" to "attractions_agent" node name
        if next_agent == "attractions":
            return "attractions_agent"
        return next_agent
    except Exception as e:
        logger.error(f"Error in router agent: {e}")
        # Use our backup routing logic for error cases
        logger.info(f"Using backup routing logic: {next_agent_backup}")
        return next_agent_backup

# Agent execution functions with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_agent_with_retry(agent_func, inputs):
    """Execute an agent function with retry logic"""
    try:
        # Attempt to invoke the agent
        result = agent_func.invoke(inputs)
        
        # Basic validation of the result
        if isinstance(result, dict):
            # For attractions agent, check for attractions list
            if "attractions" in result and not result["attractions"]:
                raise ValueError("Empty attractions list returned")
                
            # For route planner, check for route_plan with day_by_day_itinerary
            if "route_plan" in result and not result.get("route_plan", {}).get("day_by_day_itinerary"):
                raise ValueError("Invalid route plan structure")
                
            # For finalizer, check for final_plan 
            if "final_plan" in result and not result.get("final_plan"):
                raise ValueError("Empty final plan returned")
        
        return result
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        raise

def run_attractions_agent(state: AgentState) -> AgentState:
    """Find tourist attractions for the destination"""
    try:
        attractions_agent = create_attractions_agent()
        logger.info(f"Finding attractions for {state['destination']}")
        result = execute_agent_with_retry(attractions_agent, {
            "messages": state["messages"],
            "destination": state["destination"]
        })
        
        attractions = result.get("attractions", [])
        logger.info(f"Found {len(attractions)} attractions: {[a.get('name', 'Unknown') for a in attractions]}")
        
        # Update the state with attractions
        return {
            **state,
            "attractions": attractions,
            "messages": list(state["messages"]) + [HumanMessage(content=f"Found attractions for {state['destination']}")],
        }
    except Exception as e:
        logger.error(f"Error in attractions agent: {e}")
        return state

def run_route_planner_agent(state: AgentState) -> AgentState:
    """Plan a route between attractions"""
    # Create a more detailed attractions string with explicit numbering
    attractions_list = []
    for i, attr in enumerate(state["attractions"], 1):
        attractions_list.append(f"{i}. {attr['name']}: {attr['description']} ({attr['recommended_time']})")
    
    attractions_details = "\n".join(attractions_list)
    
    try:
        logger.info(f"Planning route for {len(state['attractions'])} attractions in {state['destination']}")
        
        # Add a human message about the attractions to make the context clearer
        state["messages"].append(
            HumanMessage(content=f"I found these {len(state['attractions'])} attractions in {state['destination']}. "
                        f"Please create a route plan for visiting them:\n\n{attractions_details}")
        )
        
        route_agent = create_route_planner_agent()
        result = execute_agent_with_retry(route_agent, {
            "messages": state["messages"],
            "destination": state["destination"],
            "attractions_details": attractions_details
        })
        
        route_plan = result.get("route_plan", {})
        
        # Validate the route plan has the expected structure
        if not route_plan or not route_plan.get("day_by_day_itinerary"):
            logger.warning("Route plan is missing required fields, retrying with more explicit instructions")
            # Try one more time with even more explicit instructions
            state["messages"].append(
                HumanMessage(content="Please create a route plan in the correct JSON format with 'day_by_day_itinerary', "
                            "'transportation_tips', and 'estimated_costs' fields.")
            )
            result = execute_agent_with_retry(route_agent, {
                "messages": state["messages"],
                "destination": state["destination"],
                "attractions_details": attractions_details
            })
            route_plan = result.get("route_plan", {})
        
        logger.info(f"Route plan created with {len(route_plan.get('day_by_day_itinerary', []))} days")
        
        # Update the state with route plan
        return {
            **state,
            "route_plan": route_plan,
            "messages": list(state["messages"]) + [HumanMessage(content=f"Created route plan for {state['destination']}")],
        }
    except Exception as e:
        logger.error(f"Error in route planner agent: {e}")
        return state

def run_finalizer_agent(state: AgentState) -> AgentState:
    """Create final consolidated travel plan"""
    attractions_details = "\n".join([
        f"- {attr['name']}: {attr['description']} ({attr['recommended_time']})"
        for attr in state["attractions"]
    ])
    
    route_details = json.dumps(state["route_plan"], indent=2)
    
    try:
        logger.info(f"Creating final plan for {state['destination']} with {len(state['attractions'])} attractions and route plan")
        finalizer_agent = create_finalizer_agent()
        result = execute_agent_with_retry(finalizer_agent, {
            "messages": state["messages"],
            "destination": state["destination"],
            "attractions_details": attractions_details,
            "route_details": route_details
        })
        
        final_plan = result.get("final_plan", {})
        logger.info(f"Final plan created: {json.dumps(final_plan, indent=2)[:200]}...")
        
        # Update the state with final plan
        new_state = {
            **state,
            "final_plan": final_plan,
            "messages": list(state["messages"]) + [
                HumanMessage(content=f"Finalized travel plan for {state['destination']}")
            ],
        }
        logger.info(f"Final state updated with plan of length {len(json.dumps(new_state['final_plan']))} chars")
        return new_state
    except Exception as e:
        logger.error(f"Error in finalizer agent: {e}")
        return state

# Create the graph
def create_travel_planning_graph() -> StateGraph:
    """Create the travel planning workflow graph"""
    # Define the nodes in the graph
    workflow = StateGraph(AgentState)    # Create a router node that returns the state, not just a string
    def router_wrapper(state: AgentState) -> dict:
        # Check the actual state values to determine what we have
        has_attractions = bool(state["attractions"] and len(state["attractions"]) > 0)
        has_route = bool(state["route_plan"] and len(state["route_plan"].keys()) > 0)
        has_final = bool(state["final_plan"] and len(state["final_plan"].keys()) > 0)
        
        # Log the current state for debugging
        attractions_status = f"Completed ({len(state['attractions'])} found)" if has_attractions else "Not started"
        route_status = "Completed" if has_route else "Not started"
        final_status = "Completed" if has_final else "Not started"
        
        logger.info(f"Router state: Destination={state['destination']}, " 
                   f"Attractions={attractions_status}, Route={route_status}, Final={final_status}")
        
        # Add more information to help the router make a decision
        if not has_attractions:
            state["messages"].append(HumanMessage(content="We need to find attractions for this destination first."))
        elif not has_route:
            state["messages"].append(HumanMessage(content="We have attractions, now we need a route plan."))
        elif not has_final:
            state["messages"].append(HumanMessage(content="We have attractions and route plan, now we need to finalize."))
        
        # Get the next node name from route_agent - the conditional edges will use this
        next_node = route_agent(state)
        logger.info(f"Router decided next node: {next_node}")
        
        # Return the state with possibly modified messages
        return state
        
    # Add the nodes
    workflow.add_node("router", router_wrapper)
    workflow.add_node("attractions_agent", run_attractions_agent)
    workflow.add_node("route_planner", run_route_planner_agent)
    workflow.add_node("finalizer", run_finalizer_agent)
      # Add conditional edges using the original route_agent to determine the next node
    workflow.add_conditional_edges(
        "router",
        {
            "attractions_agent": lambda state: route_agent(state) == "attractions_agent", 
            "route_planner": lambda state: route_agent(state) == "route_planner",
            "finalizer": lambda state: route_agent(state) == "finalizer",
        }
    )
    
    # Always connect attraction agent's output back to router for next decision
    workflow.add_edge("attractions_agent", "router")
    
    # Always connect route planner's output back to router for next decision
    workflow.add_edge("route_planner", "router")
    
    # Connect finalizer to END only when it's done
    # Do not automatically end - force it to be the router's decision
    workflow.add_edge("finalizer", END)
    
    # Set the entry point
    workflow.set_entry_point("router")
    
    return workflow.compile()

def plan_travel(destination: str, debug_mode: bool = False) -> Dict:
    """
    Main function to plan travel for a given destination
    
    Args:
        destination: Travel destination to plan for
        debug_mode: Whether to include debug information in the output
        
    Returns:
        Dictionary containing the final travel plan
    """
    # Initialize state
    state = {
        "messages": [HumanMessage(content=f"I want to plan a trip to {destination}")],
        "destination": destination,
        "attractions": [],
        "route_plan": {},
        "final_plan": {}
    }
    
    try:
        logger.info(f"Starting travel planning for {destination}")
        
        
        # Step 1: Get attractions
        logger.info("Finding attractions")
        state = run_attractions_agent(state)
        logger.info(f"After attractions step: {len(state['attractions'])} attractions found")
        
        # Check if we have attractions before proceeding
        if not state["attractions"] or len(state["attractions"]) == 0:
            logger.warning("No attractions found, stopping workflow")
            return {
                "destination": destination,
                "error": "No attractions could be found for this destination",
                "attractions": [],
                "route_plan": {}
            }
        
        # Step 2: Plan route
        logger.info("Planning route")
        state = run_route_planner_agent(state)
        logger.info(f"After route planning: route keys: {state['route_plan'].keys()}")
        
        # Check if we have a route before proceeding
        if not state["route_plan"] or len(state["route_plan"]) == 0:
            logger.warning("No route plan created, stopping workflow")
            return {
                "destination": destination,
                "error": "Could not create a route plan",
                "attractions": state["attractions"],
                "route_plan": {}
            }
        
        # Step 3: Generate final plan
        logger.info("Creating final plan")
        state = run_finalizer_agent(state)
        logger.info(f"After finalization: final plan keys: {state['final_plan'].keys() if state['final_plan'] else 'None'}")
        
        # Log the final result for debugging
        logger.info("Travel planning complete")
            
        # Return full result if in debug mode, otherwise just the travel plan
        if debug_mode:
            return {
                "destination": destination,
                "plan": state["final_plan"],
                "debug": {
                    "attractions_count": len(state["attractions"]),
                    "has_route_plan": bool(state["route_plan"]),
                    "has_final_plan": bool(state["final_plan"]),
                    "attractions": state["attractions"],
                    "route_plan": state["route_plan"]
                }
            }
        else:
            return {
                "destination": destination,
                "plan": state["final_plan"]
            }
    except Exception as e:
        logger.error(f"Error planning travel: {e}")
        return {
            "destination": destination,
            "error": str(e),
            "attractions": state.get("attractions", []),
            "route_plan": state.get("route_plan", {})
        }

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Plan a trip using AI agents")
    parser.add_argument("destination", help="Travel destination to plan for")
    parser.add_argument("--output", "-o", help="Path to save the travel plan output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Include debug information in output")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum number of steps in the agent workflow")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Check environment variables
        missing_vars = []
        for env_var in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]:
            if not os.getenv(env_var):
                missing_vars.append(env_var)
        
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please set these variables in your .env file or environment")
            return
            
        print(f"Planning a trip to {args.destination}...")
        print(f"Using Azure OpenAI deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")
        
        # Set max steps for the graph to prevent infinite loops
        os.environ["LANGGRAPH_MAX_STEPS"] = str(args.max_steps)
        
        # Run the travel planning process
        result = plan_travel(args.destination, debug_mode=args.debug)
        
        # Save to file if requested
        if args.output:
            logger.info(f"Saving travel plan to {args.output}")
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Print result
        print("\nTRAVEL PLAN:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Check if the plan is empty
        if not result.get("plan") and not result.get("error"):
            print("\nWarning: The travel plan is empty. This may indicate:")
            print("- The agent workflow didn't complete properly")
            print("- The environment variables might be incorrect")
            print("- The Azure OpenAI service might have issues")
            print("\nTry running with --debug to see more information")
        
        if args.output:
            print(f"\nTravel plan saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Failed to generate travel plan: {e}")

if __name__ == "__main__":
    main()