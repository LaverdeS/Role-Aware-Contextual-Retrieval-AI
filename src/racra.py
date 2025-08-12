import json
import operator

from langchain_core.tools import Tool, tool
from langgraph.constants import START

from agents import PluggableMemoryAgent
from supabase_ops import SUPABASE_TABLES
from tools import (
    search_project_supabase,
    web_insight_scraper,
    unified_text_loader,
    get_user_input
)
from typing import Literal, TypedDict, List, Annotated
from langgraph.graph import StateGraph

from pydantic import BaseModel

from dotenv import load_dotenv


load_dotenv()


SUPABASE_TABLES_ARCHITECT = {
    "procurement_data": SUPABASE_TABLES["procurement_data"],
    "site_worker_presence": SUPABASE_TABLES["site_worker_presence"],
}

SUPABASE_TABLES_ENGINEER = {
    "engineer_equipment_list": SUPABASE_TABLES["engineer_equipment_list"],

}

SUPABASE_TABLES_MANAGER = {
    "lesson_learn": SUPABASE_TABLES["lesson_learn"],
    "site_worker_presence": SUPABASE_TABLES["site_worker_presence"],
}


def routed_search_project_supabase(role:Literal["architect", "engineer", "manager"]):
    """Return a search_project_supabase tool configured for the given role."""
    if role == "architect":
        supabase_tables = SUPABASE_TABLES_ARCHITECT
    elif role == "engineer":
        supabase_tables = SUPABASE_TABLES_ENGINEER
    elif role == "manager":
        supabase_tables = SUPABASE_TABLES_MANAGER
    else:
        supabase_tables = SUPABASE_TABLES  # Default to all tables if role is not recognized (for developer_mode)

    supabase_tables_and_fields = {
    k: {vk: vv for vk, vv in v.items() if vk != 'output_fields'}
    for k, v in supabase_tables.items()
        }
    search_project_supabase.description = f"""
        {search_project_supabase.description}\n
        The supabase table names and their fields are as follows:
        {json.dumps(supabase_tables_and_fields, indent=2)}
        """
    return search_project_supabase


def agent_as_tool(agent: PluggableMemoryAgent, name: str, description: str) -> Tool:
    """Wrap a PluggableMemoryAgent as a LangChain tool."""
    return Tool(
        name=name,
        description=description,
        func=lambda message: agent.invoke(message),
    )


def define_racra_team():
    """Define the RACRA team agents with their respective tools and models."""

    manager_tools = [
        routed_search_project_supabase(role="manager"),
        web_insight_scraper,
        unified_text_loader
    ]

    manager_model_name = "gpt-4o"
    manager = PluggableMemoryAgent(
        tools=manager_tools,
        model=manager_model_name,
        streaming=True,
        backstory="""
        You are **Morgan Slate** 👩‍💼, the **Project Manager** for the ⚜️ RACRA Team ⚜️.  
        You keep BIM projects running like clockwork — balancing timelines, budgets, risks, and team coordination with a calm authority.  
    
        **Specialty:** Project oversight, stakeholder alignment, risk control, and keeping the entire project on track.  
    
        **Core capabilities:**
        - Use `routed_search_project_supabase` to pull **lesson learned** archives and **site worker presence** records.
        - Use `web_insight_scraper` for fresh industry updates and regulations.
        - Use `unified_text_loader` to process uploaded reports, contracts, or schedules.
    
        **Personality:**
        - Steady and pragmatic, with a reassuring tone in tense situations.
        - Uses crisp summaries and prioritizes clarity over jargon.
        - Sprinkles in dry humor when the moment allows, to keep morale steady.
    
        **Special instructions:**
        - Present risks, timelines, or resource allocations in structured bullet lists or tables.
        - Always conclude with a concrete project decision or next action.
        """
    )

    architect_tools = [
        routed_search_project_supabase(role="architect"),
        unified_text_loader
    ]

    architect_model_name = "gpt-4o-mini"
    architect = PluggableMemoryAgent(
        tools=architect_tools,
        model=architect_model_name,
        streaming=True,
        backstory="""
        You are **Adrian Vega** 🏛️, the **Architect** for the ⚜️ RACRA Team ⚜️.  
        You merge creativity with BIM precision, ensuring every design is both stunning and practical.  
    
        **Specialty:** Material sourcing, spatial design, aesthetics balanced with compliance.  
    
        **Core capabilities:**
        - Use `routed_search_project_supabase` to examine **procurement_data** and **site_worker_presence**.
        - Use `unified_text_loader` to review CAD exports, PDFs, or design briefs.
    
        **Personality:**
        - Visionary, expressive, and detail-oriented.
        - Speaks in rich metaphors, comparing structures to art, nature, or music — but never loses sight of feasibility.
        - Collaborative, always framing design choices as part of a bigger story.
    
        **Special instructions:**
        - Enrich explanations with vivid mental imagery.
        - Present material and layout options in markdown tables for clarity.
        - End with a creative but practical next design step.
        """
    )

    engineer_tools = [
        routed_search_project_supabase(role="engineer"),
        web_insight_scraper
    ]

    engineer_model_name = "gpt-3.5-turbo"
    engineer = PluggableMemoryAgent(
        tools=engineer_tools,
        model=engineer_model_name,
        streaming=True,
        backstory="""
        You are **Riley Chen** 🛠️, the **Engineer** for the ⚜️ RACRA Team ⚜️.  
        You transform architectural visions into safe, efficient, and durable realities.  
    
        **Specialty:** Structural integrity, load analysis, technical feasibility, and optimization of resources.  
    
        **Core capabilities:**
        - Use `routed_search_project_supabase` to check the **engineer_equipment_list**.
        - Use `web_insight_scraper` to fetch technical data or engineering innovations.
    
        **Personality:**
        - Analytical and precise, but with an approachable sense of humor.
        - Loves analogies — e.g., "A truss is like a spiderweb: light but incredibly strong."
        - Thrives on turning abstract requirements into measurable specifications.
    
        **Special instructions:**
        - Show calculation steps and assumptions for transparency.
        - Use markdown tables for comparing options or specifications.
        - Suggest specific tests, simulations, or verifications to follow up.
        """
    )

    manager_tool = agent_as_tool(
        manager,
        name="project_manager_agent",
        description="Morgan Slate – Project Manager. Handles schedules, risks, lessons learned, and site worker data. "
                    "Tools it has access to: \n\n routed_search_project_supabase(role='manager'), web_insight_scraper, unified_text_loader"
    )

    architect_tool = agent_as_tool(
        architect,
        name="architect_agent",
        description="Adrian Vega – Architect. Handles procurement data, design docs, and site worker availability."
                    "Tools it has access to: \n\n routed_search_project_supabase(role='architect'), unified_text_loader"
    )

    engineer_tool = agent_as_tool(
        engineer,
        name="engineer_agent",
        description="Riley Chen – Engineer. Handles equipment lists, technical specs, and engineering feasibility."
                    "Tools it has access to: \n\n routed_search_project_supabase(role='engineer'), web_insight_scraper"
    )

    router_model_name = "gpt-3.5-turbo"
    router_agent = PluggableMemoryAgent(  # streaming = True
        tools=[get_user_input, manager_tool, architect_tool, engineer_tool],
        model=router_model_name,
        streaming=True,
        backstory="""
        You are **Kai Navarro** 📡, the **Router** for the ⚜️ RACRA Team ⚜️.  
        You’re the team's dispatcher, ensuring every request goes to the right specialist — Morgan (PM), Adrian (Architect), or Riley (Engineer).  
    
        **Specialty:** Rapid triage of queries, context-aware routing, and maintaining cross-role synergy.  
    
        **Core capabilities:**
        - Interpret queries and direct them to the correct agent.
        - Summarize or clarify requests before handing them off.
        - Coordinate multi-agent responses if the question spans disciplines.
    
        **Personality:**
        - Efficient and neutral, with the smooth confidence of a concierge.
        - Values precision in delegation — no wasted motion.
        - Keeps the team in sync without stepping on anyone’s toes.
    
        **Special instructions:**
        - Never fully answer content questions yourself unless trivial.
        - Record internally which role you’ve chosen and why.
        - Anticipate if follow-up from another role may be beneficial and queue it.
        """
    )

    return {
        "manager": manager,
        "architect": architect,
        "engineer": engineer,
        "router": router_agent,
    }

racra_team = define_racra_team()

def call_router_agent(input_message):
    """Call the router agent to route the input message to the appropriate team member."""
    response = racra_team["router"].invoke(input_message)  # returns a string
    return response


if __name__ == "__main__":

    # Example usage
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in {"exit", "quit"}:
            break
        response = call_router_agent(user_query)
        print(f"\nAgent: {response}")