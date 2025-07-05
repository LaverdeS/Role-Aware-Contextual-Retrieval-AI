import os
import logging
import gradio as gr
import gradio_ui

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.globals import set_debug
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    search_project_supabase,
    web_insight_scraper,
    unified_text_loader
)
from fire import Fire
from utils import print_conversation


set_debug(False)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
load_dotenv()


class PluggableMemoryAgent:
    """A pluggable memory agent that can use various tools and has memory capabilities to keep the context of through session interactions."""
    def __init__(self, tools: list, model:str="gpt-4o"):
        try:
            self.llm = AzureChatOpenAI(                
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_type="azure",
                temperature=0.1,
                max_tokens=500,)
        except Exception as e:
            logging.error(f"Failed to initialize LLM with model {model}: {str(e)}")
            raise ModuleNotFoundError(f"Failed to initialize LLM with model = {model}. Please check the model name and your OpenAI API key.")

        self.name = "TestAI: Pluggable Memory Agent with Generic Role"
        self.description = "An agent that can use various tools and has memory capabilities to keep the context of through session interactions."

        # Plug‑and‑play memory
        self.checkpointer = MemorySaver()  # MemorySaver is a short-term memory that keeps track of the session context.

        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            checkpointer=self.checkpointer,
            prompt="""
            You are Ethan... Ethan Bright 👨‍💼, a human who only talks to humans (no need to say this unless asked).
            You’re part of the RACRA Team 🪽: a smart, role-aware AI for Building Information Modeling.
            You help Architects, Engineers, and Project Managers by connecting data and using sharp, context-driven reasoning to boost productivity and decision-making.
            Your tone is friendly and professional, sprinkled with emojis, and you know how to crack a light joke when the moment’s right—keeping things both helpful and approachable.
            Uniquely, you love using clever analogies to explain complex ideas clearly and memorably.
            """,  # Give backstory prompt to the agent (SystemMessage)
            debug=False,
        )


    def chat(self, message, thread_id: str = "default") -> str:
        """Send a message to the agent and get a response."""
        human_message = HumanMessage(content=message)
        print_conversation([{"role": "user", "content": human_message}])
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke({"messages": [human_message]}, config=config)
        response = result["messages"][-1].content
        print_conversation([{"role": "assistant", "content": response}])
        return response


    def invoke(self, message: str, reset_agent_memory: bool = False, thread_id: str = "default") -> str:
        """Invoke the agent with a task."""
        if reset_agent_memory:
            self.checkpointer.delete_thread(thread_id)
        response = self.chat(message, thread_id=thread_id)
        return response


def agents_execution(execution_mode: str):
    """Run the agent in the specified execution mode."""
    if execution_mode not in ["demo", "ui"]:
        raise ValueError("Invalid execution mode. Use 'demo' or 'ui'.")

    logging.info(f"Execution mode set to: {execution_mode}")

    tools = [
        search_project_supabase,
        web_insight_scraper,
        unified_text_loader
    ]
    logging.info(f"Using tools:\n")

    BOLD = "\033[1m"
    END = "\033[0m"

    for tool in tools:
        logging.debug(f"Tool: {BOLD}{tool.name}{END}\n{tool.description}\n")

    if execution_mode == "demo":
        model_name = "gpt-4o-mini"
        agent_with_memory = PluggableMemoryAgent(
            tools=tools,
            model=model_name
        )
        logging.info(f"Running in demo mode using {model_name}. Use 'ui' for web UI mode.")

        # general
        _ = agent_with_memory.invoke("Hi!")
        _ = agent_with_memory.invoke("What is the latest research on AI in construction (today is July 2025)?")
        agent_with_memory.invoke("What is the latest research on AI in BIM?")
        _ = agent_with_memory.invoke("What have we talked about?")

        # supabase
        _ = agent_with_memory.invoke("Share with me the total count of the available equipment")
        _ = agent_with_memory.invoke("construction equipment")

        conversation_history = agent_with_memory.checkpointer.get(config={"configurable": {"thread_id": "default"}})
        logging.debug(f"\n\nConversation history keys: {list(conversation_history.keys())}")


    elif execution_mode == "ui":
        model_name = "gpt-4o"
        agent_with_memory = PluggableMemoryAgent(
            tools=tools,
            model=model_name
        )
        logging.info(f"Running in web UI mode using {model_name}. Use 'demo' for demo mode.")
        gradio_ui.GradioUI(agent_with_memory).launch()


if __name__ == "__main__":
    Fire(agents_execution)
