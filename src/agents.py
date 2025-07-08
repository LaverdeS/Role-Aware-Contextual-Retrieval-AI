import json
import logging
import gradio as gr
import asyncio
import gradio_ui

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
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
    def __init__(self, tools: list, model:str="gpt-4o", streaming: bool = True):
        self.streaming = streaming
        try:
            self.llm = ChatOpenAI(model=model, temperature=0, max_retries=3, streaming=streaming)
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
            You’re part of the ⚜️️ RACRA Team ⚜️: a smart, role-aware AI for Building Information Modeling.
            You help Architects, Engineers, and Project Managers by connecting data and using sharp, context-driven reasoning to boost productivity and decision-making.
            Your tone is friendly and professional, you use emojis fitting the context, and you know how to crack a light joke when the moment’s right—keeping things both helpful and approachable.
            Uniquely, you love using clever analogies to explain complex ideas clearly and memorably.
            When asked about yourself, you should paraphrase and fit to the context of the conversation.
            Also use markdown formatting and backticks when you find suitable to make your responses more readable, engaging (bold, italics, ...) 
            and to highlight important information, concepts, roles (such as **Engineer**, **Architect**, **Project Manager**), variables, steps, etc.
            
            Whenever you decide to use a database search/retrieval tool, provide the table with all the field/columns available in the tool's response.
            Whenever you seem fitted to answer a question using a table format, go ahead and do so. The table should be formatted in Markdown.
            You should follow the user's formatting preferences if they have specified any in their message.
            NEVER end your response with an open message like:
             - 'If you need more details or have any other questions, feel free to ask! ...'
             - 'If there's anything else you'd like to explore or need assistance with, just let me know!'
             - 'If there's a specific task or question you have in mind, feel free to ask!'
            or similar; instead, think ahead and provide one of: OR(genera suggestion, query suggestion, next step) that the user might find useful based on the context of the conversation and the tools available to you.
            
            Thanks to the tool `unified_text_loader` you have the capability to interpret images directly passing a `file_path` to the tool. The same tool can be used to load text files, PDFs, and other document formats.
            """,  # Give backstory prompt to the agent (SystemMessage)
            debug=False,
        )


    def invoke(self, message: str, reset_agent_memory: bool = False, thread_id: str = "default") -> str:
        """Invoke the agent with a task. Send a message to the agent and get a response."""
        if reset_agent_memory:
            self.checkpointer.delete_thread(thread_id)

        human_message = HumanMessage(content=message)
        print_conversation([{"role": "user", "content": human_message}])

        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke({"messages": [human_message]}, config=config)
        response = result["messages"][-1].content
        print_conversation([{"role": "assistant", "content": response}])

        return response


    async def invoke_stream(self, message: str, reset_agent_memory: bool = False, thread_id: str = "default"):
        """Stream the agent's response token-by-token."""
        if reset_agent_memory:
            self.checkpointer.delete_thread(thread_id)

        human_message = HumanMessage(content=message)
        print_conversation([{"role": "user", "content": human_message}])
        config = {"configurable": {"thread_id": thread_id}}

        stream = self.agent.astream_events(
            {"messages": [human_message]}, config=config, version="v2"
        )

        async for event in stream:
            kind = event["event"]
            data = event.get("data") or {}
            chunk = data.get("chunk") or data.get("output") or ""
            name = event.get("name", "Unnamed Chain")

            #yield f"{kind}: {name}"
            await asyncio.sleep(0.02)

            if kind == "on_chat_model_stream":
                yield chunk.content  # this makes the streaming effect; must be str, objs will fail

            elif kind == "on_chain_end" and name == "call_model":
                if chunk["messages"][-1].additional_kwargs.get("tool_calls", False):
                    latest_tool_call = chunk["messages"][-1].additional_kwargs["tool_calls"][-1]
                    latest_tool_name = latest_tool_call['function']['name']
                    latest_tool_args = latest_tool_call['function']['arguments']
                    yield f"\n🛠️ **{latest_tool_name}**\n"
                    yield f"```json\n{json.dumps(json.loads(latest_tool_args), indent=2)}\n```\n"

        await asyncio.sleep(0.05)


async def run_stream(agent, prompt):
    """Helper methods for running the agent and stream the response."""
    async for chunk in agent.invoke_stream(prompt):
        print(chunk, end="", flush=True)


def agents_execution(execution_mode: str = "ui", streaming: bool = True):
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
        agent_with_memory = PluggableMemoryAgent(  # streaming = True
            tools=tools,
            model=model_name,
            streaming=streaming
        )
        logging.info(f"Running in demo mode using {model_name}. Use 'ui' for web UI mode.")

        if  not streaming:
            logging.info("Streaming mode is disabled. This will return the full response at once.")

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

        else:
            logging.info("Streaming mode is enabled. This will stream the response token-by-token.")

            query = "Hi!"
            asyncio.run(run_stream(agent_with_memory, query))

            # supabase
            query = "What are all the Contruction Equipment available in the projects supabase database?"
            asyncio.run(run_stream(agent_with_memory, query))

    elif execution_mode == "ui":
        model_name = "gpt-4o-mini"
        agent_with_memory = PluggableMemoryAgent(
            tools=tools,
            model=model_name,
            streaming=streaming
        )
        logging.info(f"Running in web UI mode using {model_name}. Use 'demo' for demo mode.")

        gradio_ui.GradioUI(agent_with_memory).launch()


if __name__ == "__main__":
    Fire(agents_execution)
