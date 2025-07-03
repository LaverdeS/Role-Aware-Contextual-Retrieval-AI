import json
import inspect
import shutil

from langchain_core.messages import HumanMessage, AIMessage

from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown
from rich.json import JSON

from typing import Callable, Union


# helper functions
def print_conversation(messages: list[dict[str, Union[str, list, dict]]]):
    """Prints a formatted conversation using Rich library."""
    console = Console(width=200, soft_wrap=True)

    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")

        try:
            if isinstance(content, dict):
                output = content.get('output')
                if isinstance(output, HumanMessage):
                    content = output.content

            elif isinstance(content, (HumanMessage, AIMessage)):
                content = content.content

            if isinstance(content, (dict, list)):
                rendered_content = JSON.from_data(content)
            else:  # most cases will be strings
                rendered_content = Markdown(str(content).strip())

        except Exception as e:
            rendered_content = Markdown(f"[Error rendering content: {e}]")

        role_colors = {
            "Assistant": "magenta",
            "User": "green",
            "System": "blue",
            "Tool": "yellow",
            "Token": "white"
        }
        border_style_color = role_colors.get(role, "red")

        panel = Panel(
            rendered_content,
            title=f"[bold blue]{role}[/]",
            border_style=border_style_color,
            expand=True,
        )

        console.print(panel)


def print_tool_call(tool: Callable, tool_name: str, args: dict):
    """Prints the tool call for debugging purposes."""
    sig = inspect.signature(tool)
    print_conversation(
        messages=[
            {
                'role': 'Tool-Call',
                'content': f"Calling `{tool_name}`{sig}"
            },
            {
                'role': 'Tool-Args',
                'content': args
            }
        ],
    )


def print_tool_response(response: Union[str, list, dict]):
    """Prints the tool response for debugging purposes."""
    print_conversation(
        messages=[
            {
                'role': 'Tool-Response',
                'content': response
            }
        ],
    )