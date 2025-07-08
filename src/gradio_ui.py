import os
import re
import shutil
import gradio as gr

from typing import Optional
from smolagents.agents import MultiStepAgent
from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.utils import _is_package_available


def stream_to_gradio(
        agent,
        task: str,
        reset_agent_memory: bool = False,
        additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )

    print(f"👤 User: ", task)

    # synchronously invoke the agent
    final_answer = agent.invoke(task)
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")


async def async_stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )

    print(f"👤 User: ", task)
    partial_response = ""
    try:
        async for chunk in agent.invoke_stream(task):
            partial_response += chunk
            yield gr.ChatMessage(role="assistant", content=partial_response)
    except Exception as e:
        yield gr.ChatMessage(role="assistant", content=f"Error during streaming: {str(e)}")


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages, session_state):

        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            partial_response = ""
            for chunk in session_state["agent"].invoke(prompt):
                partial_response += chunk
                if messages and messages[-1].role == "assistant":
                    messages[-1].content = partial_response  # Update last assistant message
                else:
                    messages.append(gr.ChatMessage(role="assistant", content=partial_response))
                yield messages

            yield messages
        except Exception as e:
            messages.append(gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"))
            yield messages

    async def async_interact_with_agent(self, prompt, messages, session_state):

        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            partial_response = ""
            async for chunk in session_state["agent"].invoke_stream(prompt):
                partial_response += chunk
                if messages and messages[-1].role == "assistant":
                    messages[-1].content = partial_response  # Update last assistant message
                else:
                    messages.append(gr.ChatMessage(role="assistant", content=partial_response))
                yield messages

            yield messages
        except Exception as e:
            messages.append(gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"))
            yield messages

    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt"]

        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        import gradio as gr

        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
            gr.Button(interactive=False),
        )

    def launch(self, share: bool = True, **kwargs: object) -> None:
        import gradio as gr

        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ')}"
                    "\n> This web ui allows you to interact with a **RACRA**: A `LangGraph` multi-agent conversational system equipped role-aware RAG and several other tools. \nThis role-sensitive agent is designed to streamline access to critical information for `Architects`, `Engineers`, and `Project Managers`."
                    + (f"\n\n**Agent description:**\n{self.description}\n\n" if self.description else "")
                )

                with gr.Group():
                    gr.Markdown("**Select your role**", container=True)
                    with gr.Row():
                        architect_btn = gr.Button("👷 Architect", variant="secondary")
                        engineer_btn = gr.Button("🔧 Engineer", variant="secondary")
                        pm_btn = gr.Button("📊 Project Manager", variant="secondary")
                    
                    gr.Markdown("**Your request**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                    
                    # Role button click handlers
                    def select_role(role_name, messages):
                        role_message = f"{role_name}, "
                        messages.append(gr.ChatMessage(role="system", content=role_message))
                        return messages, role_message
                    
                    architect_btn.click(
                        fn=lambda messages: select_role("Architect", messages),
                        inputs=[stored_messages],
                        outputs=[stored_messages, text_input]
                    )
                    
                    engineer_btn.click(
                        fn=lambda messages: select_role("Engineer", messages),
                        inputs=[stored_messages],
                        outputs=[stored_messages, text_input]
                    )
                    
                    pm_btn.click(
                        fn=lambda messages: select_role("Project Manager", messages),
                        inputs=[stored_messages],
                        outputs=[stored_messages, text_input]
                    )

                # If an upload folder is provided, enable the upload feature
                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )

                gr.HTML("<br><br><h4><center>Made by:</center></h4>")
                with gr.Row():
                    gr.HTML("""
                            <h6 style='text-align: center;'>
                                Sebastian Laverde Alfonso<br>
                                <a href="https://github.com/LaverdeS" target="_blank">GitHub</a> |
                                <a href="mailto:lavmlk20201@gmail.com">Email</a>
                            </h6>
                        """)
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
            )

            # Set up event handlers
            streaming = self.agent.streaming if hasattr(self.agent, 'streaming') else False
            agent_interaction_method = self.interact_with_agent if not streaming else self.async_interact_with_agent

            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(agent_interaction_method, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(agent_interaction_method, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

        demo.launch(debug=True, share=share, **kwargs)


__all__ = ["stream_to_gradio", "async_stream_to_gradio", "GradioUI"]
