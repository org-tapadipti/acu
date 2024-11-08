import json
import logging
import os
import time
import traceback
from typing import Final, List, Optional, Tuple, Type, cast, Any
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO

from agentdesk.device_v1 import Desktop
from devicebay import Device
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState
from skillpacks.server.models import V1ActionSelection
from surfkit.agent import TaskAgent
from taskara import Task, TaskStatus
from tenacity import before_sleep_log, retry, stop_after_attempt
from threadmem import RoleMessage, RoleThread
from mllm import Prompt
from toolfuse.util import AgentUtils

from anthropic import (
    Anthropic,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tool import SemanticDesktop, router
from .anthropic import ComputerTool, ToolCollection, ToolResult

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))

console = Console(force_terminal=True)

tool_collection = ToolCollection(
        ComputerTool(),
    )
print("API KEY: ", os.environ["ANTHROPIC_API_KEY"])
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

tools_mapping = {
    "key": "hot_key",
    "type": "type_text",
    "mouse_move": "move_mouse",
    "left_click": "click",
    "left_click_drag": "drag_mouse",
    "right_click": "N/A",
    "middle_click": "N/A",
    "double_click": "double_click",
    "screenshot": "take_screenshots",
    "cursor_position": "mouse_coordinates",
}

class SurfPizzaConfig(BaseModel):
    pass


class SurfPizza(TaskAgent):
    """A GUI desktop agent that slices up the image"""

    def solve_task(
        self,
        task: Task,
        device: Optional[Device] = None,
        max_steps: int = 30,
    ) -> Task:
        """Solve a task

        Args:
            task (Task): Task to solve.
            device (Device): Device to perform the task on.
            max_steps (int, optional): Max steps to try and solve. Defaults to 30.

        Returns:
            Task: The task
        """

        # Post a message to the default thread to let the user know the task is in progress
        task.post_message("assistant", f"Starting task '{task.description}'")

        # Create threads in the task to update the user
        console.print("creating threads...")
        task.ensure_thread("debug")
        task.post_message("assistant", "I'll post debug messages here", thread="debug")

        # Check that the device we received is one we support
        if not isinstance(device, Desktop):
            raise ValueError("Only desktop devices supported")

        # Wrap the standard desktop in our special tool
        semdesk = SemanticDesktop(task=task, desktop=device)
        computer = ComputerTool()

        # Add standard agent utils to the device
        semdesk.merge(AgentUtils())

        # # Open a site if present in the parameters
        # site = task._parameters.get("site") if task._parameters else None
        # if site:
        #     console.print(f"▶️ opening site url: {site}", style="blue")
        #     task.post_message("assistant", f"opening site url {site}...")
        #     semdesk.desktop.open_url(site)
        #     console.print("waiting for browser to open...", style="blue")
        #     time.sleep(5)

        # Get info about the desktop
        info = semdesk.desktop.info()
        screen_size = info["screen_size"]
        console.print(f"Desktop info: {screen_size}")

        # Get the json schema for the tools, excluding actions that aren't useful
        # tools = semdesk.json_schema(
        #     exclude_names=[
        #         "move_mouse",
        #         "click",
        #         "drag_mouse",
        #         "mouse_coordinates",
        #         "take_screenshot",
        #         "open_url",
        #         "double_click",
        #     ]
        # )
        # console.print("tools: ", style="purple")
        # console.print(JSON.from_data(tools))

        SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
        * You are utilising an Linux virtual machine of screen size {screen_size} with internet access.
        * To open firefox, please just click on the web browser icon.  Note, firefox-esr is what is installed on your system.
        * When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
        * When using your computer function calls, they take a while to run and send back to you.
        * The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
        </SYSTEM_CAPABILITY>

        <IMPORTANT>
        * When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
        * If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
        </IMPORTANT>"""

        self.system = BetaTextBlockParam(
            type="text",
            text=f"{SYSTEM_PROMPT}",
        )

        # Create our thread and start with a system prompt
        thread = RoleThread()
        messages = []
        thread.post(
            role="user",
            msg=(
                f"Y{SYSTEM_PROMPT}"
                f"Your current task is {task.description}."
            ),
        )

        messages.append(
            {
                "role": "user",
                "content": [BetaTextBlockParam(type="text", text=task.description)],
            })
        # response = router.chat(thread, namespace="system")
        # console.print(f"system prompt response: {response}", style="blue")
        # thread.add_msg(response.msg)

        # Loop to run actions
        max_steps = 30
        print(f"\n\nMax steps: {max_steps}")

        for i in range(max_steps):
            console.print(f"-------step {i + 1}", style="green")

            try:
                # for message in thread.messages():
                #     print(f"\n\n\nStill in solve task. Looping through messages in thread. First message = {message}")
                #     break
                thread, messages, done = self.take_action(semdesk, computer, messages, task, thread)
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.save()
                task.post_message("assistant", f"❗ Error taking action: {e}")
                return task

            if done:
                console.print("task is done", style="green")
                return task

            time.sleep(2)

        task.status = TaskStatus.FAILED
        task.save()
        task.post_message("assistant", "❗ Max steps reached without solving task")
        console.print("Reached max steps without solving task", style="red")

        return task

    @retry(
        stop=stop_after_attempt(1),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def take_action(
        self,
        semdesk: SemanticDesktop,
        computer: ComputerTool,
        messages: list[BetaMessageParam],
        task: Task,
        thread: RoleThread,
    ) -> Tuple[RoleThread, bool]:
        """Take an action

        Args:
            desktop (SemanticDesktop): Desktop to use
            task (str): Task to accomplish
            thread (RoleThread): Role thread for the task

        Returns:
            bool: Whether the task is complete
        """
        try:
            # Check to see if the task has been cancelled
            if task.remote:
                task.refresh()
            console.print("task status: ", task.status.value)
            if (
                task.status == TaskStatus.CANCELING
                or task.status == TaskStatus.CANCELED
            ):
                console.print(f"task is {task.status}", style="red")
                if task.status == TaskStatus.CANCELING:
                    task.status = TaskStatus.CANCELED
                    task.save()
                return thread, messages, True

            console.print("taking action...", style="white")

            # Create a copy of the thread, and remove old images
            _thread = thread.copy()
            # _thread.remove_images()

            # # Take a screenshot of the desktop and post a message with it
            # screenshot_img = semdesk.desktop.take_screenshots()[0]
            # console.print(f"screenshot img type: {type(screenshot_img)}")
            # task.post_message(
            #     "assistant",
            #     "current image",
            #     images=[screenshot_img],
            #     thread="debug",
            # )

            # # Get the current mouse coordinates
            # x, y = semdesk.desktop.mouse_coordinates()
            # console.print(f"mouse coordinates: ({x}, {y})", style="white")

            # # Craft the message asking the MLLM for an action
            # msg = RoleMessage(
            #     role="user",
            #     text=(
            #         "Here is a screenshot of the current desktop, please select an action from the provided schema."
            #         "Please return just the raw JSON"
            #     ),
            #     images=[screenshot_img],
            # )
            # _thread.add_msg(msg)

            # Make the action selection
            # response = router.chat(
            #     _thread,
            #     namespace="action",
            #     # expect=V1ActionSelection,
            #     agent_id=self.name(),
            # )

            # # messages_to_send = []
            # msgs = _thread.messages()
            # # msgs = [_thread.messages()[0], _thread.messages()[-1]]
            # idx = 0
            # for message in msgs:#_thread.messages():
            #     idx += 1
            #     print(f"\n\n\nLooping through messages in thread. Message #{idx} = {message}")
            #     r = message.role
            #     print(f"\n\nr: {r}")
            #     content = []

            #     if r == "assistant":
            #         print("in assistant message")

            #         if message.metadata["type"] == "text":
            #             content.append({
            #                 "type": message.metadata["type"],
            #                 "text": message.text
            #             })
            #         elif message.metadata["type"] == "tool_use":
            #             content.append({
            #                 "type": message.metadata["type"],
            #                 "id":message.metadata["id"],
            #                 "name":message.metadata["name"],
            #                 "input":message.metadata["input"],
            #             })

            #     else:
            #         if message.images:
            #             i = message.images[0]
            #             if i:
            #                 comma_index = i.index(',')
            #                 base64_image = i[comma_index+1:]
            #                 # img_bytes = i.tobytes()
            #                 # base64_image = base64.b64encode(img_bytes).decode('utf-8')
                            
            #                 c = {"type": "image", 
            #                     "source": {
            #                         "type": "base64",
            #                         "media_type": "image/png",
            #                         "data": base64_image,
            #                     }}
                            
            #                 if message.metadata and message.metadata.get("tool_use_id"):
            #                     print("in here")
            #                     content.append({
            #                         "type": message.metadata["type"],
            #                         "tool_use_id": message.metadata["tool_use_id"],
            #                         "content": [c]
            #                     })
            #                 else:
            #                     content.append(c)

            #         else:
            #             if message.text:
            #                 content.append({"type": "text", "text": message.text})


            #     msg_to_send = {"role": r, "content": content}

            #     messages_to_send.append(msg_to_send)

            # print(f"\n\nlen messages_to_send: {len(messages_to_send)}")
            # # print(f"\n\ messages_to_send: {messages_to_send}")

            # for mess in messages_to_send:
            #     print(f"\n\nrole: {mess['role']}")
            #     print(f"type: {mess.get('type')}")
            #     if mess['role'] == "user":
            #         for c in mess['content']:
            #             print("Inside content")
            #             print(f"type: {c['type']}")
            #             if c['type'] == "text":
            #                 print(f"text: {c['text']}")
            #             elif c['type'] == "image":
            #                 print(f"source-type: {c['source']['type']}")
            #                 print(f"source-media_type: {c['source']['media_type']}")
            #                 print(f"source-data: {c['source']['data'][0:200]}")
            #             elif c['type'] == "tool_result":
            #                 print(f"type: {c['type']}")
            #                 print(f"tool_use_id: {c['tool_use_id']}")
            #                 for ci in c['content']:
            #                     print(f"type: {c['type']}")
            #                     print(f"source-type: {ci['source']['type']}")
            #                     print(f"source-media_type: {ci['source']['media_type']}")
            #                     print(f"source-data: {ci['source']['data'][0:200]}")

            try:
                raw_response = client.beta.messages.with_raw_response.create(
                    max_tokens=4096,
                    messages=messages,
                    model="claude-3-5-sonnet-20241022",
                    system=[self.system],
                    tools=tool_collection.to_params(),
                    betas=["computer-use-2024-10-22"],
                )
            except (APIStatusError, APIResponseValidationError, APIError) as e:
                console.print("Error taking action: ", e)
                traceback.print_exc()
                task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
                raise e
                # return _thread, messages, True
            except Exception as e:
                console.print("Exception taking action: ", e)
                traceback.print_exc()
                task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
                raise e
                # return _thread, messages, True

            response = json.loads(raw_response.content.decode('utf-8'))
            print(f"\n\n\nResponse type:\n{type(response)}")
            print(f"Response:\n{response}")

            f_response = raw_response.parse()
            print(f"\n\n\nFreshly parsed Response type:\n{type(f_response)}")
            print(f"Response:\n{f_response}")
            response_params = self._response_to_params(f_response)
            print(f"\n\n\nresponse_params type:\n{type(response_params)}")
            print(f"response_params:\n{response_params}")

            messages.append(
                {
                    "role": "assistant",
                    "content": response_params,
                }
            )

            # Craft the message asking the MLLM for an action
            for resp_content in response["content"]:
                if resp_content["type"] == "text":
                    msg = RoleMessage(
                        role=response["role"],
                        text=resp_content["text"],
                        metadata={
                            'type': resp_content["type"]
                        })
                elif resp_content["type"] == "tool_use":
                    msg = RoleMessage(
                        role=response["role"],
                        text="",
                        metadata=resp_content
                    )

                _thread.add_msg(msg)

            # prompt = Prompt(
            #     thread=thread,
            #     response=resp_msg,
            #     response_schema=expect,  # type: ignore
            #     namespace=namespace,
            #     agent_id=agent_id,
            #     owner_id=owner_id,
            #     model=response.model or model,
            #     logits=logits,
            #     logit_metrics=metrics,
            #     temperature=temperature,
            # )

            # task.add_prompt(response.prompt)

            if response["stop_reason"] == "end_turn":
                console.print("final result: ", style="green")
                console.print(JSON.from_data(response["content"][0]))
                task.post_message(
                    "assistant",
                    f"✅ I think the task is done, please review the result: {response['content'][0]['text']}",
                )
                task.status = TaskStatus.FINISHED
                task.save()
                return _thread, messages, True

            tool_result_content: list[BetaToolResultBlockParam] = []
            for content_block in response_params:
                if content_block["type"] == "tool_use":
                    task.post_message("assistant", f"👁️ {content_block.get('text')}")
                    # task.post_message("assistant", f"💡 {selection.reason}")

                    input_args = cast(dict[str, Any], content_block["input"])
                    action_name = tools_mapping[input_args["action"]]
                    
                    task.post_message(
                        "assistant",
                        f"▶️ Taking action '{action_name}' with parameters: {input_args}",
                    )

                    action_params = input_args.copy()
                    del action_params["action"]

                    try:
                        result = self._execute_action(semdesk, task, action_name, action_params)

                        task.post_message(
                            "assistant",
                            "current image",
                            # images=result.base64_image,
                            thread="debug",
                        )

                        tool_result_content.append(
                            self._make_api_tool_result(result, content_block["id"])
                        )

                        msg = RoleMessage(
                            role="user",
                            text="",
                            # images=result.base64_image,#content[0]["source"]["data"],#[result["screenshot"]],
                            metadata={
                                "type": "tool_result",
                                "tool_use_id": content_block["id"],
                            }
                        )
                        print(f"\nCreated msg: {msg}")
                        _thread.add_msg(msg)

                    except Exception as e:
                        print(f"Error occurred: {str(e)}")
                        print(f"Error type: {type(e)}")
                        raise
                    print("Finished tool_use")
            
            if not tool_result_content:
                return _thread, messages, True

            messages.append({"content": tool_result_content, "role": "user"})
            
            # for content_block in response["content"]:
            #     try:
            #         # if content_block["type"] == "text":
            #         #     _thread.add_msg(RoleMessage(role="user", text=content_block["text"]))
            #         # Post to the user letting them know what the modle selected
            #         if content_block["type"] == "tool_use":
            #             print("Entering tool_use")
            #             tool_use_id = content_block["id"]
            #             input_args = cast(dict[str, Any], content_block["input"])
            #             # print("Arg: ", cast(dict[str, Any], content_block["input"]))
            #             print("Arg: ", input_args)
            #             action_name = tools_mapping[input_args["action"]]

            #             # if not selection:
            #             #     raise ValueError("No action selection parsed")
                        
            #             console.print("action selection: ", style="white")
            #             console.print(JSON.from_data(content_block["input"]))

            #             task.post_message("assistant", f"👁️ {response['content'][0].get('text')}")
            #             # task.post_message("assistant", f"💡 {selection.reason}")

            #             task.post_message(
            #                 "assistant",
            #                 f"▶️ Taking action '{action_name}' with parameters: {input_args}",
            #             )

            #             try:
            #                 # action_response = computer(**input_args)
            #                 # _execute_action()
            #                 # _thread.remove_images()
            #                 pass
            #                 # Craft the message asking the MLLM for an action

                            
            #                 # result = computer(**cast(dict[str, Any], content_block["input"]))
            #             except Exception as e:
            #                 print(f"Error occurred: {str(e)}")
            #                 print(f"Error type: {type(e)}")
            #                 raise
            #             print("Finished tool_use")

            #     except Exception as e:
            #         console.print(f"Response failed to parse: {e}", style="red")
            #         raise

            #     # # The agent will return 'result' if it believes it's finished
            #     # if selection.action.name == "result":
            #     #     console.print("final result: ", style="green")
            #     #     console.print(JSON.from_data(selection.action.parameters))
            #     #     task.post_message(
            #     #         "assistant",
            #     #         f"✅ I think the task is done, please review the result: {selection.action.parameters['value']}",
            #     #     )
            #     #     task.status = TaskStatus.FINISHED
            #     #     task.save()
            #     #     return _thread, True

            #     # # Find the selected action in the tool
            #     # action = semdesk.find_action(selection.action.name)
            #     # console.print(f"found action: {action}", style="blue")
            #     # if not action:
            #     #     console.print(f"action returned not found: {selection.action.name}")
            #     #     raise SystemError("action not found")

            #     # # Take the selected action
            #     # try:
            #     #     action_response = semdesk.use(action, **selection.action.parameters)
            #     # except Exception as e:
            #     #     raise ValueError(f"Trouble using action: {e}")

            #     # console.print(f"action output: {action_response}", style="blue")
            #     # if action_response:
            #     #     task.post_message(
            #     #         "assistant", f"👁️ Result from taking action: {action_response}"
            #     #     )

            #     # # Record the action for feedback and tuning
            #     # task.record_action(
            #     #     state=EnvState(images=[screenshot_img]),
            #     #     prompt=response.prompt,
            #     #     action=selection.action,
            #     #     tool=semdesk.ref(),
            #     #     result=action_response,
            #     #     agent_id=self.name(),
            #     #     model=response.model,
            #     # )

            # # _thread.add_msg(response.msg)
            return _thread, messages, False

        except Exception as e:
            console.print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
            raise e

    def _response_to_params(self, 
        response: BetaMessage,
    ) -> list[BetaTextBlockParam | BetaToolUseBlockParam]:
        res: list[BetaTextBlockParam | BetaToolUseBlockParam] = []
        for block in response.content:
            if isinstance(block, BetaTextBlock):
                res.append({"type": "text", "text": block.text})
            else:
                res.append(cast(BetaToolUseBlockParam, block.model_dump()))
        return res

    def _execute_action(self, semdesk, task, action_name, action_params) -> ToolResult:
        if action_name != "screenshot":
            action = semdesk.find_action(action_name)
            console.print(f"\n\naction parameters: {action_params}")
            if action_name in ["move_mouse", "drag_mouse"] and "coordinate" in action_params:
                action_params["x"] = action_params["coordinate"][0]
                action_params["y"] = action_params["coordinate"][1]
                del action_params["coordinate"]
            if action_name == "hot_key" and "text" in action_params:
                action_params["keys"] = [action_params["text"]]
                del action_params["text"]
            print("just before semdesk call")
            action_response = semdesk.use(action, **action_params)
            print(f"\n\n\n\njust after semdesk call. Rseponse type = {type(action_response)}")

            console.print(f"action output: {action_response}", style="blue")
            if action_response:
                task.post_message(
                    "assistant", f"👁️ Result from taking action: {action_response}"
                )

        screenshot_img = semdesk.desktop.take_screenshots()[-1]
        console.print(f"screenshot img type: {type(screenshot_img)}")

        # # Screenshot by taken semdesk is in the format "data:image/png;base64,iVBORw0KGgoAAAA..."
        # # We should remove the "data:image/png;base64," part from the beginning and only keep the actual image part
        # comma_index = screenshot_img.index(',')
        # base64_image = screenshot_img[comma_index+1:]



        buffer = BytesIO()
        screenshot_img.save(buffer, format='PNG')
        # img_bytes = screenshot_img.tobytes()
        # base64_image = base64.b64encode(img_bytes).decode('utf-8')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        print(f"\n\n\n\n\nbase64_image: type = {type(base64_image)}\n len = {len(base64_image)}\n content = {base64_image[0:200]}")

        # image.save(buffered, format="JPEG")
        # img_str = base64.b64encode(buffered.getvalue())

        return ToolResult(output=None, error=None, base64_image=base64_image)

    def _maybe_prepend_system_tool_result(self, result: ToolResult, result_text: str):
        if result.system:
            result_text = f"<system>{result.system}</system>\n{result_text}"
        return result_text

    def _make_api_tool_result(self, 
        result: ToolResult, tool_use_id: str
    ) -> BetaToolResultBlockParam:
        """Convert an agent ToolResult to an API ToolResultBlockParam."""
        tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
        is_error = False
        if result.error:
            is_error = True
            tool_result_content = self._maybe_prepend_system_tool_result(result, result.error)
        else:
            if result.output:
                tool_result_content.append(
                    {
                        "type": "text",
                        "text": self._maybe_prepend_system_tool_result(result, result.output),
                    }
                )
            if result.base64_image:
                tool_result_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": result.base64_image,
                        },
                    }
                )
        return {
            "type": "tool_result",
            "content": tool_result_content,
            "tool_use_id": tool_use_id,
            "is_error": is_error,
        }

    @classmethod
    def supported_devices(cls) -> List[Type[Device]]:
        """Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        """
        return [Desktop]

    @classmethod
    def config_type(cls) -> Type[SurfPizzaConfig]:
        """Type of config

        Returns:
            Type[SurfPizzaConfig]: Config type
        """
        return SurfPizzaConfig

    @classmethod
    def from_config(cls, config: SurfPizzaConfig) -> "SurfPizza":
        """Create an agent from a config

        Args:
            config (SurfPizzaConfig): Agent config

        Returns:
            SurfPizza: The agent
        """
        return SurfPizza()

    @classmethod
    def default(cls) -> "SurfPizza":
        """Create a default agent

        Returns:
            SurfPizza: The agent
        """
        return SurfPizza()

    @classmethod
    def init(cls) -> None:
        """Initialize the agent class"""
        return


Agent = SurfPizza
