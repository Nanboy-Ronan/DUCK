import json
import operator
import ast
import re
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from pydantic import ValidationError

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        max_messages: Optional[int] = 40,
        parallel_tool_calls: Optional[bool] = None,
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        self.max_messages = max_messages

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        if parallel_tool_calls is None:
            self.model = model.bind_tools(tools)
        else:
            self.model = model.bind_tools(tools, parallel_tool_calls=parallel_tool_calls)

    def _trim_messages(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """Trim message history to avoid runaway context growth."""
        if not self.max_messages or len(messages) <= self.max_messages:
            return messages

        system_msg = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
        start_idx = 1 if system_msg else 0

        head = messages[start_idx : start_idx + 1]
        tail_budget = self.max_messages - (1 if system_msg else 0) - len(head)
        tail = messages[-tail_budget:] if tail_budget > 0 else []

        trimmed = []
        if system_msg:
            trimmed.append(system_msg)
        trimmed.extend(head)
        trimmed.extend(tail)
        return trimmed

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        messages = self._trim_messages(messages)
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            tool_name = call["name"]
            if "<|channel|>" in tool_name:
                tool_name = tool_name.split("<|channel|>", 1)[0].strip()
            tool_args = call.get("args")
            if isinstance(tool_args, dict):
                tool_args = self._normalize_tool_args(tool_args, tool_name=tool_name)
            if tool_name not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                try:
                    result = self.tools[tool_name].invoke(tool_args)
                except ValidationError as exc:
                    result = (
                        f"tool_input_error: {exc}. "
                        "Please retry with all required fields and proper types."
                    )
                except Exception as exc:
                    result = f"tool_execution_error: {exc}"

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=tool_name,
                    args=tool_args,
                    content=str(result),
                )
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    @staticmethod
    def _normalize_tool_args(
        tool_args: Dict[str, Any],
        tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize tool args to fix common serialization issues."""
        def _maybe_parse_list(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            raw = value.strip()
            if not (raw.startswith("[") and raw.endswith("]")):
                return value
            try:
                return json.loads(raw)
            except Exception:
                pass
            try:
                return ast.literal_eval(raw)
            except Exception:
                return value

        def _split_inline_phrase(value: str) -> Tuple[Optional[str], Optional[str]]:
            if not isinstance(value, str):
                return None, None
            raw = value.strip()
            if "phrase" not in raw:
                return None, None
            match = re.search(r"(?:,|;|\\s)phrase\\s*[:=]\\s*", raw, flags=re.IGNORECASE)
            if not match:
                return None, None
            img = raw[: match.start()].strip().rstrip(",;")
            phrase = raw[match.end() :].strip()
            if not img or not phrase:
                return None, None
            return img, phrase

        normalized = dict(tool_args)
        for key in ("image_paths", "organs"):
            if key in normalized:
                normalized[key] = _maybe_parse_list(normalized[key])
        if tool_name == "xray_phrase_grounding":
            phrase = normalized.get("phrase")
            image_path = normalized.get("image_path")
            if (not phrase or not isinstance(phrase, str)) and isinstance(image_path, str):
                img, parsed_phrase = _split_inline_phrase(image_path)
                if img and parsed_phrase:
                    normalized["image_path"] = img
                    normalized["phrase"] = parsed_phrase
        return normalized

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)
