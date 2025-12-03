from temporalio import activity
from typing import Sequence, Any
from temporalio.common import RawValue
from temporalio.exceptions import ApplicationError
from dataclasses import dataclass
import inspect
from pydantic import BaseModel

@dataclass
class ToolArguments:
    tool_name: str
    args: dict

@activity.defn
async def invoke_tool(tool_args: ToolArguments) -> Any:
    from gemini_agent.tools import get_handler

    activity.logger.info(f"Running dynamic tool '{tool_args.tool_name}' with args: {tool_args.args}")

    handler = get_handler(tool_args.tool_name)
    if handler is None:
        activity.logger.info(f"Tool '{tool_args.tool_name}' was not found")
        raise ApplicationError(
            type="ToolNotFoundError",
            message=f"Tool '{tool_args.tool_name}' was not found",
            non_retryable=True
        )

    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) == 0:
        call_args = []
    else:
        ann = params[0].annotation
        if isinstance(tool_args.args, dict) and isinstance(ann, type) and issubclass(ann, BaseModel):
            call_args = [ann(**tool_args.args)]  # or ann.model_validate(tool_args.args) on Pydantic v2
        else:
            call_args = [tool_args.args]

    result = await handler(*call_args) if inspect.iscoroutinefunction(handler) else handler(*call_args)

    # Optionally log or augment the result
    activity.logger.info(f"Tool '{tool_args.tool_name}' result: {result}")
    return result

