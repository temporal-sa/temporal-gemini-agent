from temporalio import workflow
from datetime import timedelta
from typing import Any, Union
from dataclasses import dataclass

import json

from gemini_agent.activities.tool_invoker import invoke_tool, ToolArguments

with workflow.unsafe.imports_passed_through():
    from gemini_agent.tools import get_tools
    from gemini_agent.helpers import tool_helpers
    from gemini_agent.activities import gemini_responses

@dataclass
class FunctionCallOutput:
    """Represents a function call in the model's response."""
    type: str  # Always "function_call"
    name: str
    call_id: str
    arguments: dict[str, Any]

@dataclass
class MessageOutput:
    """Represents a message in the model's response."""
    type: str  # Always "message"
    content: str

@dataclass
class GeminiResponse:
    """Parsed Gemini response containing output items and text."""
    output: list[Union[FunctionCallOutput, MessageOutput]]
    output_text: str

def build_history_from_input(input_list: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """
    Convert input list to Gemini's expected history format.
    Returns a tuple of (history, prompt) where:
    - history: list of conversation history items in Gemini format
    - prompt: the prompt to send to the model
    """
    history = []

    # Check if the last item is a function_call_output - if so, include it in history
    last_item = input_list[-1]
    is_continuing_after_tool = last_item.get("type") == "function_call_output"

    # If we're continuing after a tool call, include all items in history
    # Otherwise, all but the last item go into history
    items_for_history = input_list if is_continuing_after_tool else input_list[:-1]

    for item in items_for_history:
        if item.get("type") == "message":
            history.append({
                "role": item["role"],
                "parts": [{"text": item["content"]}]
            })
        elif item.get("type") == "function_call":
            # Model's tool call
            history.append({
                "role": "model",
                "parts": [{
                    "function_call": {
                        "name": item["name"],
                        "args": dict(item["arguments"])
                    }
                }]
            })
        elif item.get("type") == "function_call_output":
            # Tool response (sent as user role with function_response part)
            history.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": item["call_id"],
                        "response": {"result": item["output"]}
                    }
                }]
            })

    # Determine the prompt to send
    if is_continuing_after_tool:
        # After a function response, we need to prompt Gemini to continue
        prompt = "Please continue and provide your response based on the tool results."
    else:
        prompt = last_item.get("content", "")

    return history, prompt

def parse_gemini_response(raw_response: dict[str, Any]) -> GeminiResponse:
    """
    Parse the raw Gemini response and convert to output structure.
    Returns a GeminiResponse dataclass with 'output' (list of items) and 'output_text' (str).
    """
    output: list[Union[FunctionCallOutput, MessageOutput]] = []
    output_text = ""

    parts = raw_response.get("parts", [])

    for part in parts:
        if "function_call" in part:
            # Tool call detected
            function_call = part["function_call"]
            output.append(FunctionCallOutput(
                type="function_call",
                name=function_call["name"],
                call_id=function_call["name"],  # Use name as call_id
                arguments=function_call["args"]
            ))
        elif "text" in part:
            # Text response
            output_text += part["text"]

    # If no tool calls, add text as message output
    if not any(isinstance(item, FunctionCallOutput) for item in output):
        output.append(MessageOutput(
            type="message",
            content=output_text
        ))

    return GeminiResponse(output=output, output_text=output_text)

@workflow.defn
class AgentGeminiWorkflow:
    @workflow.run
    async def run(self, input: str) -> str:

        input_list = [{"type": "message", "role": "user", "content": input}]

        # The agentic loop
        while True:

            print(80 * "=")

            # Build history and prompt from input list
            history, prompt = build_history_from_input(input_list)

            # consult the LLM
            raw_response = await workflow.execute_activity(
                gemini_responses.create,
                gemini_responses.GeminiResponsesRequest(
                    model="gemini-2.0-flash-exp",
                    instructions=tool_helpers.HELPFUL_AGENT_SYSTEM_INSTRUCTIONS,
                    history=history,
                    prompt=prompt,
                    tools=get_tools(),
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )

            # print("---- raw response ---- ")
            # print(f"{raw_response}")
            # print("--- end raw response ---")

            # Parse the raw response
            result = parse_gemini_response(raw_response)

            # For this simple example, we only have one item in the output list
            # Either the LLM will have chosen a single function call or it will
            # have chosen to respond with a message.
            item = result.output[0]

            # Now process the LLM output to either call a tool or respond with a message.

            # if the result is a tool call, call the tool
            if isinstance(item, FunctionCallOutput):
                tool_result = await self._handle_function_call(item, result, input_list)

                # add the tool call result to the input list for context
                input_list.append({"type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": tool_result})

            # if the result is not a tool call we will just respond with a message
            else:
                print(f"No tools chosen, responding with a message: {result.output_text}")
                return result.output_text


    async def _handle_function_call(self, item: FunctionCallOutput, result: GeminiResponse, input_list):
        # serialize the LLM output - the decision the LLM made to call a tool
        input_list.append({
            "type": "function_call",
            "name": item.name,
            "call_id": item.call_id,
            "arguments": item.arguments
        })

        # execute activity with the tool name and arguments
        tool_args = ToolArguments(item.name, item.arguments)

        tool_result = await workflow.execute_activity(
            invoke_tool,
            tool_args,
            start_to_close_timeout=timedelta(seconds=30),
            summary=item.name,
        )

        print(f"Made a tool call to {item.name}")

        return tool_result
