from temporalio import activity
from google import genai
from google.genai import types
from dataclasses import dataclass
from typing import Any

# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class GeminiResponsesRequest:
    model: str
    instructions: str
    history: list[dict[str, Any]]
    prompt: str
    tools: list[Any]

def serialize_response(response: Any) -> dict[str, Any]:
    """
    Convert Gemini API response to serializable format.
    Extracts function calls and text from response parts.
    """
    serialized_parts = []
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            part_dict = {}
            if hasattr(part, 'function_call') and part.function_call:
                part_dict["function_call"] = {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                }
            elif hasattr(part, 'text') and part.text:
                part_dict["text"] = part.text
            serialized_parts.append(part_dict)

    return {"parts": serialized_parts}

@activity.defn
async def create(request: GeminiResponsesRequest) -> dict[str, Any]:
    """
    Invoke Gemini API with pre-built conversation history and tools.
    Returns the raw response from generate_content() in serializable format.
    """
    # Create Gemini client (automatically picks up GOOGLE_API_KEY from environment)
    client = genai.Client()

    print(f"Create: request is {request}")

    # Create config with system instructions and tools
    config = types.GenerateContentConfig(
        system_instruction=request.instructions,
        tools=request.tools
    )

    # Build contents list from history + current prompt
    contents = []

    # Add all history items
    for history_item in request.history:
        contents.append(history_item)

    # Add the current prompt as a user message
    contents.append({
        "role": "user",
        "parts": [{"text": request.prompt}]
    })

    # Generate content with full conversation history
    response = await client.aio.models.generate_content(
        model=request.model,
        contents=contents,
        config=config
    )

    # Serialize and return response
    return serialize_response(response)
