import sys
print(sys.path)
print(f"Name is {__name__}")
print(f"Package is {"none" if __package__ is None else __package__}")
print(f"spec.parent is {__spec__.parent}")

import pytest
# import pytest_asyncio
import uuid
from typing import Any

from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.testing import WorkflowEnvironment

from gemini_agent.workflows.agent import (
    AgentGeminiWorkflow,
)

from  gemini_agent.activities.gemini_responses import (
    GeminiResponsesRequest,
)


@activity.defn(name="create")
async def create_mocked_for_get_ip_address_tool_call_result(request: GeminiResponsesRequest) -> dict[str, Any]:
    return {'parts': [{'text': 'Your IP address\nIs 19.199.198.200.\nI hope this helps you.\n'}]}

@pytest.mark.asyncio
async def test_mocked_tool_call_result():
    task_queue_name = str(uuid.uuid4())

    env = await WorkflowEnvironment.start_time_skipping()

    async with Worker(
        client=env.client,
        task_queue=task_queue_name,
        workflows=[AgentGeminiWorkflow],
        activities=[create_mocked_for_get_ip_address_tool_call_result],
    ):
        # execute workflow as usual
        result = await env.client.execute_workflow(
            AgentGeminiWorkflow.run,
            "My mocked prompt",
            id=str(uuid.uuid4()),
            task_queue=task_queue_name,
        )

        assert result == "Your IP address\nIs 19.199.198.200.\nI hope this helps you.\n"
