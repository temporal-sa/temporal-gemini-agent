import asyncio
import sys
import uuid

from temporalio.client import Client

from gemini_agent.workflows.agent import AgentGeminiWorkflow


async def main():
    client = await Client.connect(
        "localhost:7233",
    )

    query = sys.argv[1] if len(sys.argv) > 1 else "Tell me about recursion"

    # Submit the the agent workflow for execution
    result = await client.execute_workflow(
        AgentGeminiWorkflow.run,
        query,
        id=f"agentic-loop-id-{uuid.uuid4()}",
        task_queue="tool-invoking-agent-gemini-task-queue", # Changed task queue name
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
