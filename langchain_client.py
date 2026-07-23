import asyncio
import os

from dotenv import load_dotenv

# from langgraph.prebuilt import create_react_agent # deprecated
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-5.5")


def build_mcp_connections() -> dict:
    # Set WEATHER_TRANSPORT to "streamable_http" if weather server runs with transport="streamable-http".
    # weather_transport = os.getenv("WEATHER_TRANSPORT", "sse")
    weather_transport = "streamable_http"
    # weather_transport = "sse"

    connections = {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["servers/math_server.py"],
        }
    }

    if weather_transport == "streamable_http":
        connections["weather"] = {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    else:
        connections["weather"] = {
            "transport": "sse",
            "url": "http://127.0.0.1:8000/sse",
        }

    return connections


async def main():
    client = MultiServerMCPClient(build_mcp_connections(), tool_name_prefix=True)

    tools = await client.get_tools()
    print(f"Loaded tool count: {len(tools)}")
    print("Tool names:", [tool.name for tool in tools])

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SystemMessage(
            content=(
                "You are a math and weather assistant. "
                "Use MCP tools for arithmetic and weather questions. "
                "If the question is unrelated, say you can only help with math or weather."
            )
        ),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="What is 94 + 3 * 2 and what is weather in Bengaluru?")]}
    )
    print(f"Result: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
