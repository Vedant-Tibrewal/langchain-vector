import asyncio

from dotenv import load_dotenv

# from langgraph.prebuilt import create_react_agent # deprecated
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

# from langgraph.prebuilt impo
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

llm = ChatOpenAI(model="gpt-5.5")

stdio_server_params = StdioServerParameters(
    command="python",
    args=["servers/math_server.py"],
)


async def main():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("MCP session initialized.")
            # tools = await session.list_tools()
            # agent = create_agent(model=llm, tools=tools)
            # the above code gives error because of session.list_tools()
            # returns a list of tool names, but create_agent expects a list of Tool objects.
            # So we need to load the tools using load_mcp_tools

            tools = await load_mcp_tools(session)
            # print(f"Available tools: {tools}")
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=SystemMessage(
                    content="You are a math expert. Calculate the results only by using the tools provided. "
                    "Do not use any other methods or libraries for calculations. If the question is not related "
                    "to math, respond with 'I can only answer math questions.'"
                ),
            )

            # result = await agent.ainvoke({"messages": [HumanMessage(content="What is 2 + 3?")]})
            # print(f"Result: {result['messages'][-1].content}")

            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 94 + 3* 2?")]})
            print(f"Result: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
