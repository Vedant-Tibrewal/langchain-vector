from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# from tavily import TavilyClient
from langchain_tavily import TavilySearch


# tavily = TavilyClient()

 
# @tool
# def search_web(query: str) -> str:
#     """
#     Tool that searches over the internet
#     Args:
#         query (str): the search query
#     Returns:
#         the search results
#     """
#     print(f"Searching for: {query}")

#     return tavily.search(query=query)


class Source(BaseModel):
    """
    Schema for the source used by the agent
    """

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """
    Schema for the agent response with answer and sources
    """

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")



llm = ChatOpenAI(model="gpt-5")
# tools = [search_web]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {"messages": HumanMessage(content="Search for 3 job postings for AI engineer role in Bay area")}
    )
    print(result)


if __name__ == "__main__":
    main()
