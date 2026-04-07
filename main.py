from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search_web(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query (str): the search query
    Returns:
        the search results
    """
    print(f"Searching for: {query}")

    return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-5")
tools = [search_web]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        {"messages": HumanMessage(content="Search for 3 job postings for AI engineer role in Bay area")}
    )
    print(result)


if __name__ == "__main__":
    main()
