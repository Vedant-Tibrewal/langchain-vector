import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    "Get weather for location."
    return f"Weather for {location} is sunny."


if __name__ == "__main__":
    transport = "streamable_http"
    # transport = "sse"
    if transport == "streamable_http":
        # localhost:8000/mcp is the endpoint
        mcp.run(transport="streamable-http")
    else:
        # localhost:8000/sse is the endpoint
        mcp.run(transport="sse")
