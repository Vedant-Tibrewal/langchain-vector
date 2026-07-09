from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from chains import first_responder, revisor
from tools_executor import execute_tools

MAX_ITERATION = 2


def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}
    # MessagesState has inbuilt add_messages reducer, which is used to append new messages to the state.


def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_calls = sum(1 for message in state["messages"] if isinstance(message, ToolMessage))
    if count_tool_calls > MAX_ITERATION:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, {"execute_tools": "execute_tools", END: END})
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="reflexion_agent_graph.png")

res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)

last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])

print(res)

# def main():
#     print("Hello from langchain-graph-aie!")


# if __name__ == "__main__":
#     main()
