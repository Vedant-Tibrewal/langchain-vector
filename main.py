from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {
        "messages": [HumanMessage(content=res.content)]
    }  # this is a hack to get the reflection to be a human message so that it can be used in the next generation step


def should_continue(state: MessageGraph):
    # if state["messages"][-1].content != "end":
    #     return REFLECT
    # return END
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

builder.add_edge(REFLECT, GENERATE)
builder.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflection_agent_graph.png")


if __name__ == "__main__":
    print("Initializing Reflection agent...")
    inputs = HumanMessage(content="""Make this tweet better: "
                          @LangChainAI
                          - Newly Tool Calling feature is seriously underrated.
                          
                          After a long wait, it's here- making the implementation of agents across different models with function calling
                          
                          Made a video covering their newest blog post
                          """)

    response = graph.invoke({"messages": [inputs]})

    print(response)
