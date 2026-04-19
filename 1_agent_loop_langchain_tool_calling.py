from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from time import sleep

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"


# Tools


@tool
def get_product_price(product: str) -> float:
    """Get the price of a product in the catalog"""

    print(f" >> Executing get_product_price for product='{product}'")
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99,
        "keyboard": 89.99
    }

    return prices.get(product.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Discount Tier: bronze, silver, gold.
    """
    print(f" >> Executing apply_discount for price='{price}', discount_tier='{discount_tier}'")

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }

    discount = discount_percentages.get(discount_tier.lower(), 0)

    return round(price * (1 - discount / 100), 2)


@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    # Configuration and setup
    print(f" >> Starting ReAct agent loop")
    # sleep(2)
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools} # name is same as function name by default
    # print(f" >> Tools available: {list(tools_dict.keys())}")

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # llm = init_chat_model(f"openai:gpt-5", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 50)

    messages = [
        SystemMessage(
            content=(
                    "You are helpful shopping assstant." \
                    "You have access to product catalog tool " \
                    "and a discount calculation tool.\n\n" \
                    "STRICT RULES - you must follow these exactly: \n" \
                    "1. NEVER guess or assume any product price. " \
                    "You MUST call get_product_price first to get the real price.\n" \
                    "2. Only call apply_discount AFTER you have received " \
                    "a price from get_product_price. Pass the exact price " \
                    "returned by get_product_price - do NOT pass a made-up number.\n" \
                    "3. NEVER calculate discounts yourself using math. " \
                    "Always use the apply_discount tool.\n" \
                    "4. If the user does not specify a discount tier, " \
                    "ask them which tier to use - do NOT assume one."
                    )
        ),
        HumanMessage(content=question),
    ]

    # Agent Loop (Thought, Action, Observation, Tool calls, repeat)
    for iteration in range(1, MAX_ITERATIONS+1):
        print(f" --- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("No more tool calls. Agent has completed its reasoning.")
            print(f"Final answer: {ai_message.content}") 
            return ai_message.content

        # forcing it to call one tool at a time for better interpretability of the loop
        tool_call = tool_calls[0]  # Get the first tool call
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"    [Tool selected: {tool_name} with args {tool_args}]")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in available tools.")
        
        observations = tool_to_use.invoke(tool_args)

        print(f"    [Observations from tool '{tool_name}': {observations}]")

        messages.append(ai_message)  # Add the AI message with the tool call
        messages.append(
            ToolMessage(content=str(observations), tool_call_id=tool_call_id) 
        )

    print("ERROR: Max iterations reached without a final answer.")
    return None



if __name__ == "__main__":
    print("Hello LangChain agent (.bind_tools)!")
    print()
    run_agent("What is the price of a laptop with a silver discount?")
 