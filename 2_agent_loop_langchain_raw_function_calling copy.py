from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable
from time import sleep

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")  # run_type is for proper logging on langsmith, no functionality difference
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


@traceable(run_type="tool")
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


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {  # properties = parameters/arguments for the function
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],  # required arguements for the function
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.


# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # Configuration and setup
    print(f" >> Starting ReAct agent loop")
    # sleep(2)
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }
    # print(f" >> Tools available: {list(tools_dict.keys())}")


    print(f"Question: {question}")
    print("=" * 50)

    # Difference 4: Roles in messages
    messages = [
        {'role': 'system',
            'content': (
                    "You are helpful shopping assstant."
                    "You have access to product catalog tool "
                    "and a discount calculation tool.\n\n"
                    "STRICT RULES - you must follow these exactly:\n"
                    "1. NEVER guess or assume any product price. "
                    "You MUST call get_product_price first to get the real price.\n"
                    "2. Only call apply_discount AFTER you have received "
                    "a price from get_product_price. Pass the exact price "
                    "returned by get_product_price - do NOT pass a made-up number.\n"
                    "3. NEVER calculate discounts yourself using math. "
                    "Always use the apply_discount tool.\n"
                    "4. If the user does not specify a discount tier, "
                    "ask them which tier to use - do NOT assume one."
                    )
        },
        {'role': 'user', 'content': question},
    ]

    # Agent Loop (Thought, Action, Observation, Tool calls, repeat)
    for iteration in range(1, MAX_ITERATIONS+1):
        print(f" --- Iteration {iteration} ---")

        # Difference 5: ollama.chat() instead of llm_tools.invoke()
        response = ollama_chat_traced(messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls


        if not tool_calls:
            print("No more tool calls. Agent has completed its reasoning.")
            print(f"Final answer: {ai_message.content}") 
            return ai_message.content

        # forcing it to call one tool at a time for better interpretability of the loop
        tool_call = tool_calls[0]  # Get the first tool call

        # Difference 6: Attribute access(.function.name) vs dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    [Tool selected: {tool_name} with args {tool_args}]")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in available tools.")
        
        # Difference 7: Direct function call with unpacked arguments vs tool.invoke()
        observations = tool_to_use(**tool_args)

        print(f"    [Observations from tool '{tool_name}': {observations}]")

        messages.append(ai_message)  # Add the AI message with the tool call
        messages.append(
            {'role': 'tool', 'content': str(observations)}
        )

    print("ERROR: Max iterations reached without a final answer.")
    return None



if __name__ == "__main__":
    print("Hello LangChain agent (.bind_tools)!")
    print()
    run_agent("What is the price of a laptop with a silver discount?")
 