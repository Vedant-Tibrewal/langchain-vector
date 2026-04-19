import inspect
import re

from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable
from time import sleep

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---


@traceable(
    run_type="tool"
)  # run_type is for proper logging on langsmith, no functionality difference
def get_product_price(product: str) -> float:
    """Get the price of a product in the catalog"""

    print(f" >> Executing get_product_price for product='{product}'")
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99,
        "keyboard": 89.99,
    }

    return prices.get(product.lower(), 0.0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Discount Tier: bronze, silver, gold.
    """
    print(
        f" >> Executing apply_discount for price='{price}', discount_tier='{discount_tier}'"
    )

    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}

    discount = discount_percentages.get(discount_tier.lower(), 0)

    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# Change 3: Delete the JSON Schemas. Tools now live inside the the prompt as plain text.
# We derive description from the functions themselves using inspect.


def get_tool_description(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrapper (e.g. @traceable) adds *, config=None
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)


tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# CHANGE 4: Drop tools = from ollama.chat(). the LLM has no idea its an agent -
# all agency comes from the prompt above and our regex parsing below


#  -- AGENT LOOP --


@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # Configuration and setup
    print(f" >> Starting ReAct agent loop")

    print(f"Question: {question}")
    print("=" * 50)

    # CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    # Agent Loop (Thought, Action, Observation, Tool calls, repeat)
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f" --- Iteration {iteration} ---")

        full_prompt = prompt + scratchpad

        # Stop token prevents the LLM from generating its own observation
        # we inject the real tool result instead
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(" >> Parsing LLM output for tool calls...")
        final_anwer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_anwer_match:
            final_answer = final_anwer_match.group(1).strip()
            print(f" [Parsed] Final Answer")
            print("\n" + "=" * 50)
            print(f" [Parsed] Final Answer: {final_answer}")
            return final_answer


        # CHANGE 6: Parse tool calls from raw text with regex - fragile if LLM doesn't follow format.
        print(f" [Parsing] Looking for Action and Action Input in LLM output...")
        
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(" [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]
 

        print(f"  [Tool executing] {tool_name} ({args})...")
        if tool_name not in tools:
            observation = f"Error: tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))      

        print(f"  [Tool result] - {observation}")  

        scratchpad += f"{output}\nObservation: {observation}\nThought:"
        
    print("Error: Max Iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain agent (.bind_tools)!")
    print()
    run_agent("What is the price of a laptop with a silver discount?")
