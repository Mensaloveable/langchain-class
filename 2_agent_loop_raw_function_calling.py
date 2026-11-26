import logging

import ollama
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gemma-assistant"


# Make LangSmith traceable decorator available for this tool
@traceable(name="Get Product Price Tool", run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    prices = {
        "laptop": 1299.99,
        "headphone": 149.95,
        "keyboard": 89.50,
    }

    return prices.get(product.lower(), 0.0)


@traceable(name="Get Discounted Price Tool", run_type="tool")
def get_discounted_price(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available discount tiers:
    bronze, silver, gold
    """
    logging.info(
        f"Calculating discounted price for {price} with discount tier {discount_tier}"
    )
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    discount = discount_percentages.get(discount_tier.lower(), 0)

    return round(price * (1 - discount / 100), 2)


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
                        "description": "The name of the product to look up.",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_discounted_price",
            "description": (
                "Apply a discount tier to a price and return the final price. "
                "Available discount tiers: bronze, silver, gold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price of the product.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": (
                            "The discount tier to apply. "
                            "Available options: bronze, silver, gold."
                        ),
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat(messages):
    """
    A wrapper function for the Ollama chat model.
    This function is traceable and can be used to log interactions with the model.
    """
    return ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools_for_llm,
    )


#  --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    tools_dict = {
        "get_product_price": get_product_price,
        "get_discounted_price": get_discounted_price,
    }

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can answer questions about product prices "
            "and apply discount tiers to calculate the final price. "
            "You have access to the following tools: "
            "1. get_product_price(product: str) -> float: Look up the price of a product in the catalog. "
            "2. get_discounted_price(price: float, discount_tier: str) -> float: Apply a discount tier to a price and return the final price. "
            "Available discount tiers: bronze, silver, gold.\n\n"
            "STRICT RULES: you must follow these exactly:\n"
            "1. Never guess or assume any product price."
            "You must call get_product_price first to get the real price.\n"
            "2. Only call get_discounted_price after you have received a price from get_product_price. "
            "Pass the price returned by get_product_price - do NOT pass a made-up number.\n"
            "3. Never calculate discounts yourself using math. "
            "Always use the get_discounted_price tool.\n"
            "4. If the user does not specify a discount tier,"
            "ask them which tier to use. Do not assume one",
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")

        response = ollama_chat(messages=messages)
        ai_message = response.get("message", {})

        # Add the assistant's response to the message history
        messages.append(ai_message)

        tool_calls = ai_message.tool_calls
        print(f"~~~~~~~tool_calls: {tool_calls}\n\n")

        # If the AI has no tool calls, it means it has generated a final answer
        if not tool_calls:
            print("No tool call detected. Final answer:")
            print("~~~~~~~ai_message.content", ai_message.content)
            return ai_message.content

        #  process only the first tool call - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("function").get("name")
        tool_args = tool_call.get("function").get("arguments", {})

        logging.info(f"Tool selected: {tool_name} with args: {tool_args}")
        print(f"Tool selected: {tool_name} with args: {tool_args}")

        tool_to_call = tools_dict.get(tool_name)
        if tool_to_call is None:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_call(**tool_args)

        logging.info(f"Observation from tool '{tool_name}': {observation}")
        print(f"Observation from tool '{tool_name}': {observation}")

        # Add the tool output to the message history
        # (ai_message was already appended above, so don't append it again)
        messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": str(observation),
            }
        )

        # Log error if the maximum number of iterations is reached and the agent has not produced a final answer
        if iteration == MAX_ITERATIONS - 1:
            logging.error(
                "Maximum iterations reached without producing a final answer."
            )
            print("Maximum iterations reached without producing a final answer.")
            return "Error: Maximum iterations reached without producing a final answer."


if __name__ == "__main__":
    logging.info("Welcome to the LangChain Agent Loop with Tool Calling!")
    print()
    result = run_agent("What is the discounted price of a laptop with a gold discount?")

    print()
    logging.info(f"Final Answer: {result}")
    print(f"Final Answer: {result}")
