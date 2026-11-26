import logging

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gemma-assistant"





# --- Tools (LangChain @tool decorator) ---
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    prices = {
        "laptop": 1299.99,
        "headphone": 149.95,
        "keyboard": 89.50,
    }
    
    return prices.get(product.lower(), 0.0)


@tool
def get_discounted_price(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available discount tiers:
    bronze, silver, gold
    """
    logging.info(f"Calculating discounted price for {price} with discount tier {discount_tier}")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }
    
    discount = discount_percentages.get(discount_tier.lower(), 0)
    
    return round(price * (1 - discount / 100), 2)


#  --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, get_discounted_price]
    
    print(f"$$$$$$$$$$$$$get_product_price: {get_product_price}\n\n")
    
    logging.info(f"Available tools: {[tool.name for tool in tools]}")
    print(f"Available tools: {[tool.name for tool in tools]}")

    tools_dict = {tool.name: tool for tool in tools}
    print(f"$$$$$$$$$$$$$tools_dict: {tools_dict}\n\n")
    
    llm = init_chat_model(f"ollama: {MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    
    logging.info(f"~~~~~~Question~~~~~~: {question}")
    
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that can answer questions about product prices "
                "and apply discount tiers to calculate the final price. "
                "You have access to the following tools: "
                "1. get_product_price(product: str) -> float: Look up the price of a product in the catalog. "
                "2. get_discounted_price(price: float, discount_tier: str) -> float: Apply a discount tier to a price and return the final price. "
                "Available discount tiers: bronze, silver, gold.\n\n"
                "STRICT RULES: you must follow these exactly:\n"
                "1. Never guess or assume any product price."
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount after you have received a price from get product price."
                "Pass the price returned by get_product price - do NOT Pass made-up number."
                "3. Never calculate discounts yourself using math"
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier,"
                "ask them which tier to use. Do not assume one"
            )
        ),
        HumanMessage(content=question),
    ]
    
    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        ai_message = llm_with_tools.invoke(messages)
        
        # Add the assistant's response to the message history
        messages.append(ai_message)
        print(f"~~~~~~~Message: {messages}")
        print('#######################################################################################\n\n')
        
        tool_calls = ai_message.tool_calls
        print(f"~~~~~~~tool_calls: {tool_calls}\n\n")
        
        # If the AI has no tool calls, it means it has generated a final answer
        if not tool_calls:
            print("No tool call detected. Final answer:")
            print('~~~~~~~ai_message.content', ai_message.content)
            return ai_message.content
        
        #  process only the first tool call - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        
        logging.info(f"Tool selected: {tool_name} with args: {tool_args}")
        print(f"Tool selected: {tool_name} with args: {tool_args}")
        
        tool_to_call = tools_dict.get(tool_name)
        if tool_to_call is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        
        observation = tool_to_call.invoke(tool_args)
         
        logging.info(f"Observation from tool '{tool_name}': {observation}")
        print(f"Observation from tool '{tool_name}': {observation}")
         
         # Add the tool output to the message history
        
        messages.append(ai_message)
        messages.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id
        ))
        
        # Log error if the maximum number of iterations is reached and the agent has not produced a final answer
        if iteration == MAX_ITERATIONS - 1:
            logging.error("Maximum iterations reached without producing a final answer.")
            print("Maximum iterations reached without producing a final answer.")
            return "Error: Maximum iterations reached without producing a final answer."


if __name__ == "__main__":
    logging.info("Welcome to the LangChain Agent Loop with Tool Calling!")
    print()
    result = run_agent("What is the discounted price of a laptop with a gold discount?")
    
    print()
    logging.info(f"Final Answer: {result}")
    print(f"Final Answer: {result}")