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
    tools_dict = {tool.name: tool for tool in tools}
    
    llm = init_chat_model(f"ollama: {MODEL}", temperature=0.0)
    llm_with_tools = llm.with_tools(tools)
    
    logging.info(f"Question: {question}")


if __name__ == "__main__":
    logging.info("Welcome to the LangChain Agent Loop with Tool Calling!")
    print()
    result = run_agent("What is the discounted price of a laptop with a gold discount?")
    
    print()
    logging.info(f"Final Answer: {result}")