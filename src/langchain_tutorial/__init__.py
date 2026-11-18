from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()

def main() -> None:
    print("Hello from langchain-tutorial!")
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather like today in Lagos?")]}
    )
    print(f"Agent response: {response['messages'][-1].content}")
    
@tool
def search_demo(query: str) -> str:
    """
    Tool that searches over the internet for a given query and returns the results.
    
    Args:
        query (str): The search query.
    
    Returns:
        str: The search results.
    """
    
    print(f"Searching for: {query} has a sunny day")
    return f"{query} has a sunny day"

llm = ChatOllama(model="gemma4:12b")
tools = [search_demo]
agent = create_agent(model=llm, tools=tools)


if __name__ == "__main__":
    main()
