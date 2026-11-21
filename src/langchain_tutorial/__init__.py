from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
tavily_client = TavilyClient()
    
@tool
def search_demo(query: str) -> str:
    """
    Tool that searches over the internet for a given query and returns the results.
    
    Args:
        query (str): The search query.
    
    Returns:
        str: The search results.
    """
    
    print(f"Searching for: {query} ...")
    return tavily_client.search(query=query)

llm = ChatOllama(model="gemma4:12b")
tools = [TavilySearch(), search_demo]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-tutorial!")
    response = agent.invoke(
        # {"messages": [HumanMessage(content="What is the weather like today in Lagos?")]}
        {"messages": [HumanMessage(content="Search for three job in AI engineer role using LangChain that accepts engineers living in Lagos, Nigeria, that's still accepting applications, and provide the links to the job postings.")]}
    )
    print(f"Agent response: {response}")
    # print(f"Agent response: {response['messages'][-1].content}")


if __name__ == "__main__":
    main()
