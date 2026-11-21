from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient()

class Source(BaseModel):
    """
    Schema for a source used by an agent.
    
    Attributes:
        url (str): The URL of the source.
    """
    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """
    Schema for an agent's response with answer and sources.
    
    Attributes:
        content (str): The content of the response.
        sources (List[Source]): A list of sources used in the response.
    """
    answer: str = Field(description="The agent's answer to the query.")
    sources: List[Source] = Field(default_factory=list, description="A list of sources used in the response.")
    
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

llm = ChatOllama(model="devstral-small-2")
tools = [TavilySearch(), search_demo]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

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
