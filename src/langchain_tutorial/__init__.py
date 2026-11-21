import logging
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient()

logger = logging.getLogger(__name__)


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
    sources: List[Source] = Field(
        default_factory=list, description="A list of sources used in the response."
    )


@tool
def search_demo(query: str) -> str:
    """
    Tool that searches over the internet for a given query and returns the results.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """

    logger.info("Searching for: %s ...", query)
    return tavily_client.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tools = [TavilySearch(), search_demo]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Hello from langchain-tutorial!")
    response = agent.invoke(
        # {"messages": [HumanMessage(content="What is the weather like today in Lagos?")]}
        {
            "messages": [
                HumanMessage(
                    content="Search for three job in AI engineer role using LangChain that accepts engineers living in Lagos, Nigeria, that's still accepting applications, and provide the links to the job postings."
                )
            ]
        }
    )
    logger.info("Agent response: %s", response)
    # logger.info("Agent response: %s", response["messages"][-1].content)


if __name__ == "__main__":
    main()
