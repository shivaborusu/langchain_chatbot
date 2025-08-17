# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["/Users/shivaborusu/Development/Repos/langchain_chatbot/src/math_server.py"],
)

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3, max_tokens=500)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # print("TOOLS ", tools)

            # Create and run the agent
            agent = create_react_agent(llm, tools)
            # Format messages as a list of dicts
            agent_response = await agent.ainvoke({"messages": "what's 3 + 5"})
            print(agent_response)

if __name__ == "__main__":
    asyncio.run(main())