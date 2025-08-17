from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools

async def main():
    async with streamablehttp_client(url="https://huggingface.co/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            print("Available Tools, ", tools)


if __name__ == "__main__":
    asyncio.run(main())