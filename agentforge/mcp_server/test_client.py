import asyncio
from fastmcp import Client
from agentforge.mcp_server.server import mcp


async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        result = await client.call_tool("kb_search", {"query": "printer not working", "top_k": 2})
        print("\nkb_search result:")
        print(result.data)


if __name__ == "__main__":
    asyncio.run(main())