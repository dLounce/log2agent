"""Standalone discovery test — run from terminal, not Jupyter."""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["mockbank_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            print(f"MockBank exposes {len(result.tools)} tools:")
            for t in result.tools:
                print(f"  {t.name}: {t.description[:60]}")

if __name__ == "__main__":
    asyncio.run(main())
