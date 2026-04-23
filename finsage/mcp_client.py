import asyncio
import sys
import json
import os
import argparse
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# We use standard config logic from FinSage
from config.models import GROQ_FAST

class FinSageMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize Groq Client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    async def connect_to_server(self, server_url: str):
        """Connect to the MCP server over SSE"""
        print(f"Connecting to SSE server at {server_url}...")
        
        sse_transport = await self.exit_stack.enter_async_context(sse_client(url=server_url))
        self.sse, self.write = sse_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.sse, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Groq and available MCP tools"""
        messages = [
            {"role": "system", "content": "You are a helpful financial assistant with access to various financial tools. When asked a question, use your tools to fetch the necessary data before answering."},
            {"role": "user", "content": query}
        ]

        response = await self.session.list_tools()
        
        # Format tools for Groq
        groq_tools = []
        for tool in response.tools:
            groq_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        # Initial Groq API call
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_FAST,
            tools=groq_tools,
            max_tokens=2048
        )

        response_message = chat_completion.choices[0].message
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            # We have tool calls
            print(f"\n[Model chose to use {len(tool_calls)} tool(s)]")
            
            # Add the assistant's message with tool calls to the history
            messages.append(response_message)
            
            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                print(f"[Executing {tool_name} with args {tool_args}]")
                
                # Call MCP tool
                result = await self.session.call_tool(tool_name, tool_args)
                
                # Ensure result is text for appending to messages
                if hasattr(result, 'content') and isinstance(result.content, list):
                    # MCP content is usually a list of text/image blocks
                    tool_result_text = "\n".join([c.text for c in result.content if getattr(c, 'type', '') == 'text'])
                else:
                    tool_result_text = str(result)
                
                # Feed tool result back to Groq
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result_text,
                })
            
            # Get final response from Groq
            final_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=GROQ_FAST,
                max_tokens=2048
            )
            return final_completion.choices[0].message.content
            
        else:
            return response_message.content

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                
                if not query:
                    continue
                    
                response = await self.process_query(query)
                print("\nAssistant:", response)
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    parser = argparse.ArgumentParser(description="FinSage MCP client")
    parser.add_argument(
        "server_url",
        nargs="?",
        default="http://localhost:8001/sse",
        help="MCP server SSE URL (default: http://localhost:8001/sse)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Run a single query and exit (non-interactive mode)",
    )
    args = parser.parse_args()

    server_url = args.server_url
        
    client = FinSageMCPClient()
    try:
        await client.connect_to_server(server_url)
        if args.query and args.query.strip():
            response = await client.process_query(args.query.strip())
            print("\nAssistant:", response)
        else:
            await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
