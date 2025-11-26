from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
import sys
import os

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import Prompts

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession
from dotenv import load_dotenv
import json
from typing import Any
from openai import AsyncOpenAI

load_dotenv()



class Graph(BaseModel):
    query: str
    ready_prompt: str = None
    agent_response: str = None
    judge_score: int = 0
    fact_check_result: str = None # не реализовано
    fail_cnt: int = 0
    fail_message: str = None
    urls: list[str] = []
    called_tools: list[str] = []



class AgentSystem:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://foundation-models.api.cloud.ru/v1"
        )

        self.my_prompts = Prompts()
        self.prompt_generator = [{"role": "system", "content": self.my_prompts.generate_prompt}]
        self.answer_generator = [{"role": "system", "content": self.my_prompts.agent_prompt}]
        self.judge_generator = [{"role": "system", "content": self.my_prompts.critical_prompt}] 


        self.state_graph = StateGraph(Graph)
        self.state_graph.add_node("prompt_generate", self.prompt_generate_node)
        self.state_graph.add_node("answer_generate", self.answer_generate_node)
        self.state_graph.add_node("judge", self.judge_node)
        self.state_graph.add_node("fail", self.fail_node)

        self.state_graph.add_edge(START, "prompt_generate")
        self.state_graph.add_edge("prompt_generate", "answer_generate")
        self.state_graph.add_edge("answer_generate", "judge")
        self.state_graph.add_conditional_edges(
            "judge",
            self._route_judge
        )
        self.state_graph.add_edge("fail", END)
        self.compiled_graph = self.state_graph.compile()

        # Use the same Python interpreter that runs the server
        import sys
        # Build absolute path to MCP server
        mcp_server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mcp-server",
            "mcp_server.py"
        )
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[mcp_server_path],
            env=os.environ.copy()
        )
        self.openai_tools: list[dict[str, Any]] = []
        self._tools_loaded = False

    async def prompt_generate_node(self, state: Graph) -> dict:
        self.prompt_generator.append({"role": "user", "content": state.query})
        completion = await self.client.chat.completions.create(
            model=os.getenv("LLM_NAME"),
            messages=self.prompt_generator
        )
        return {"ready_prompt": completion.choices[0].message.content}

    async def answer_generate_node(self, state: Graph) -> dict:
        await self._ensure_openai_tools()
        self.answer_generator.append({"role": "user", "content": state.ready_prompt})
        
        # Only pass tools if we have them
        create_kwargs = {
            "model": os.getenv("LLM_NAME"),
            "messages": self.answer_generator,
        }
        if self.openai_tools:
            create_kwargs["tools"] = self.openai_tools
            # Force model to use search_qdrant tool for knowledge base lookup
            create_kwargs["tool_choice"] = {"type": "function", "function": {"name": "search_qdrant"}}
        
        completion = await self.client.chat.completions.create(**create_kwargs)
        message = completion.choices[0].message
        tool_calls = self._extract_tool_calls(message)
        called_tools: list[str] = []

        collected_urls: list[str] = []
        
        if tool_calls:
            # Add assistant message with tool call to history
            self.answer_generator.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": self._get_tool_call_id(tc),
                        "type": "function",
                        "function": {
                            "name": self._get_tool_call_function_name(tc),
                            "arguments": self._get_tool_call_function_arguments(tc)
                        }
                    }
                    for tc in tool_calls
                ]
            })
            
            tool_results = []
            for tool_call in tool_calls:
                tool_name = self._get_tool_call_function_name(tool_call)
                arguments_raw = self._get_tool_call_function_arguments(tool_call)
                try:
                    arguments = json.loads(arguments_raw) if arguments_raw else {}
                except json.JSONDecodeError:
                    arguments = {}
                if tool_name:
                    called_tools.append(tool_name)
                tool_output = await self._call_mcp_tool(tool_name, arguments)
                
                # Extract URLs from tool output
                urls_from_tool = self._extract_urls_from_tool_output(tool_output)
                collected_urls.extend(urls_from_tool)
                
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": self._get_tool_call_id(tool_call),
                    "content": tool_output
                })

            self.answer_generator.extend(tool_results)
            completion = await self.client.chat.completions.create(
                model=os.getenv("LLM_NAME"),
                messages=self.answer_generator
            )
            message = completion.choices[0].message

        agent_response = self._message_content_to_text(message)
        return {"agent_response": agent_response, "called_tools": called_tools, "urls": collected_urls}

    async def _ensure_openai_tools(self) -> None:
        if self._tools_loaded or not self.server_params:
            return
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
        except Exception:
            return

        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]
        self._tools_loaded = True

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if not self.server_params:
            return ""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
        except Exception as exc:
            return f"Tool {tool_name} failed: {exc}"

        return self._content_blocks_to_text(result.content)

    def _extract_urls_from_tool_output(self, tool_output: str) -> list[str]:
        """Extract URLs from structured tool output JSON."""
        try:
            data = json.loads(tool_output)
            if isinstance(data, dict) and 'urls' in data:
                return data.get('urls', [])
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def _message_content_to_text(self, message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, list):
            chunks = []
            for block in content:
                text = self._get_attr(block, "text")
                if text:
                    chunks.append(text)
                elif isinstance(block, dict):
                    text_value = block.get("text")
                    if text_value:
                        chunks.append(text_value)
            content = "\n".join(chunks)
        elif content is None and isinstance(message, dict):
            content = message.get("content")
        return content or ""

    def _content_blocks_to_text(self, blocks: list[Any]) -> str:
        chunks = []
        for block in blocks or []:
            text = self._get_attr(block, "text")
            if text:
                chunks.append(text)
            elif isinstance(block, dict):
                text_value = block.get("text")
                if text_value:
                    chunks.append(text_value)
        return "\n".join(chunks)

    def _extract_tool_calls(self, message: Any) -> list[Any]:
        # First check standard OpenAI tool_calls
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        if tool_calls:
            return tool_calls
        

        content = getattr(message, "content", None) or ""
        if "function call" in content.lower() and "<|role_sep|>" in content:
            try:
                _, raw_payload = content.split("<|role_sep|>", 1)
                tool_data = json.loads(raw_payload.strip())
                # Create a synthetic tool call object
                return [{
                    "id": "qwen_call_1",
                    "function": {
                        "name": tool_data.get("name"),
                        "arguments": json.dumps(tool_data.get("arguments", {}))
                    }
                }]
            except (ValueError, json.JSONDecodeError):
                pass
        
        return []

    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return default

    def _get_tool_call_id(self, tool_call: Any) -> str:
        """Get tool call ID from either dict or Pydantic object."""
        if isinstance(tool_call, dict):
            return tool_call.get("id", "")
        return getattr(tool_call, "id", "")
    
    def _get_tool_call_function_name(self, tool_call: Any) -> str:
        """Get function name from tool call."""
        if isinstance(tool_call, dict):
            func = tool_call.get("function", {})
            return func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
        func = getattr(tool_call, "function", None)
        if func is None:
            return ""
        if isinstance(func, dict):
            return func.get("name", "")
        return getattr(func, "name", "")
    
    def _get_tool_call_function_arguments(self, tool_call: Any) -> str:
        """Get function arguments from tool call."""
        if isinstance(tool_call, dict):
            func = tool_call.get("function", {})
            return func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
        func = getattr(tool_call, "function", None)
        if func is None:
            return "{}"
        if isinstance(func, dict):
            return func.get("arguments", "{}")
        return getattr(func, "arguments", "{}")

    def _route_judge(self, state: Graph) -> str:
        """Route based on judge score and fail count."""
        if state.fail_cnt > 4:
            return "fail"
        if state.judge_score > 7:
            return END
        return "prompt_generate"

    async def judge_node(self, state: Graph) -> dict:
        self.judge_generator.append({"role": "user", "content": state.agent_response})
        completion = await self.client.chat.completions.create(
            model=os.getenv("LLM_NAME"),
            messages=self.judge_generator
        )
        content = completion.choices[0].message.content
        parsed = json.loads(content)
        score = int(parsed.get("judge_score", 0))
        return {"judge_score": score, "fail_cnt": state.fail_cnt + 1}
    
    async def fail_node(self, state: Graph) -> dict:
        return {"fail_message": "Ошибка в генерации ответа. Переформулируйте вопрос"}

    async def graph_invoke(self, query) -> dict:
        res = await self.compiled_graph.ainvoke(Graph(query=query))
        return res