import json
import asyncio
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .graph import create_graph

app = FastAPI()

# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str

# 初始化图
research_graph = create_graph()

async def event_generator(topic: str) -> AsyncIterator[str]:
    """
    适配器：将 LangGraph 事件转换为前端兼容的 SSE 格式。
    """
    # 1. 发送初始化状态
    yield f"data: {json.dumps({'type': 'status', 'message': '正在规划科研路径...'}, ensure_ascii=False)}\n\n"
    
    inputs = {"topic": topic, "tasks": [], "notes": []}
    
    # 运行图并流式获取事件
    async for event in research_graph.astream_events(inputs, version="v1"):
        kind = event["event"]
        name = event["name"]
        data = event["data"]
        
        # --- 阶段 1: 规划完成 ---
        if kind == "on_chain_end" and name == "planner":
            output = data.get("output")
            if output and "tasks" in output:
                tasks = output["tasks"]
                # 转换为前端 Todo 格式
                todo_list_payload = {
                    "type": "todo_list",
                    "tasks": [t.model_dump() for t in tasks]
                }
                yield f"data: {json.dumps(todo_list_payload, ensure_ascii=False)}\n\n"
                
                # 发送第一条状态更新
                yield f"data: {json.dumps({'type': 'status', 'message': f'已生成 {len(tasks)} 个研究任务，准备开始并行搜索...'}, ensure_ascii=False)}\n\n"

        # --- 阶段 2: 任务执行中 (Researcher) ---
        # 捕获 Researcher 开始执行的信号
        if kind == "on_chain_start" and name == "researcher":
            # 我们可以从 input 中获取 task 信息
            task_input = data.get("input", {}).get("task")
            if task_input:
                # 通知前端任务开始
                yield f"data: {json.dumps({'type': 'task_status', 'task_id': task_input.id, 'status': 'in_progress'}, ensure_ascii=False)}\n\n"
        
        # 捕获工具调用 (模拟前端的 tool_call)
        if kind == "on_tool_start":
            # 这里的 metadata 需要我们在 graph 运行时注入，或者简单泛化
            yield f"data: {json.dumps({'type': 'status', 'message': f'正在调用 ArXiv 检索相关论文...'}, ensure_ascii=False)}\n\n"

        # 捕获 Researcher 完成
        if kind == "on_chain_end" and name == "researcher":
            output = data.get("output")
            # 注意：output 只有 {"notes": [...]}，我们需要重新构建上下文
            # 在实际生产中，我们可以把 task_id 传在 output 里
            # 这里简化处理：发送一个通用的完成信号，前端可能不会精确更新特定卡片的 summary，
            # 除非我们修改 researcher_node 让它返回 task_id
            pass 

        # --- 阶段 3: 报告生成 ---
        if kind == "on_chain_end" and name == "reporter":
            final_report = data.get("output", {}).get("final_report")
            if final_report:
                yield f"data: {json.dumps({'type': 'final_report', 'report': final_report}, ensure_ascii=False)}\n\n"
    
    # 结束信号
    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

@app.post("/research/stream")
async def stream_research(payload: ResearchRequest):
    return StreamingResponse(
        event_generator(payload.topic),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)