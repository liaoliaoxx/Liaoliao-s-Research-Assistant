import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState, ResearchTask
from src.config import get_llm
from src.tools import search_arxiv

# --- 1. Planner Node ---
class PlanResponse(BaseModel):
    """The structured list of research tasks."""
    tasks: List[ResearchTask]

def planner_node(state: AgentState):
    llm = get_llm()
    topic = state["topic"]
    
    # 强制让 LLM 输出结构化数据
    structured_llm = llm.with_structured_output(PlanResponse)
    
    system_prompt = (
        "你是一名资深的科研顾问。请将用户的研究主题拆解为 3-5 个具体的学术调研任务。"
        "任务应覆盖：现有相关工作(Related Work)、核心方法论(Methodology)、基准对比(Benchmarks)。"
        "每个任务必须包含一个针对 ArXiv 的具体搜索查询词(query)。"
    )
    
    try:
        plan = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"研究主题：{topic}")
        ])
        return {"tasks": plan.tasks}
    except Exception as e:
        # 降级处理
        fallback_task = ResearchTask(
            id=1, title="综述调研", intent="了解背景", query=topic
        )
        return {"tasks": [fallback_task]}

# --- 2. Researcher Node (Worker) ---
# 注意：这个 Node 接收特殊的参数，因为它是由 Send API 调用的
def researcher_node(state: dict):
    # state 这里不仅包含全局状态，还包含被分发的 task
    task: ResearchTask = state["task"]
    llm = get_llm()
    
    # 绑定工具
    llm_with_tools = llm.bind_tools([search_arxiv])
    
    # 1. 执行搜索 (这里简化为让模型自己决定调不调用，或者强制调用)
    # 为了保证效果，我们强制拼接 Prompt
    search_msg = llm_with_tools.invoke([
        SystemMessage(content="你是科研助手。请调用 search_arxiv 工具获取信息。"),
        HumanMessage(content=f"请搜索关于 '{task.query}' 的论文。")
    ])
    
    tool_output = "未检索到信息"
    sources = []
    
    # 解析工具调用
    if search_msg.tool_calls:
        for tool_call in search_msg.tool_calls:
            if tool_call["name"] == "search_arxiv":
                tool_output = search_arxiv.invoke(tool_call["args"])
                # 简单提取一下 source url 用于前端展示
                if "http" in tool_output:
                    sources.append("ArXiv Sources (见报告详情)")

    # 2. 生成总结
    summarize_prompt = ChatPromptTemplate.from_template(
        "你是一名学术研究员。请根据以下检索到的论文摘要，为任务“{title}”撰写一段简洁的学术综述。\n\n"
        "检索内容：\n{context}\n\n"
        "要求：\n1. 引用核心结论和数据。\n2. 使用 Markdown 格式。\n3. 如果没有内容，说明未找到。"
    )
    
    chain = summarize_prompt | llm
    summary = chain.invoke({"title": task.title, "context": tool_output})
    
    # 更新任务状态
    updated_task = task.model_copy()
    updated_task.status = "completed"
    updated_task.summary = summary.content
    updated_task.sources = sources
    
    # 构建笔记条目 (会被追加到全局 notes)
    note_content = (
        f"### 任务 {task.id}: {task.title}\n"
        f"**Intent**: {task.intent}\n"
        f"**Summary**: \n{summary.content}\n"
        f"**Sources**: {sources}\n"
        f"---\n"
    )
    
    return {
        "notes": [note_content], 
        # 我们不能直接在这里更新 tasks 列表，因为并行写入会有问题
        # 在 LangGraph 中，通常通过 Reducer 或者在父节点汇聚结果
        # 这里为了简化，我们仅通过 notes 传递结果，Reporter 会重新解析 notes
        # 或者利用 LangGraph 的 reducer 特性更新 tasks（较复杂），
        # 简单方案：Reporter 节点根据 notes 重写 final_report
    }

# --- 3. Reporter Node ---
def reporter_node(state: AgentState):
    llm = get_llm()
    notes = state.get("notes", [])
    topic = state["topic"]
    
    context = "\n".join(notes)
    
    prompt = ChatPromptTemplate.from_template(
        "你是一名顶级期刊审稿人。请根据以下收集到的任务笔记，撰写一份关于“{topic}”的科研可行性分析报告。\n\n"
        "所有笔记内容：\n{context}\n\n"
        "报告结构：\n"
        "1. **Abstract**: 核心发现摘要\n"
        "2. **Literature Review**: 分类综述\n"
        "3. **Gap Analysis**: 现有研究的不足与机会点\n"
        "4. **References**: 涉及的论文列表\n"
    )
    
    chain = prompt | llm
    response = chain.invoke({"topic": topic, "context": context})
    
    return {"final_report": response.content}