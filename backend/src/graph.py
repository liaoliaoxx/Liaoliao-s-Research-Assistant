from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from state import AgentState
from src.nodes import planner_node, researcher_node, reporter_node

def route_to_researchers(state: AgentState):
    """Map step: 将任务分发给 worker"""
    tasks = state["tasks"]
    # 使用 Send API 并行发送任务
    return [Send("researcher", {"task": t}) for t in tasks]

def create_graph():
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("reporter", reporter_node)
    
    # 设置入口
    workflow.set_entry_point("planner")
    
    # 规划 -> (并行) -> 研究
    workflow.add_conditional_edges("planner", route_to_researchers, ["researcher"])
    
    # 研究 -> 报告 (LangGraph 会自动等待所有并行分支完成才进入下一步)
    workflow.add_edge("researcher", "reporter")
    
    # 报告 -> 结束
    workflow.add_edge("reporter", END)
    
    return workflow.compile()