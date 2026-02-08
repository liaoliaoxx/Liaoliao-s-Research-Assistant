import operator
from typing import Annotated, List, TypedDict, Optional, Any
from pydantic import BaseModel, Field

# 定义单个任务的结构
class ResearchTask(BaseModel):
    id: int
    title: str
    intent: str
    query: str
    status: str = "pending"  # pending, in_progress, completed
    summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list) # 存储论文/网页链接或标题

# 定义整个图的全局状态
class AgentState(TypedDict):
    topic: str  # 用户输入的主题
    tasks: List[ResearchTask] # 任务列表
    # 使用 operator.add 实现追加模式，收集所有节点的日志/笔记/中间产物
    notes: Annotated[List[str], operator.add] 
    final_report: str