import os
from functools import lru_cache
from dotenv import load_dotenv # 确保已安装 python-dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

# 1. 强制加载 .env 文件
load_dotenv()

class Configuration:
    @classmethod
    def from_env(cls):
        return cls()

    @property
    def llm_provider(self):
        return os.getenv("LLM_PROVIDER", "ollama")

    @property
    def model_name(self):
        return os.getenv("LLM_MODEL_ID", "llama3.2")

    @property
    def base_url(self):
        # 2. 修改：将 localhost 改为 127.0.0.1，解决 Mac 上的连接问题
        return os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434")

@lru_cache
def get_llm() -> BaseChatModel:
    config = Configuration.from_env()
    
    print(f"Loading LLM: {config.llm_provider} | Model: {config.model_name} | URL: {config.base_url}")
    
    if config.llm_provider == "ollama":
        return ChatOllama(
            model=config.model_name,
            base_url=config.base_url,
            temperature=0
        )
    else:
        return ChatOpenAI(
            model=config.model_name,
            api_key=os.getenv("LLM_API_KEY", "dummy"),
            base_url=config.base_url,
            temperature=0
        )