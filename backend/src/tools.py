from langchain_core.tools import tool
import arxiv
from loguru import logger

@tool
def search_arxiv(query: str) -> str:
    """
    Search specifically for academic papers on ArXiv.
    Useful for finding scientific papers, technical reports, and research summaries.
    """
    logger.info(f"Searching ArXiv for: {query}")
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3, # 每个任务查3篇最相关的
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for r in client.results(search):
            # 格式化每篇论文的信息
            paper_info = (
                f"Title: {r.title}\n"
                f"Authors: {', '.join([a.name for a in r.authors])}\n"
                f"Published: {r.published.strftime('%Y-%m-%d')}\n"
                f"PDF Link: {r.pdf_url}\n"
                f"Summary: {r.summary.replace(chr(10), ' ')}\n"
            )
            results.append(paper_info)
            
        return "\n---\n".join(results) if results else "No papers found."
    except Exception as e:
        return f"ArXiv search failed: {str(e)}"