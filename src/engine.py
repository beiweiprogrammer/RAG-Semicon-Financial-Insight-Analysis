from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import pickle
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.query_engine import SubQuestionQueryEngine,RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
def create_router_engine(vector_index, summary_nvda, summary_amd):
    reranker = FlagEmbeddingReranker(
    top_n = 5,
    use_fp16=True
)
    nvda_engine = vector_index.as_query_engine(
    filters = MetadataFilters(filters=[ExactMatchFilter(key='company',value='NVIDA')]),
    similarity_top_k = 50,
    node_processors = [reranker]
)
    amd_engine = vector_index.as_query_engine(
    filters = MetadataFilters(filters=[ExactMatchFilter(key='company',value='AMD')]),
    similarity_top_k = 5,
    node_processors = [reranker]
)
    summary_engine_amd = summary_amd.as_query_engine(response_mode="tree_summarize")
    summary_engine_nvda = summary_nvda.as_query_engine(response_mode="tree_summarize")
    query_engine_tool = [
    QueryEngineTool(
        query_engine=nvda_engine,
        metadata=ToolMetadata(
            name = 'nvda basic 10_k',
            description='Used for answering NVDA 10-k detail or low level questions'
        )
    ),
    QueryEngineTool(
        query_engine=amd_engine,
        metadata=ToolMetadata(
            name = 'amd basic 10_k',
            description='Used for answering AMD 10-k detail or low level questions'
        )
    )
]
    sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tool,
    verbose=True
)
    query_engine_tool_summary = [
    QueryEngineTool(
        query_engine=summary_engine_nvda,
        metadata=ToolMetadata(
            name = 'nvda summary 10_k',
            description='Used for answering NVDA 10-k summary or high level questions'
        )
    ),
    QueryEngineTool(
        query_engine=summary_engine_amd,
        metadata=ToolMetadata(
            name = 'amd summary 10_k',
            description='Used for answering AMD 10-k summary or high level questions'
        )
    )
]
    sub_question_engine_summary = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tool_summary,
    verbose=True
)
    router_tools = [
    
    QueryEngineTool(
        query_engine=sub_question_engine,
        metadata = ToolMetadata(
            name='comparison_tool',
            description='Used for comparing two companies AMD and NVDA'
        )
    ),
    QueryEngineTool(
        query_engine= sub_question_engine_summary,
        metadata=ToolMetadata(
            name="summary_tool",
            description="used for answering questions which is related to summary or are higher-level"
        )
    )
]
    selector_llm = OpenAI(model="gpt-4.1-mini", temperature=0)
    final_route_engine = RouterQueryEngine(
    selector = LLMSingleSelector.from_defaults(llm = selector_llm),
    query_engine_tools = router_tools,
    verbose = True
)
    return final_route_engine