from llama_cloud_services import LlamaParse
from llama_index.core.node_parser import SemanticSplitterNodeParser, MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
def load_and_parse(file_path, company_name):
    parser = LlamaParse(result_type='markdown',
                    verbose=True)
    document = parser.load_data(file_path)
    for i in document:
        i.metadata['company'] = company_name
        i.metadata['year'] = '2024'
    all_index = document

    # 表格解析器
    element_parser = MarkdownElementNodeParser(
        llm = OpenAI('gpt-4o'),
        num_workers=8
    )
    #语义切分器
    embed_model = OpenAIEmbedding()
    semantic_splitter = SemanticSplitterNodeParser(
        embed_model=embed_model
    )
    print('正在提取表格')
    node = element_parser.get_nodes_from_documents(all_index)
    base_nodes, objects = element_parser.get_nodes_and_objects(node)
    print("正在对文本进行语义切分...")
    text_node = semantic_splitter.get_nodes_from_documents(all_index)
    final_node = text_node + objects
    return final_node
