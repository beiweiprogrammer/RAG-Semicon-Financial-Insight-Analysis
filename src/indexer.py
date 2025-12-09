from llama_index.core import SummaryIndex, VectorStoreIndex, StorageContext, load_index_from_storage
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
def get_indexes(stored_node = None, rebuild=False):
    if stored_node is None and rebuild == False:
        raise ValueError('Please speicify stored_node or load summary index from' \
        'storage')
    elif stored_node is not None and rebuild == True:
        nodes_nvda = [i for i in stored_node if i.metadata['company'] == 'NVIDA']
        nodes_AMD = [i for i in stored_node if i.metadata['company'] == 'AMD']
        summary_nvda = SummaryIndex(nodes_nvda)
        summary_amd = SummaryIndex(nodes_AMD)
        perisis_dir = './summary_index'
        summary_amd.set_index_id('summary_amd')
        summary_nvda.set_index_id('summary_nvda')
        summary_nvda.storage_context.persist(f'{perisis_dir}/nvda')
        summary_amd.storage_context.persist(f'{perisis_dir}/amd')
        client = qdrant_client.QdrantClient(path='./qdrant_drive')
        vector_store = QdrantVectorStore(client=client, collection_name='financial_reports')
        storage_context = StorageContext.from_defaults(vector_store)
        vector_index = VectorStoreIndex(stored_node, storage_context=storage_context)
    else:
        client = qdrant_client.QdrantClient(path='./qdrant_drive')
        vector_store = QdrantVectorStore(client=client, collection_name='financial_reports')
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        ctx_amd = StorageContext.from_defaults(persist_dir='./summary_index/amd')
        summary_amd = load_index_from_storage(ctx_amd)
        ctx_nvda = StorageContext.from_defaults(persist_dir='./summary_index/nvda')
        summary_nvda = load_index_from_storage(ctx_nvda)
    return vector_index, summary_nvda, summary_amd



