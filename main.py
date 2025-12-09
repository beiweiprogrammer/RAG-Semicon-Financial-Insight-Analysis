from dotenv import load_dotenv
load_dotenv()
import nest_asyncio
nest_asyncio.apply()
from src.parser import load_and_parse
from src.engine import create_router_engine
from src.indexer import get_indexes
from eval.dataset import text_answers, text_questions
from eval.evaluate import final_evaluate
def main():
    # final_node_amd = load_and_parse('smicon_insight_rag/pdf_data/amd_10k.pdf','amd')
    # print(1)
    # final_node_nvda = load_and_parse('smicon_insight_rag/pdf_data/nvda_10k.pdf','nvda')
    # print(2)
    # final_node = final_node_amd + final_node_nvda
    vector_index, summary_nvda, summary_amd = get_indexes(stored_node = None, rebuild=True)
    print(3)
    final_engine = create_router_engine(vector_index, summary_nvda, summary_amd)
    print(final_evaluate(text_answers=text_answers, text_questions=text_questions, final_route_engine=final_engine, save=True))

if __name__ == '__main__':
    main()
