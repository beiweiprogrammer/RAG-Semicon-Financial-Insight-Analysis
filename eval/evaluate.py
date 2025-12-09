import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy)
from ragas import evaluate
import time
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential, 
    retry_if_exception_type
)
import openai
from ragas.llms import llm_factory
from llama_index.llms.openai import OpenAI
from ragas.llms import LlamaIndexLLMWrapper,LangchainLLMWrapper
@retry(
    retry=retry_if_exception_type(Exception), # 也可以指定具体的 openai.RateLimitError
    wait=wait_random_exponential(multiplier=1, max=240),
    stop=stop_after_attempt(8)
)
def safe_query(engine, question):
    return engine.query(question)
def final_evaluate(text_questions, final_route_engine, text_answers, save):
    if not save:
        answers = []
        contexts = []
        for question in text_questions:
            time.sleep(90)
            response = safe_query(final_route_engine, question)
            answers.append(response.response)
            retrieved_texts = [node.node.get_content() for node in response.source_nodes]
            contexts.append(retrieved_texts)
        data_dict = {'question':text_questions,
                    'answer':answers,
                    'contexts':contexts,
                    'ground_truth':text_answers}
        rag_dataset = Dataset.from_dict(data_dict)
        pd.DataFrame(data_dict).to_csv('answer_result.csv')
    else:
        test_dataset = pd.read_csv('answer_result.csv')
        question = []
        answer = []
        contexts = []
        truth = []
        for i in range(15):
            the_len = len(test_dataset.iloc[i]['contexts'])
            question.append(test_dataset.iloc[i]['question'])
            answer.append(test_dataset.iloc[i]['answer'])
            contexts.append([test_dataset.iloc[i]['contexts'][2:int(the_len/2)]])
            truth.append(test_dataset.iloc[i]['ground_truth'])
        data_dict = {'question':question,
             'answer':answer,
             'contexts':contexts,
             'ground_truth':truth}
        rag_dataset = Dataset.from_dict(data_dict)
    metrics = [
        answer_relevancy
    ]

    client = OpenAI()
    eval_llm = LlamaIndexLLMWrapper(client)
    results = evaluate(
            dataset=rag_dataset,
            metrics=metrics,
            batch_size = 1,
            llm=eval_llm,

        )
    return results
