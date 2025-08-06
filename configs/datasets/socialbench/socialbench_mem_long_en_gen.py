
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SocialBenchDataset, SocialBenchMEMEvaluator


socialbench_reader_cfg = dict(
    input_columns=['profile', 'dialogue', 'name'],
    output_column='label',
    train_split='test')


QUERY_TEMPLATE = """
==Profile==
{profile}

==Conversations==
{dialogue}

You are playing the role of {name}, you need to embody the knowledge and style of {name}.
Based on the provided role Profile and Conversations, you must produce a reply as the Assistant to response to the latest User's message (one term is enough):
Assistant: 
""".strip()

socialbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,
                    generation_kwargs=dict(do_sample=False))
)

socialbench_eval_cfg = dict(evaluator=dict(
    type=SocialBenchMEMEvaluator))


socialbench_datasets = [
    dict(
        abbr=f'SocialBenchDataset_MEM_LONG_EN',
        type=SocialBenchDataset,
        lang='en',
        subcate='individual-mem-long',
        path='/home/aiops/fengxc/rolecompass/data/socialbench/conversation_memory.json',
        reader_cfg=socialbench_reader_cfg,
        infer_cfg=socialbench_infer_cfg,
        eval_cfg=socialbench_eval_cfg,
    )
]
