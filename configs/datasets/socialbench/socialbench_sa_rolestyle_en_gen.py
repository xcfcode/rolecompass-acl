from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import SocialBenchDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

socialbench_reader_cfg = dict(
    input_columns=['profile', 'dialogue', 'name', 'choices'],
    output_column='label',
    train_split='test')


QUERY_TEMPLATE = """
==Profile==
{profile}

==Conversations==
{dialogue}

You are playing the role of {name}, you need to embody the knowledge and style of {name}.
Based on the provided role Profile and Conversations, please choose the best option (A, B, C, or D):
{choices}

Your selection:
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
    type=AccEvaluator), pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

socialbench_datasets = [
    dict(
        abbr=f'SocialBenchDataset_SA_ROLESTYLE_EN',
        type=SocialBenchDataset,
        lang='en',
        subcate='individual-sa-rolestyle',
        path='/home/aiops/fengxc/rolecompass/data/socialbench/self_awareness.json',
        reader_cfg=socialbench_reader_cfg,
        infer_cfg=socialbench_infer_cfg,
        eval_cfg=socialbench_eval_cfg,
    )
]
