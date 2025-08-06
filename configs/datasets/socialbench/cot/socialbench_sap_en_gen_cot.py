from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import SocialBenchDataset
from opencompass.utils.text_postprocessors import last_option_postprocess

socialbench_reader_cfg = dict(
    input_columns=['profile', 'dialogue', 'name', 'choices'],
    output_column='label',
    train_split='test')


QUERY_TEMPLATE = """
==Profiles==
{profile}

==Conversations==
{profile}

You are playing the role of {profile}, you need to embody the social preference of {profile}.
Based on the provided role profile and conversations, please choose the best option (A, B, C, or D) as your response:
{choices}

First, think step by step based on the provided information and start the thought process with “Thought:”. 
Then, start the answer with “Choice:” and provide an answer in the form of A, B, C, or D.""".strip()

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
    type=AccEvaluator), pred_postprocessor=dict(type=last_option_postprocess, options='ABCD'))

socialbench_datasets = [
    dict(
        abbr=f'SocialBenchDataset_SAP_EN',
        type=SocialBenchDataset,
        lang='en',
        subcate='Group-SAP',
        path='/home/aiops/fengxc/rolecompass/data/socialbench/social_preference.json',
        reader_cfg=socialbench_reader_cfg,
        infer_cfg=socialbench_infer_cfg,
        eval_cfg=socialbench_eval_cfg,
    )
]
