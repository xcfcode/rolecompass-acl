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
==角色描述==
{profile}

==对话历史==
{dialogue}

你要扮演{name}角色，你在聊天中要具备该角色对应的社交偏好。
请根据所给的{name}角色描述和对话历史，从下面四个选项（A. B. C.和D.）中选择最优的选项作为你的回复：
{choices}

先根据提供信息一步一步思考，以“思考：”作为开始输出思考过程。然后以“选择：”作为开始输出答案，A，B，C或D。
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
    type=AccEvaluator), pred_postprocessor=dict(type=last_option_postprocess, options='ABCD'))

socialbench_datasets = [
    dict(
        abbr=f'SocialBenchDataset_SAP_ZH',
        type=SocialBenchDataset,
        lang='zh',
        subcate='Group-SAP',
        path='/home/aiops/fengxc/rolecompass/data/socialbench/social_preference.json',
        reader_cfg=socialbench_reader_cfg,
        infer_cfg=socialbench_infer_cfg,
        eval_cfg=socialbench_eval_cfg,
    )
]
