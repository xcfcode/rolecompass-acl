
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SocialBenchDataset, SocialBenchCOTMEMEvaluator


socialbench_reader_cfg = dict(
    input_columns=['profile', 'dialogue', 'name'],
    output_column='label',
    train_split='test')


QUERY_TEMPLATE = """
==角色描述==
{profile}

==对话历史==
{dialogue}

你要扮演{name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{name}角色描述和对话历史，根据最后一个User的对话再补充一轮你作为Assistant的回复（一轮就好）：

先根据提供信息一步一步思考，以“思考：”作为开始输出思考过程。然后以“Assistant:”作为开始输出回复。 
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
    type=SocialBenchCOTMEMEvaluator))


socialbench_datasets = [
    dict(
        abbr=f'SocialBenchDataset_MEM_LONG_ZH',
        type=SocialBenchDataset,
        lang='zh',
        subcate='individual-mem-long',
        path='/home/aiops/fengxc/rolecompass/data/socialbench/conversation_memory.json',
        reader_cfg=socialbench_reader_cfg,
        infer_cfg=socialbench_infer_cfg,
        eval_cfg=socialbench_eval_cfg,
    )
]
