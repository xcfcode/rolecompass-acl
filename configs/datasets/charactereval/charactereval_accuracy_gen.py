from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CharacterEvalDataset, CharacterEvalEvaluator

charactereval_reader_cfg = dict(
    input_columns=['role', 'context', 'system_message', 'round_num'],
    output_column='label',
    train_split='test')


# charactereval_infer_cfg = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             round=[
#                 dict(
#                     role="HUMAN", prompt="{system_message}\n\n下述内容为你和另外一个角色之间的对话：{context}。\n\n你的任务是根据对话上文，给出你作为{role}的一句话回复。格式为'{role}：回复'。"),
#             ]
#         )
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer)
# )

# charactereval_eval_cfg = dict(evaluator=dict(
#     type=CharacterEvalEvaluator, path="/home/aiops/fengxc/HF_models/BaichuanCharRM"))

# charactereval_infer_cfg = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             begin=[
#                 dict(role='SYSTEM', fallback_role='HUMAN',
#                      prompt='{system_message}'),
#             ],
#             round=[
#                 dict(role="HUMAN",
#                      prompt="对话历史：\n{context}。"),
#                 dict(role="BOT", prompt="{role}："),
#             ]
#         )
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer)
# )

# charactereval_eval_cfg = dict(evaluator=dict(
#     type=CharacterEvalEvaluator, path="/home/aiops/fengxc/HF_models/BaichuanCharRM"),    pred_role='BOT')


charactereval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{system_message}\n\n##对话历史##\n{context}\n\n##回复##\n{role}："
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

charactereval_eval_cfg = dict(evaluator=dict(
    type=CharacterEvalEvaluator, path="/home/aiops/fengxc/HF_models/BaichuanCharRM"))

charactereval_datasets = [
    dict(
        abbr='CHARACTER_EVAL_Accuracy',
        type=CharacterEvalDataset,
        metric_name='Accuracy',
        path='/home/aiops/fengxc/rolecompass/data/charactereval/',
        reader_cfg=charactereval_reader_cfg,
        infer_cfg=charactereval_infer_cfg,
        eval_cfg=charactereval_eval_cfg)
]
