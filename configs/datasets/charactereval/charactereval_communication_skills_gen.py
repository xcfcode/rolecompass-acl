from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CharacterEvalDataset, CharacterEvalEvaluator

charactereval_reader_cfg = dict(
    input_columns=['role', 'context', 'system_message', 'round_num'],
    output_column='label',
    train_split='test')


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
        abbr='CHARACTER_EVAL_Communication_skills',
        type=CharacterEvalDataset,
        metric_name='Communication_skills',
        path='/home/aiops/fengxc/rolecompass/data/charactereval/',
        reader_cfg=charactereval_reader_cfg,
        infer_cfg=charactereval_infer_cfg,
        eval_cfg=charactereval_eval_cfg)
]
