from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import RougeEvaluator
from opencompass.datasets.rolebench import RoleGeneralizationEnglishDataset, rolebench_en_cot_postprocessor

role_generalization_eng_reader_cfg = dict(
    input_columns=['role', 'desc', 'question'],
    output_column='answer',
    train_split='train',
    test_split='test'
)

role_generalization_eng_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are {role}, your description is: {desc}. Now please answer some questions to accurately show your personality traits! Your speaking style should fully imitate the personality role assigned to you! Please do not expose that you are an artificial intelligence model or a language model, you must always remember that you are only assigned one personality role. First, think step by step based on the provided information and start the thought process with “Thought:”. Then, begin the response with “Answer:”.'),
            ],
            round=[
                dict(role='HUMAN', prompt='{question}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

role_generalization_eng_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=rolebench_en_cot_postprocessor)
)

role_generalization_eng_datasets = [
    dict(
        abbr='RoleBench_role_eng',
        type=RoleGeneralizationEnglishDataset,
        path='/home/aiops/fengxc/rolecompass/data/rolebench',
        reader_cfg=role_generalization_eng_reader_cfg,
        infer_cfg=role_generalization_eng_infer_cfg,
        eval_cfg=role_generalization_eng_eval_cfg)
]
