from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CrossDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

cross_reader_cfg = dict(
    input_columns=['character', 'question', 'summary'],
    output_column='label',
    train_split='test')


QUERY_TEMPLATE = """
You are a helpful assistant proficient in analyzing the motivation for the character's decision in novels. You will be given the profile about character {character} in a novel. Your task is to choose the most accurate primary motivation for the character's decision according to the character's profile.

Character Profile:
name: {character}
Summary of this character: {summary}

Question:
{question}

Your selection (You can only output A, B, C or D, and no other characters.):
""".strip()

cross_infer_cfg = dict(
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

cross_eval_cfg = dict(evaluator=dict(
    type=AccEvaluator), pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

cross_datasets = [
    dict(
        abbr=f'CrossDataset',
        type=CrossDataset,
        path='/home/aiops/fengxc/rolecompass/data/cross/motivation_dataset.json',
        persona_path='/home/aiops/fengxc/rolecompass/data/cross/truth_persona_all_dimension.json',
        reader_cfg=cross_reader_cfg,
        infer_cfg=cross_infer_cfg,
        eval_cfg=cross_eval_cfg,
    )
]
