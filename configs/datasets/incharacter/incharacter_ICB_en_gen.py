from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InCharacterDataset, IncharacterEvaluator

incharacter_reader_cfg = dict(
    input_columns=['character_instruction',
                   'experimenter', 'query', 'character_name'],
    output_column='label',
    train_split='test')


QUERY_TEMPLATE = """
{character_instruction}
{experimenter}: {query}
{character_name}:
""".strip()

incharacter_infer_cfg = dict(
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

incharacter_eval_cfg = dict(evaluator=dict(type=IncharacterEvaluator))

incharacter_datasets = [
    dict(
        abbr=f'InCharacter_ICB_EN',
        type=InCharacterDataset,
        lang='en',
        questionnaire_name='ICB',
        path='/home/aiops/fengxc/rolecompass/data/incharacter/',
        reader_cfg=incharacter_reader_cfg,
        infer_cfg=incharacter_infer_cfg,
        eval_cfg=incharacter_eval_cfg,
    )
]
