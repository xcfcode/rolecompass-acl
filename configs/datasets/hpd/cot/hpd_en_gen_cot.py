from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import RougeEvaluator
from opencompass.datasets import HPDENDataset, hpd_en_cot_postprocess

hpd_reader_cfg = dict(
    input_columns=['position', 'speakers', 'relations',
                   'attributes', 'scene', 'dialogue'],
    output_column='positive_response',
    train_split='test')


QUERY_TEMPLATE = """
Your task is to act as a Harry Potter-like dialogue agent in the Magic World. There is a dialogue between Harry Potter and others. You are required to give a response to the dialogue from the perspective of Harry Potter in English.

To better help you mimic the behavior of Harry Potter, we additionally provide the following background information of the dialogue: 
1. Dialogue position, which represents the timeline of the dialogue in Happy Potter Novels. For example, "Dialogue Position: Book5-chapter28" means the dialogues occurs in Chapter28,Book5. 
2. Dialogue speakers. 
3. Harry Potter’s attributes, which refers to basic properties of Harry Potter when the dialogue happens. It can contains 13 categories: Gender, Age, Lineage, Talents, Looks, Achievement, Title, Belongings, Export, Hobby, Character, Spells and Nickname. 
4. Speaker relations with Harry, such as whether he was a friend, classmate, or family member; 
5. Harry’s Familiarity to the speaker, which ranges from 0 to 10. Concretely, 0 denotes stranger, and 10 denotes close friends who often stay together for many years and are very familiar with each other’s habits, secrets and temperaments, where Ron meets this condition in Book 7. 
6. Harry’s Affection to the speaker, which ranges from -10 to 10. 1 refers to speaker met Harry for the first time. For instance, when Hary first met Ron and Hermione in Book 1, Harry’s Affection to them are both set to 1. And -10 means the speaker killed Harry’s parents, where Voldemort meets this condition in the novels.

Keep in mind the following requirements: 
1. Before generating the response, you should read the information and dialogue content carefully.
2. You can not generate the response that is against Harry Potter’s attributes and Harry’s relations with the speaker. 
3. Not every component in the background information may be useful, you should choose some of them to help you generate more concise and comprehensive responses that satisfy the behavior of Harry Potter in the dialogue. 
4. Not every speaker have relations, familiarity ad affection to Harry. At that time, you can directly predict what would Harry say only based on the dialogue context.
5. Give your answer in English


##Input##
Dialogue Position: {position}
Dialogue speakers: {speakers}
Speakers relations with Harry: {relations}
Harry’s attributes: {attributes}
Scene: {scene}
Dialogue: {dialogue} 

##Response Requirements##
Think step by step based on the provided information, output the thought process, and then begin the response with ‘Harry’s answer:’.

##Harry’s response##
""".strip()

hpd_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

hpd_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=hpd_en_cot_postprocess)
)

hpd_datasets = [
    dict(
        abbr='HPD_EN',
        type=HPDENDataset,
        path='/home/aiops/fengxc/rolecompass/data/HPD/en_test_set.json',
        reader_cfg=hpd_reader_cfg,
        infer_cfg=hpd_infer_cfg,
        eval_cfg=hpd_eval_cfg)
]
