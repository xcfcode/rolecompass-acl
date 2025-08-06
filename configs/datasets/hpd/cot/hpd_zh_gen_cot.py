from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets import HPDZHDataset, hpd_zh_cot_postprocess

hpd_reader_cfg = dict(
    input_columns=['position', 'speakers', 'relations',
                   'attributes', 'scene', 'dialogue'],
    output_column='positive_response',
    train_split='test')


QUERY_TEMPLATE = """
你的任务是扮演一个魔法世界中类似《哈利·波特》的对话代理。以下是一段关于哈利·波特与他人之间的对话。你需要从哈利·波特的视角给出对话的回应。

为了更好地模仿哈利·波特的行为，提供以下背景信息：
1. 对话位置：表示对话在《哈利·波特》系列小说中的时间线。例如，“对话位置：第五部-第28章”表示这段对话发生在第五部第28章中。
2. 对话发言者：列出与哈利对话的人物。
3. 哈利的属性：指对话发生时哈利·波特的基本特征，包括13个类别：性别、年龄、血统、天赋、外貌、成就、头衔、物品、专业、爱好、性格、咒语和昵称。
4. 发言者与哈利的关系：例如是朋友、同学还是家人。
5. 哈利对发言者的熟悉度：范围为0到10。具体而言，0表示陌生人，10表示多年来经常在一起、非常熟悉彼此习惯、秘密和性格的挚友。例如，在第七部中，罗恩就符合这个条件。
6. 哈利对发言者的好感度：范围为-10到10。1表示发言者第一次见到哈利，例如哈利在第一部中初遇罗恩和赫敏时，对他们的好感度都是1。而-10表示发言者杀死了哈利的父母，例如伏地魔在小说中就符合这一条件。

##请注意以下要求##
1. 在生成回答之前，你需要仔细阅读提供的信息和对话内容。
2. 回答不能与哈利的属性或他与发言者的关系相悖。
3. 并非每个背景信息都必须使用，你应该选择一些信息来生成简洁且符合对话语境的回答。
4. 并非所有发言者都与哈利有关系、熟悉度或好感度。在这种情况下，你仅根据对话内容预测哈利会说什么。
5. 用中文回复。

##输入##
对话位置：{position}
对话发言者：{speakers}
发言者与哈利的关系：{relations}
哈利的属性：{attributes}
场景：{scene}
对话内容：{dialogue}

##回答要求##
结合提供的信息先一步一步思考，输出思考内容，然后以“哈利的回答：”作为开始进行回答。

##哈利的回答##
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
    evaluator=dict(type=JiebaRougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=hpd_zh_cot_postprocess)
)

hpd_datasets = [
    dict(
        abbr='HPD_ZH',
        type=HPDZHDataset,
        path='/home/aiops/fengxc/rolecompass/data/HPD/cn_test_set.json',
        reader_cfg=hpd_reader_cfg,
        infer_cfg=hpd_infer_cfg,
        eval_cfg=hpd_eval_cfg)
]
