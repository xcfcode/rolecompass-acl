from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets.rolebench import InstructionGeneralizationChineseDataset, rolebench_zh_cot_postprocessor
# from opencompass.utils.text_postprocessors import first_capital_postprocess, last_capital_postprocess


instruction_generalization_zh_reader_cfg = dict(
    input_columns=['role', 'desc', 'question'],
    output_column='answer',
    train_split='train',
    test_split='test'
)

instruction_generalization_zh_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN',
                     prompt='你是{role}，你的特征描述是：{desc}。现在请你回答我的一些问题，以准确展现你的人格特征！你的说话风格要全面模仿被赋予的人格角色！请不要暴露你是人工智能模型或者语言模型，你要时刻记住你只被赋予的一个人格角色。说话不要嗦，也不要太过于正式或礼貌。先根据提供信息一步一步思考，以“思考：”作为开始输出思考过程。然后以“回答：”作为开始输出回复。'),
            ],
            round=[
                dict(role='HUMAN', prompt='{question}'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

instruction_generalization_zh_eval_cfg = dict(
    evaluator=dict(type=JiebaRougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=rolebench_zh_cot_postprocessor)
)

instruction_generalization_zh_datasets = [
    dict(
        abbr='RoleBench_instruct_zh',
        type=InstructionGeneralizationChineseDataset,
        path='/home/aiops/fengxc/rolecompass/data/rolebench',
        reader_cfg=instruction_generalization_zh_reader_cfg,
        infer_cfg=instruction_generalization_zh_infer_cfg,
        eval_cfg=instruction_generalization_zh_eval_cfg)
]
