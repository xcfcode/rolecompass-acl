from opencompass.models import HuggingFacewithChatTemplate
import torch

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-R1-Distill-Llama-8B',
        path='/home/aiops/fengxc/HF_models/DeepSeek-R1-Distill-Llama-8B',
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=2),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
