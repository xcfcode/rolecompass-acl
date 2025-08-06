from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='QwQ-32B-Preview',
        path='/home/aiops/fengxc/HF_models/QwQ-32B-Preview',
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
