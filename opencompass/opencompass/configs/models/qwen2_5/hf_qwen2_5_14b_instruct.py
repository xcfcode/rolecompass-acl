from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-14b-instruct-hf',
        path='/home/aiops/fengxc/HF_models/Qwen2.5-14B-Instruct/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
