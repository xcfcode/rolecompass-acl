from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-32b-instruct-hf',
        path='/home/aiops/fengxc/HF_models/Qwen2.5-32B-Instruct/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
