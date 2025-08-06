from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-72b-instruct-hf',
        path='Qwen2.5-72B-Instruct',
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
