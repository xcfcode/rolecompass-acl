from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="ministral-8B-instruct-2410-hf",
        path="Ministral-8B-Instruct-2410",
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
