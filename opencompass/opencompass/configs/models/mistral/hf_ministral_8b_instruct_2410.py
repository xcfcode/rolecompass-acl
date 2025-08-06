from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="ministral-8B-instruct-2410-hf",
        path="/home/aiops/fengxc/HF_models/Ministral-8B-Instruct-2410/models--mistralai--Ministral-8B-Instruct-2410/snapshots/4847e87e5975a573a2a190399ca62cd266c899ad",
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
