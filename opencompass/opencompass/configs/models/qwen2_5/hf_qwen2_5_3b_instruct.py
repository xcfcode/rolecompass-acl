from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-3b-instruct-hf',
        path='/home/aiops/fengxc/HF_models/Qwen2.5-3B-Instruct/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
