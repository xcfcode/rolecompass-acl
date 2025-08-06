from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-72b-instruct-hf',
        path='/home/aiops/fengxc/HF_models/Qwen2.5-72B-Instruct/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
