from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-0.5b-instruct-hf',
        path='/home/aiops/fengxc/HF_models/Qwen2.5-0.5B-Instruct/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
