from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-R1-Distill-Qwen-14B',
        path='/home/aiops/fengxc/HF_models/DeepSeek-R1-Distill-Qwen-14B',
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=4),
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
