from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-70b-instruct-hf',
        path='/home/aiops/fengxc/HF_models/Llama-3.3-70B-Instruct/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
