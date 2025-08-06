from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3_2-3b-instruct-hf',
        path='/home/aiops/fengxc/HF_models/Llama-3.2-3B-Instruct/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
