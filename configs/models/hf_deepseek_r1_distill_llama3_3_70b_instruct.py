from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-R1-Distill-Llama-70B',
        path='/home/aiops/fengxc/HF_models/DeepSeek-R1-Distill-Llama-70B',
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=8),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
