from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3_2-1b-instruct-hf',
        path='Llama-3.2-1B-Instruct',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
