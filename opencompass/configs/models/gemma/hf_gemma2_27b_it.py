from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gemma2-27b-it-hf',
        path='gemma-2-27b-it',
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=2),
        stop_words=['<end_of_turn>'],
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
