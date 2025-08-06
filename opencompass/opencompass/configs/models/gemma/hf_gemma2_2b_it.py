from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gemma2-2b-it-hf',
        path='/home/aiops/fengxc/HF_models/gemma-2-2b-it/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8',
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<end_of_turn>'],
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
