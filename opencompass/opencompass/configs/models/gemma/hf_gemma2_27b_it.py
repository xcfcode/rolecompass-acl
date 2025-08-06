from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gemma2-27b-it-hf',
        path='/home/aiops/fengxc/HF_models/gemma-2-27b-it/models--google--gemma-2-27b-it/snapshots/aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0',
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=2),
        stop_words=['<end_of_turn>'],
        model_kwargs=dict(
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    )
]
