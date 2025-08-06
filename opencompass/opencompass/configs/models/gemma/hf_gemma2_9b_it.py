from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gemma2-9b-it-hf',
        path='/home/aiops/fengxc/HF_models/gemma-2-9b-it/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819',
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
