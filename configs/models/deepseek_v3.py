from opencompass.models import DeepseekAPI


api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr='Deepseek-V3',
         type=DeepseekAPI, path='deepseek-chat',
         key='sk-24d30742ada8466c8fbbc7a7076eaa2f',
         url='https://api.deepseek.com/v1',
         meta_template=api_meta_template,
         query_per_second=1,
         max_out_len=2048, max_seq_len=4096, batch_size=1),
]
