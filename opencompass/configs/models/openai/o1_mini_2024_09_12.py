from opencompass.models import OpenAISDK

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='o1-mini-2024-09-12',
        type=OpenAISDK,
        path='o1-mini-2024-09-12',
        key='',
        meta_template=api_meta_template,
        query_per_second=1,
        batch_size=1,
        temperature=1,
        # you can change it for large reasoning inference cost, according to: https://platform.openai.com/docs/guides/reasoning
        max_completion_tokens=1024),
]
