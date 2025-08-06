import codecs
import json
import collections
from datasets import Dataset, DatasetDict
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator
from .base import BaseDataset
from tqdm import tqdm


def format_question(dialogue, choices=None):
    conversations = ""
    for con in dialogue:
        role = con['from']
        text = con['value']
        conversations += f"{role}: {text}\n"

    options = ""
    if choices is not None:
        for choice, text in choices.items():
            options += f"{choice}. {text}\n"
    Output = collections.namedtuple('Output', ['dialogue', 'options'])
    return Output(dialogue=conversations, options=options)


@LOAD_DATASET.register_module()
class SocialBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, lang: str, subcate: str):
        dataset = DatasetDict()
        for split in ['test']:
            f = codecs.open(path, "r", "utf-8")
            data = json.load(f)
            dataset_list = []
            for item in data:
                # choose language zh or en
                if lang == item['meta']['lang'].strip():
                    # Conversation Memory
                    if subcate.lower() in ["individual-mem-long", "individual-mem-short"]:
                        if subcate.lower() == item['meta']['category'].lower():
                            outputs = format_question(
                                dialogue=item['dialogue'])
                            dataset_list.append({
                                'dialogue': outputs.dialogue,
                                'instruction': item['instruction'],
                                'label': item['label'],
                                'lang': item['meta']['lang'],
                                'name': item['meta']['name'],
                                'profile': item['meta']['profile'][item['meta']['name']],
                                'category': item['meta']['category']
                            })
                    # Emotional Perception
                    elif subcate.lower() in ["individual-ep-dialogueemotiondetect", "individual-ep-humorsarcasmdetect", "individual-ep-situationunderstanding"]:
                        if subcate.lower() == item['meta']['category'].lower():
                            outputs = format_question(
                                dialogue=item['dialogue'], choices=item['choices'])
                            dataset_list.append({
                                'dialogue': outputs.dialogue,
                                'instruction': item['instruction'],
                                'choices': outputs.options,
                                'label': item['label'],
                                'lang': item['meta']['lang'],
                                'category': item['meta']['category']
                            })
                    # Self-Awareness
                    elif subcate.lower() in ["individual-sa-rolestyle", "individual-sa-roleknowledge"]:
                        if subcate.lower() == item['meta']['category'].lower():
                            outputs = format_question(
                                dialogue=item['dialogue'], choices=item['choices'])
                            dataset_list.append({
                                'dialogue': outputs.dialogue,
                                'instruction': item['instruction'],
                                'choices': outputs.options,
                                'label': item['label'][0].strip(),
                                'lang': item['meta']['lang'],
                                'name': item['meta']['name'],
                                'profile': item['meta']['profile'][item['meta']['name']],
                                'category': item['meta']['category']
                            })
                    # Social Preference
                    elif subcate.lower() in ["group-sap"]:
                        outputs = format_question(
                            dialogue=item['dialogue'], choices=item['choices'])
                        dataset_list.append({
                            'dialogue': outputs.dialogue,
                            'instruction': item['instruction'],
                            'choices': outputs.options,
                            'label': item['label'][0].strip(),
                            'lang': item['meta']['lang'],
                            'name': item['meta']['name'],
                            'profile': item['meta']['profile'][item['meta']['name']],
                            'category': item['meta']['category']
                        })
            dataset[split] = Dataset.from_list(dataset_list)

        return dataset


@ICL_EVALUATORS.register_module()
class SocialBenchMEMEvaluator(BaseEvaluator):
    """https://github.com/X-PLUG/SocialBench/blob/1287b67034f6c76c4b3b068c51a3a9300f85ee0a/dataset.py#L213C5-L213C41"""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        predictions = [
            prediction.lower() for prediction in predictions
        ]
        processed_answers = [[j for j in i]
                             for i in references]

        cnt = 0
        details = []
        for pred, answers in zip(predictions, processed_answers):
            detail = {'pred': pred, 'answer': answers}

            one_score = 0
            for keyword in answers:
                one_score += 1 if keyword.lower() in pred else 0
            cnt += one_score / len(answers)
            details.append(detail)

        score = cnt / len(predictions) * 100
        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class SocialBenchCOTMEMEvaluator(BaseEvaluator):
    """https://github.com/X-PLUG/SocialBench/blob/1287b67034f6c76c4b3b068c51a3a9300f85ee0a/dataset.py#L213C5-L213C41"""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        processed_predictions = []
        for prediction in predictions:
            if "Assistant:" in prediction:
                prediction = prediction.split("Assistant:")[-1]
            processed_predictions.append(prediction.lower())
        predictions = processed_predictions

        processed_answers = [[j for j in i]
                             for i in references]

        cnt = 0
        details = []
        for pred, answers in zip(predictions, processed_answers):
            detail = {'pred': pred, 'answer': answers}

            one_score = 0
            for keyword in answers:
                one_score += 1 if keyword.lower() in pred else 0
            cnt += one_score / len(answers)
            details.append(detail)

        score = cnt / len(predictions) * 100
        return {'score': score, 'details': details}
