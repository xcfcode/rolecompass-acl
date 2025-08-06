import os
import json
import copy
import tqdm
import torch
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS, ICL_EVALUATORS
from opencompass.openicl.icl_evaluator import BaseEvaluator, LMEvaluator
from .base import BaseDataset

from .BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from .BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer


# def concat_messages(conversations, role, system):
#     """
#     "佟湘玉：展堂。（下楼看到老邢）老邢。\n老邢：（拿出一张纸）你来得正好。\n佟湘玉：展堂呢？\n老邢：赶紧在这上面签个字。\n佟湘玉：（不解）签啥字呀？（拿过纸）\n老邢：（坐下）他是你的伙计呀，又是在你地盘被收押的，你不得签个字担保呀？（递过毛笔）\n佟湘玉：老邢，展堂没有得狂犬病，那都是小郭……\n老邢：（打断掌柜的）你先把字签了再说。（掌柜的停口）再说这事儿你跟我说没用，你最好问问娄知县去。\n佟湘玉：（不屑）还娄知县？\n老邢：（起身，走到客栈门口，指着地面）看看，看看。\n佟湘玉：你们还敢打人？\n老邢：这不是打人，这是老白咬的。（坐）"


#     "id": 10991,
#     "role": "佟湘玉",
#     "novel_name": "武林外传",
#     "context": "郭芙蓉：掌柜的\n佟湘玉：吵吵啥干活去\n郭芙蓉：你怎么穿成这个样子\n佟湘玉：这就是我婆婆给我做的你觉得怎么样呀\n郭芙蓉：还凑合吧但是这是飞贼的打扮你穿成这样好像不太合适吧"

#     """
#     round = []
#     first_query = system
#     if conversations[0]['from'] == role:
#         first_response = f"好的！现在我来扮演{role}。" + \
#             "我首先发话：" + conversations[0]['value']
#     else:
#         first_response = f"好的！现在我来扮演{role}。"

#     round.append({"role": "HUMAN", "prompt": first_query})
#     round.append({"role": "BOT", "prompt": first_response})

#     for i in range(len(conversations)):
#         if conversations[i]['from'] == role:
#             if i == 0:
#                 continue
#             else:
#                 assert conversations[i-1]['from'] != role
#                 query = f"{conversations[i-1]['from']
#                            }：" + conversations[i-1]['value']
#                 response = f"{conversations[i]['from']
#                               }：" + conversations[i]['value']
#             round.append({"role": "HUMAN", "prompt": query})
#             round.append({"role": "BOT", "prompt": response})
#     assert conversations[-1]['from'] != role

#     query = f"{conversations[-1]['from']}：" + conversations[-1]['value']
#     round.append({"role": "HUMAN", "prompt": query})

#     # post-process
#     dialogue = ""
#     for one_round in round:
#         dialogue += one_round["prompt"]

#     # list of dict to string
#     round_string = json.dumps(round, ensure_ascii=False)
#     return round_string


# def make_inputs(context):
#     dialogues = context.split('\n')
#     inputs = []
#     for dial in dialogues:
#         role = dial.split("：")[0]
#         dial = "：".join(dial.split("：")[1:])
#         inputs.append({"from": role, "value": dial})
#     return inputs


EN2ZH = {'Accuracy': '知识准确率', 'Utterance': '言语一致性', 'Consistency': '对话一致性', 'Behavior': '行为一致性', 'Hallucination': '知识幻觉性',
         'Humanlikeness': '类人程度', 'Communication_skills': '交流技巧', 'Empathy': '共情度', 'Coherence': '对话连贯性', 'Diversity': '表现多样性', 'Exposure': '知识曝光度', 'Fluency': '对话流利度'}


@LOAD_DATASET.register_module()
class CharacterEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, metric_name: str, *args, **kwargs):
        # init
        dataset = DatasetDict()
        dataset_list = []

        # load data
        with open(os.path.join(path, f"divide_by_metric/{metric_name}.json"), 'r') as f:
            datas = json.load(f)
        with open(os.path.join(path, "character_profiles.json"), 'r') as f:
            role_informations = json.load(f)

        for data in datas:
            # get input information
            context = data['context']
            role = data['role']
            role_information = role_informations[role]
            role_system = f'''{role_information}
                现在请你扮演一个角色扮演专家。你的任务是根据上述信息扮演{role}，并根据给定对话历史进行一句话对话回复。'''

            # format
            data['role_information'] = str(role_information)
            data['metric_en'] = metric_name
            data['metric_zh'] = EN2ZH[metric_name]

            # messages, query = concat_messages(
            #     make_inputs(context), role, role_system)

            dataset_list.append({
                'role': role,
                'context': context,
                'system_message': role_system,
                'label': data
            })
        dataset['test'] = Dataset.from_list(dataset_list)

        return dataset


def format_input(example):
    input_text = "<RoleInfo>\n\n" \
        + example['role_information'] + "\n\n<Context>\n\n" + example['context'] + \
        "\n\n<Response>\n\n" + example['model_output'] + \
        "\n\n<Dimension>\n\n" + example["metric_zh"]
    return input_text


# @ICL_EVALUATORS.register_module()
# class CharacterEvalEvaluator(BaseEvaluator):

#     def __init__(self, path) -> None:
#         # load baichuan character reward model: https://huggingface.co/morecry/BaichuanCharRM
#         self.tokenizer = BaichuanTokenizer.from_pretrained(path)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.padding_side = "left"
#         self.rm_model = BaichuanCharRM.from_pretrained(
#             path, torch_dtype=torch.bfloat16).cuda()
#         super().__init__()

#     def score(self, predictions, references):

#         if len(predictions) != len(references):
#             return {
#                 'error': 'predictions and references have different '
#                 'length'
#             }

#         # post-process
#         records = []
#         for prediction, reference in zip(predictions, references):
#             # Prevent continuous generation
#             clean_prediction = prediction.split("\n")[0]
#             reference["model_output"] = clean_prediction
#             records.append(reference)

#         # Run Character Reward Model
#         # codes are borrowed from: https://github.com/morecry/CharacterEval/blob/main/run_char_rm.py
#         for record in tqdm.tqdm(records):
#             input_text = format_input(record)
#             input_ids = self.tokenizer.encode(
#                 text=input_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
#             if len(input_ids) > 4096:
#                 input_ids = input_ids[-4096:]
#             input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
#             with torch.no_grad():
#                 score = self.rm_model(input_ids=input_ids)[1].item() * 4 + 1
#                 record[record['metric_en']] = score

#         # codes are borrowed from: https://github.com/morecry/CharacterEval/blob/main/compute_score.py
#         details = []
#         for record in records:
#             details.append(record[record['metric_en']])

#         final_score = sum(details) / len(details)

#         return {'score': final_score, 'details': details}


@ICL_EVALUATORS.register_module()
class CharacterEvalEvaluator(BaseEvaluator):

    def __init__(self, path) -> None:
        # load baichuan character reward model: https://huggingface.co/morecry/BaichuanCharRM
        self.tokenizer = BaichuanTokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.rm_model = BaichuanCharRM.from_pretrained(
            path, torch_dtype=torch.bfloat16).cuda()
        super().__init__()

    def score(self, predictions, references):

        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        # post-process
        records = []
        for prediction, reference in zip(predictions, references):
            # Prevent continuous generation
            clean_prediction = prediction.split("\n")[0]
            reference["model_output"] = clean_prediction
            records.append(reference)

        # Run Character Reward Model
        # codes are borrowed from: https://github.com/morecry/CharacterEval/blob/main/run_char_rm.py
        for record in tqdm.tqdm(records):
            input_text = format_input(record)
            input_ids = self.tokenizer.encode(
                text=input_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            if len(input_ids) > 4096:
                input_ids = input_ids[-4096:]
            input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
            with torch.no_grad():
                score = self.rm_model(input_ids=input_ids)[1].item() * 4 + 1
                record[record['metric_en']] = score

        # codes are borrowed from: https://github.com/morecry/CharacterEval/blob/main/compute_score.py
        details = []
        for record in records:
            details.append(record[record['metric_en']])

        final_score = sum(details) / len(details)

        return {'score': final_score, 'details': details}


@ICL_EVALUATORS.register_module()
class CharacterEvalCOTEvaluator(BaseEvaluator):

    def __init__(self, path) -> None:
        # load baichuan character reward model: https://huggingface.co/morecry/BaichuanCharRM
        self.tokenizer = BaichuanTokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.rm_model = BaichuanCharRM.from_pretrained(
            path, torch_dtype=torch.bfloat16).cuda()
        super().__init__()

    def score(self, predictions, references):

        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        # post-process
        records = []
        for prediction, reference in zip(predictions, references):
            # Prevent continuous generation
            # TODO
            if "回复：" in prediction:
                prediction = prediction.split("回复：")[-1]
            clean_prediction = prediction.split("\n")[0]
            reference["model_output"] = clean_prediction
            records.append(reference)

        # Run Character Reward Model
        # codes are borrowed from: https://github.com/morecry/CharacterEval/blob/main/run_char_rm.py
        for record in tqdm.tqdm(records):
            input_text = format_input(record)
            input_ids = self.tokenizer.encode(
                text=input_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            if len(input_ids) > 4096:
                input_ids = input_ids[-4096:]
            input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
            with torch.no_grad():
                score = self.rm_model(input_ids=input_ids)[1].item() * 4 + 1
                record[record['metric_en']] = score

        # codes are borrowed from: https://github.com/morecry/CharacterEval/blob/main/compute_score.py
        details = []
        for record in records:
            details.append(record[record['metric_en']])

        final_score = sum(details) / len(details)

        return {'score': final_score, 'details': details}
