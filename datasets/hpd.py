import codecs
import json
import collections
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset


def get_relations_and_attributes_en(relations):
    character_relation_relations_desps = []
    character_relation_attributes_desps = []

    for character in list(relations.keys()):

        character_relation_data = relations[character]

        char_relation_to_harry_desp = ""
        for relation in ["friend", "classmate", "teacher", "family", "immediate family", "lover", "opponent", "colleague", "teammate", "enemy"]:
            if character_relation_data[relation] == 1.0:
                char_relation_to_harry_desp += f"""{
                    character}'s relation with Harry is {relation}. """

        harry_affection_to_char = str(
            character_relation_data['Harry\'s affection to him'])
        harry_affection_to_char_desp = f"""Harry\'s affection to {
            character}: {harry_affection_to_char}"""

        harry_familiarity_to_char = str(
            character_relation_data['Harry\'s familiarity with him'])
        harry_familiarity_to_char_desp = f"""Harry's familiarity with {
            character}: {harry_familiarity_to_char}"""

        character_relation_relations_desps.append(char_relation_to_harry_desp)
        character_relation_attributes_desps.append(
            harry_affection_to_char_desp+"\n"+harry_familiarity_to_char_desp)

    return "\n".join(character_relation_relations_desps), "\n".join(character_relation_attributes_desps),


def get_relations_and_attributes_zh(relations):
    character_relation_relations_desps = []
    character_relation_attributes_desps = []

    for character in list(relations.keys()):

        character_relation_data = relations[character]

        char_relation_to_harry_desp = ""

        for relation in ["朋友", "同学", "老师", "家人", "直系亲属", "恋人", "对手/对头", "同事", "魁地奇队友", "敌人"]:
            if character_relation_data[relation] == 1.0:
                char_relation_to_harry_desp += f"""{
                    character}和哈利的关系是{relation}. """

        harry_affection_to_char = str(
            character_relation_data['哈利对他好感度'])
        harry_affection_to_char_desp = f"""哈利对{
            character}的好感度: {harry_affection_to_char}"""

        harry_familiarity_to_char = str(
            character_relation_data['哈利对他熟悉度'])
        harry_familiarity_to_char_desp = f"""哈利对{
            character}的熟悉度: {harry_familiarity_to_char}"""

        character_relation_relations_desps.append(char_relation_to_harry_desp)
        character_relation_attributes_desps.append(
            harry_affection_to_char_desp+"\n"+harry_familiarity_to_char_desp)

    return "\n".join(character_relation_relations_desps), "\n".join(character_relation_attributes_desps),


@LOAD_DATASET.register_module()
class HPDENDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        for split in ['test']:
            f = codecs.open(path, "r", "utf-8")
            data = json.load(f)
            dataset_list = []

            sessions = list(data.keys())
            for session in sessions:
                item = data[session]

                relations, attributes = get_relations_and_attributes_en(
                    item['relations with Harry'])

                dataset_list.append({
                    'position': item['position'],
                    'speakers': ", ".join(item['speakers']),
                    'scene': item['scene'],
                    'dialogue': "\n".join(item['dialogue']),
                    'positive_response': item['positive_response'],
                    'attributes': attributes,
                    'relations': relations
                })
            dataset[split] = Dataset.from_list(dataset_list)
        return dataset


@LOAD_DATASET.register_module()
class HPDZHDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        for split in ['test']:
            f = codecs.open(path, "r", "utf-8")
            data = json.load(f)
            dataset_list = []

            sessions = list(data.keys())
            for session in sessions:
                item = data[session]

                relations, attributes = get_relations_and_attributes_zh(
                    item['关系'])

                dataset_list.append({
                    'position': item['位置'],
                    'speakers': ", ".join(item['说话人']),
                    'scene': item['背景'],
                    'dialogue': "\n".join(item['对话历史']),
                    # different from English dataset
                    'positive_response': item['正确答案'][0],
                    'attributes': attributes,
                    'relations': relations
                })
            dataset[split] = Dataset.from_list(dataset_list)
        return dataset


def hpd_zh_cot_postprocess(text: str) -> str:
    if "哈利的回答" in text:
        return text.split("哈利的回答：")[-1].strip()
    return text


def hpd_en_cot_postprocess(text: str) -> str:
    if "Harry’s answer" in text:
        return text.split("Harry’s answer:")[-1].strip()
    return text
