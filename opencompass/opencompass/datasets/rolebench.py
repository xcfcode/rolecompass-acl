import json
import os
import random

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class RoleBenchBaseDataset(BaseDataset):

    @staticmethod
    def load_single(source_file, desc_list):
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = [json.loads(line) for line in f.readlines()]
        dataset = [{
            'role': item['role'],
            'desc': desc_list[item['role']],
            'question': item['question'],
            'answer': item['generated'][0]
        } for item in source_data]
        return dataset

    @staticmethod
    def load_desc(path):
        # path = get_data_path(path, local_mode=True)
        with open(path, 'r', encoding='utf-8') as f:
            desc_list = json.load(f)
        return desc_list

    @staticmethod
    def load_dataset(path, desc_list):
        train_data_list = RoleBenchBaseDataset.load_single(
            os.path.join(path, 'general/train.jsonl'), desc_list)
        train_data_list.extend(
            RoleBenchBaseDataset.load_single(
                os.path.join(path, 'role_specific/train.jsonl'), desc_list))
        test_dataset = RoleBenchBaseDataset.load_single(
            os.path.join(path, 'general/test.jsonl'), desc_list)
        test_dataset.extend(
            RoleBenchBaseDataset.load_single(
                os.path.join(path, 'role_specific/test.jsonl'), desc_list))
        # # test_dataset = test_dataset[:1000]
        # return Dataset.from_list(train_data_list).shuffle(
        #     seed=42), Dataset.from_list(test_dataset).shuffle(seed=42)

        random.seed(42)  # 设置随机种子，确保打乱结果一致
        random.shuffle(train_data_list)
        random.shuffle(test_dataset)

        # down sample to 1000 examples
        return Dataset.from_list(train_data_list), Dataset.from_list(test_dataset[:1000])


@LOAD_DATASET.register_module()
class InstructionGeneralizationEnglishDataset(RoleBenchBaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        desc_list = RoleBenchBaseDataset.load_desc(
            os.path.join(path, 'profiles-eng/desc.json'))
        path = os.path.join(path, 'rolebench-eng/instruction-generalization')
        train_dataset, test_dataset = RoleBenchBaseDataset.load_dataset(
            path, desc_list)
        return DatasetDict({'train': train_dataset, 'test': test_dataset})


@LOAD_DATASET.register_module()
class RoleGeneralizationEnglishDataset(RoleBenchBaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        desc_list = RoleBenchBaseDataset.load_desc(
            os.path.join(path, 'profiles-eng/desc.json'))
        path = os.path.join(path, 'rolebench-eng/role-generalization')
        train_dataset, test_dataset = RoleBenchBaseDataset.load_dataset(
            path, desc_list)
        return DatasetDict({'train': train_dataset, 'test': test_dataset})


@LOAD_DATASET.register_module()
class InstructionGeneralizationChineseDataset(RoleBenchBaseDataset):

    @staticmethod
    def load(path):
        desc_list = RoleBenchBaseDataset.load_desc(
            os.path.join(path, 'profiles-zh/desc.json'))
        path = os.path.join(path, 'rolebench-zh')
        train_dataset, test_dataset = RoleBenchBaseDataset.load_dataset(
            path, desc_list)
        return DatasetDict({'train': train_dataset, 'test': test_dataset})


def rolebench_zh_cot_postprocessor(text: str) -> str:
    if "回答：" in text:
        return text.split("回答：")[-1].strip()
    return text


def rolebench_en_cot_postprocessor(text: str) -> str:
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text
