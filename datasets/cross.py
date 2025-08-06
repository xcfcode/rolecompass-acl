import codecs
import json
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset


def convert_json_to_dict_golden(json_data):
    # https://github.com/Joanna0123/character_profiling/blob/315db3f51ba12c4515fb1cbad79c3de94fe6e650/code/utils.py#L161
    result_dict = {}
    for item in json_data:
        title = item.get('title')
        persona = item.get('persona', {})

        for name, details in persona.items():
            traits = details.get('traits', {})
            relationships = details.get('relationships', [])
            events = details.get('events', [])
            personality = details.get('personality', {})
            summary = details.get('summary', {})

            result_dict[title] = {
                'character': name,
                'traits': traits,
                'relationships': relationships,
                'events': events,
                'personality': personality,
                'summary': summary
            }
    return result_dict


@LOAD_DATASET.register_module()
class CrossDataset(BaseDataset):

    @staticmethod
    def load(path: str, persona_path: str):
        dataset = DatasetDict()
        for split in ['test']:
            # load question
            f = codecs.open(path, "r", "utf-8")
            multi_choice_questions = json.load(f)

            # load persona
            g = codecs.open(persona_path, "r", "utf-8")
            persona = json.load(g)
            persona = convert_json_to_dict_golden(persona)

            titles = list(persona.keys())

            dataset_list = []
            for book_title in titles:
                for character, questions in multi_choice_questions[book_title].items():
                    for question in questions:

                        option = "\n".join(
                            question["Multiple Choice Question"]["Options"])
                        question_format = f"""Scenario: {question["Multiple Choice Question"]["Scenario"]}\nQuestion: {
                            question["Multiple Choice Question"]["Question"]}\nOptions:\n{option}"""
                        id = question["Multiple Choice Question"]["id"]

                        character_persona = persona[book_title]
                        dataset_list.append({
                            'id': id,
                            'character': character,
                            'question': question_format,
                            'summary': character_persona['summary'],
                            'label': question["Multiple Choice Question"]["Correct Answer"],
                            'reason': question["Multiple Choice Question"]["Reasons"][0]
                        })
            dataset[split] = Dataset.from_list(dataset_list)

        return dataset
