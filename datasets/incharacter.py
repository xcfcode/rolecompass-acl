import openai
from openai import OpenAI
import os
import json
import time
from datasets import Dataset, DatasetDict
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator
from .base import BaseDataset
import tiktoken


def calculate_measured_alignment(preds, labels, questionnaire_name, questionnaire, labels_pdb, dims):

    assert (preds.keys() == labels.keys())

    # 提取代理类型（agent_types），如模型或测试组
    agent_types = list(set([rpa[1] for rpa in preds.keys()]))

    # 根据问卷类型设置分数范围的最大值和最小值
    if questionnaire_name == '16Personalities':
        range_max = 100
        range_min = 0
    else:
        # 从问卷元数据中获取分数范围
        range_max = questionnaire['range'][1]
        range_min = questionnaire['range'][0]

    # 计算分数范围的中值和跨度
    range_middle = (range_max + range_min) / 2
    range_span = range_max - range_min

    # 用于存储多次测试的评估结果
    # multitime_metrics = []

    # 初始化各种统计量（MSE、MAE、准确性等）的累积变量
    sum_mse_each_dim = {a: {d: 0 for d in dims} for a in agent_types}
    sum_mae_each_dim = {a: {d: 0 for d in dims} for a in agent_types}
    correct_single_each_dim = {a: {d: 0 for d in dims} for a in agent_types}
    correct_full = {a: 0 for a in agent_types}

    count_single_each_dim = {a: {d: 0 for d in dims} for a in agent_types}
    count_full = {a: 0 for a in agent_types}

    # for each character
    for rpa in preds.keys():
        pred = preds[rpa]  # 当前预测
        label = labels[rpa]  # 当前真实标签
        a = rpa[1]  # agent_type

        full_correct = True  # 是否所有维度预测均正确
        full_X = True  # 是否所有维度均被标记为不可用（'X'）

        # 遍历当前任务的所有维度
        for dim in label.keys():
            label_score = label[dim]['score']  # 真实分数
            label_type = label[dim]['type']  # 真实类型（高/低）

            # 如果维度类型为'X'，跳过该维度
            if labels_pdb[rpa][dim]['type'] == 'X':
                continue
            else:
                full_X = False  # 存在有效维度

            pred_score = pred[dim][0]  # 预测分数
            pred_type = 'H' if pred_score > range_middle else 'L'  # 预测类型（高/低）

            count_single_each_dim[a][dim] += 1  # 统计维度总数

            # 判断预测类型是否正确
            if pred_type == label_type:
                correct_single_each_dim[a][dim] += 1
            else:
                full_correct = False  # 如果有一个维度错误，则整体错误

            # 累加均方误差（MSE）和绝对误差（MAE）
            sum_mse_each_dim[a][dim] += ((pred_score -
                                         label_score) / range_span) ** 2
            sum_mae_each_dim[a][dim] += abs((pred_score -
                                            label_score) / range_span)

        # 如果存在有效维度，统计完整正确的任务
        if not full_X:
            if full_correct:
                correct_full[a] += 1
            count_full[a] += 1

    # 聚合每个维度的统计量（包括总和）
    for count in [sum_mse_each_dim, sum_mae_each_dim, correct_single_each_dim, count_single_each_dim]:
        for a in agent_types:
            count[a]['all'] = sum(count[a].values())

    # 聚合所有代理类型的统计量
    for count in [sum_mse_each_dim, sum_mae_each_dim, correct_single_each_dim, correct_full, count_single_each_dim, count_full]:
        if isinstance(count[agent_types[0]], dict):
            count['all'] = {}
            for dim in dims + ['all']:
                count['all'][dim] = sum(count[a][dim] for a in agent_types)
        else:
            count['all'] = sum(count.values())

    # 计算每个代理类型的最终评估指标
    metrics = {}

    for a in agent_types + ['all']:
        single_acc = {}  # 每维度准确率
        single_mse = {}  # 每维度均方误差
        single_mae = {}  # 每维度绝对误差

        for dim in dims + ['all']:
            single_acc[dim] = correct_single_each_dim[a][dim] / \
                count_single_each_dim[a][dim]
            single_mse[dim] = sum_mse_each_dim[a][dim] / \
                count_single_each_dim[a][dim]
            single_mae[dim] = sum_mae_each_dim[a][dim] / \
                count_single_each_dim[a][dim]

        # 计算完整预测的准确率
        full_acc = correct_full[a] / count_full[a]

        metrics[a] = {'single_acc': single_acc, 'single_mse': single_mse,
                      'single_mae': single_mae, 'full_acc': full_acc}

    return metrics


prompts = {
    "general": {
        "background_template": '''You are an expert in Psychometrics, especially {}. I am conducting the {} test on someone. I am gauging his/her position on the {} dimension through a series of open-ended questions. For clarity, here's some background this particular dimension:
===
{}
===

My name is {}. I've invited a participant, {}, and we had many conversations in {}. I will input the conversations.

Please help me assess {}'s score within the {} dimension of {}.
''',
        "two_score_output": '''You should provide the percentage of each category, which sums to 100%, e.g., 30% A and 70% B.
Please output in the following json format:
===
{{
    "analysis": <your analysis based on the conversations>,
    "result": {{ "{}": <percentage 1>, "{}": <percentage 2> }} (The sum of percentage 1 and percentage 2 should be 100%. Output with percent sign.)
}}''',
        "one_score_output": '''You should provide the score of {} in terms of {}, which is a number between {} and {}. {} denotes 'not {} at all', {} denotes 'neutral', and {} denotes 'strongly {}'. Other numbers in this range represent different degrees of '{}'.
Please output in the following json format:
===
{{
    "analysis": <your analysis based on the conversations>,
    "result": <your score>
}}'''
    },
}


def load_questionnaire(path, questionnaire_name):
    questionnaire_path = os.path.join(
        path, f'questionnaires/{questionnaire_name}.json')
    with open(questionnaire_path, 'r', encoding='utf-8') as f:
        questionnaire = json.load(f)

    questionnaire["dimensions"] = sorted([_['cat_name']
                                          for _ in questionnaire['categories']])

    # transform into list
    questions = []
    for idx in questionnaire['questions']:
        q = questionnaire['questions'][idx]
        q.update({'id': idx})
        if q['dimension']:
            # remove None-dimension questions
            questions.append(q)

    questionnaire['questions_list'] = questions

    return questionnaire


def get_characters_and_labels(path):

    with open(os.path.join(path, "characters.json"), 'r') as f:
        character_info = json.load(f)

    with open(os.path.join(path, "characters_labels_filtered.json"), 'r') as f:
        character_labels = json.load(f)

    return character_info, character_labels


def get_character_instruction(persona_path):
    with open(persona_path, 'r') as f:
        lines = f.readlines()
        character_instruction = json.loads(lines[0])['text']

    return character_instruction


client = OpenAI(
    api_key="",
    base_url=""
)
client.base_url = ""


def truncate_text(text, max_tokens=14000, model="gpt-3.5"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)


def get_response_gpt(sys_prompt, inputs, model, retry_count=0):

    query = [{'role': 'system', 'content': sys_prompt}]
    if len(inputs) > 0:
        query.append({'role': 'user', 'content': truncate_text(inputs)})

    try:
        temperature = 0.5
        response = client.chat.completions.create(
            model=model,  # 对话模型的名称
            messages=query,
            temperature=temperature,  # 值在[0,1]之间，越大表示回复越具有不确定性
            top_p=1,
            frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容,
            timeout=60
        )
        return response.choices[0].message.content

    except openai.BadRequestError as e:
        return_value = '[TOKEN LIMIT]'
        import pdb
        pdb.set_trace()
        return return_value

    except Exception as e:
        if retry_count < 5:
            time.sleep(5)
            return get_response_gpt(sys_prompt, inputs, model, retry_count+1)

        print(f'Fail to get response after {retry_count} retry')


def string2json(llm_response):
    llm_response = llm_response.strip("`")
    if llm_response.startswith('json'):
        llm_response = llm_response[4:]

    try:
        json_response = json.loads(llm_response)
    except:
        orig_response = llm_response

        try:
            llm_response = llm_response[llm_response.find("{"):]
            llm_response = '\n'.join([line.split("//")[0]
                                     for line in llm_response.splitlines()])

            json_response = json.loads(llm_response)
        except:
            try:
                llm_response = orig_response
                # Use regex to find the JSON string in the provided text
                json_string_match = re.search(
                    r"```json\n(.+?)\n```", llm_response, re.DOTALL)

                # Extract the JSON string if found
                json_string = json_string_match.group(0)[8:-4]
                json_response = json.loads(json_string)

            except:
                try:
                    llm_response = orig_response

                    json_string_match = re.search(
                        r'\{\s*"analysis":\s*"([^"]+)",\s*"result":\s*(\{.*?\})\s*\}', llm_response, re.DOTALL)

                    # Extract the JSON string if found
                    json_string = json_string_match.group(0)

                    def fix_json_percentages(json_str):
                        # Transform 20% to
                        fixed_json = re.sub(
                            r'(?<![\'"])(\d+)%', r'"\1%"', json_str)
                        return fixed_json

                    json_string = fix_json_percentages(json_string)
                    json_response = json.loads(json_string)

                except:
                    return False

    return json_response


def get_response(sys_prompt, inputs, model, use_gpt4):
    model = model.lower().replace(' ', '')
    # if model.startswith('gpt-3.5'):
    #     model = 'gpt-3.5-turbo'
    #     print("model:", model)
    #     return get_response_gpt(sys_prompt, inputs, model)
    # elif model.startswith('gpt-4'):
    #     model = 'gpt-4-turbo'
    #     return get_response_gpt(sys_prompt, inputs, model)
    # if model.startswith('gpt-3.5'):
    #     model = 'gpt-3.5-turbo'
    #     print("model:", model)
    #     return get_response_gpt(sys_prompt, inputs, model)
    if use_gpt4:
        model = 'gpt-4-turbo'
        print("model:", model)
        return get_response_gpt(sys_prompt, inputs, model)
    else:
        model = 'gpt-3.5-turbo'
        print("model:", model)
        return get_response_gpt(sys_prompt, inputs, model)


def get_response_json(sys_prompt, inputs, model, post_processing_funcs=[string2json]):

    nth_generation = 0
    use_gpt4 = False

    while (True):
        response = get_response(sys_prompt, inputs, model, use_gpt4)
        for post_processing_func in post_processing_funcs:
            response = post_processing_func(response)
        json_response = response

        if json_response and 'result' in json_response.keys() and isinstance(json_response, dict):
            break
        else:
            use_gpt4 = True
            nth_generation += 1
            print("nth_generation:", nth_generation)
            if nth_generation > 5:
                import pdb
                pdb.set_trace()
                break

    return json_response


def avg(lst):
    return sum(lst)/len(lst)


@LOAD_DATASET.register_module()
class InCharacterDataset(BaseDataset):

    @staticmethod
    def load(path: str, questionnaire_name: str, lang: str):
        dataset = DatasetDict()
        dataset_list = []

        """load questionnaire"""
        questionnaire = load_questionnaire(path, questionnaire_name)
        questions_list = questionnaire['questions_list']
        # delete some information
        for key in ["questions_list", "questions"]:
            questionnaire.pop(key, None)

        """load characters"""
        character_info, character_labels = get_characters_and_labels(path)

        all_characters = list(set(character_info.keys()))
        for character in all_characters:
            """ Language """
            language = character[character.rfind('-')+1:]  # en or zh
            if language != lang:
                continue
            character_info[character]["language"] = language

            """ Agent Type """
            if "RoleLLM" in character_info[character]["agent"].keys():
                agent_type = "RoleLLM"
            elif "ChatHaruhi" in character_info[character]["agent"].keys():
                agent_type = "ChatHaruhi"
            # special case
            if character == 'Sheldon-en':
                agent_type = "ChatHaruhi"
            character_info[character]["agent_type"] = agent_type

            """ Character Name """
            character_name = character_info[character]["agent"][agent_type]
            character_info[character]["character_name"] = character_name
            character_info[character]["character_name_with_lang"] = character

            """ Persona """
            persona_path = os.path.join(path, "persona", agent_type.lower(), f"""{
                character_name}.jsonl""")
            character_instruction = get_character_instruction(persona_path)
            character_info[character]["character_instruction"] = character_instruction

            # character_info[character]["character_labels"] = dict()
            # character_info[character]["character_labels"]['annotation'] = character_labels["annotation"][character][questionnaire_name]
            # if questionnaire_name in ['BFI', '16Personalities']:
            """ Golden Label"""
            character_info[character]["character_labels"] = character_labels

            for question in questions_list:
                query = question[f'rewritten_{language}']
                """ For Evaluation """
                character_target_item = dict()
                character_target_item.update(character_info[character])
                character_target_item.update({"query": query})
                character_target_item.update(
                    {"dimensions": questionnaire["dimensions"]})
                character_target_item.update(
                    {"question": question})
                character_target_item.update(
                    {"questionnaire_name": questionnaire_name})

                character_target_item.update(
                    {"questionnaire": questionnaire})
                dataset_list.append({
                    # for input prompt construction
                    'character_name': character_name,
                    'character_instruction': character_instruction,
                    'experimenter': character_info[character]["experimenter"],
                    'query': query,
                    # for evaluation
                    'label': character_target_item
                })

        dataset["test"] = Dataset.from_list(dataset_list)
        return dataset


@ICL_EVALUATORS.register_module()
class IncharacterEvaluator(BaseEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):

        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        # Final evaluation results
        all_assessment_results = {}

        # Some constants TODO
        dimentions = references[0]["dimensions"]
        questionnaire_name = references[0]["questionnaire_name"]
        language = references[0]["language"]
        questionnaire = references[0]["questionnaire"]
        character_labels = references[0]["character_labels"]

        # Group results by characters
        char2items = dict()
        for prediction, reference in zip(predictions, references):
            character_name = reference['character_name']
            item = dict()
            item.update(reference)
            # for cot
            if "Assistant:" in prediction:
                prediction = prediction.split("Assistant:")[-1].strip()
            item.update({'prediction': prediction})
            if character_name in char2items:
                char2items[character_name].append(item)
            else:
                char2items[character_name] = [item]

        # processing for one charcter
        for character_name in char2items.keys():
            character_name_with_lang = char2items[character_name][0]["character_name_with_lang"]
            agent_type = char2items[character_name][0]["agent_type"]
            experimenter = char2items[character_name][0]['experimenter']
            dim_results = dict()
            # processing for one dimension
            for dim in dimentions:
                """
                Get Dim-related Responses
                """
                dim_responses = [{'id': item['question']['id'], 'question': item['query'], 'response_open': item['prediction'],
                                  'experimenter': item['experimenter']} for item in char2items[character_name] if item['question']['dimension'] == dim]

                """
                Prompt Construction
                """
                # produce dim-related conversation
                conversations = ''
                for i, r in enumerate(dim_responses):
                    # question
                    conversations += f'{i+1}.\n'
                    conversations += f"""{r['experimenter']
                                          }: {r['question']}\n"""
                    # answer
                    response = r['response_open']
                    conversations += f"{response}\n"

                language_name = {'zh': 'Chinese',
                                 'en': 'English'}[language]

                # background prompt construction
                background_prompt = prompts["general"]['background_template'].format(
                    questionnaire_name, questionnaire_name, dim, questionnaire["prompts"]["dim_desc"][dim], experimenter, character_name, language_name, character_name, dim, questionnaire_name)

                # output prompt construction
                if questionnaire_name == '16Personalities':
                    background_prompt = background_prompt.replace(
                        '16Personalities', '16Personalities (highly similar to MBTI)', 1)

                    dim_cls1, dim_cls2 = dim.split('/')

                    output_format_prompt = prompts["general"]['two_score_output'].format(
                        dim_cls1, dim_cls2)

                else:
                    neutural_score = (
                        questionnaire['range'][0] + questionnaire['range'][1]) / 2

                    if neutural_score == int(neutural_score):
                        neutural_score = int(neutural_score)

                    output_format_prompt = prompts["general"]['one_score_output'].format(
                        dim, questionnaire_name, questionnaire['range'][0], questionnaire['range'][1], questionnaire['range'][0], dim, neutural_score, questionnaire['range'][1], dim, dim)

                # syetem prompt construction
                sys_prompt = background_prompt + output_format_prompt

                # user input construction
                user_input = 'Our conversation is as follows:\n' + conversations + '\n'

                # anonymous, prevent data leakage
                for a in char2items[character_name][0]["alias"]:
                    sys_prompt = sys_prompt.replace(
                        a, '<the participant>')
                    user_input = user_input.replace(
                        a, '<the participant>')
                sys_prompt = sys_prompt.replace(
                    experimenter, '<the experimenter>')
                user_input = user_input.replace(
                    experimenter, '<the experimenter>')
                sys_prompt = sys_prompt.replace(
                    'I ', 'I (<the experimenter>) ', 1)

                user_input = user_input.replace(
                    character_name, '<the participant>')

                # for evaluating the personality of vanilla GPT
                bad_words = ['as an AI language model,', 'As an AI language model,',
                             'As an AI,', 'as an AI,', 'I am an AI language model,', 'being an AI,']

                for bad_word in bad_words:
                    user_input = user_input.replace(bad_word, '')

                sys_prompt = sys_prompt.replace("Other numbers in this range represent different degrees of 'Conscientiousness'.",
                                                "Other numbers in this range represent different degrees of 'Conscientiousness'. You must give a score, and you are not allowed to give answers like 'N/A' and 'not applicable'.", 1)

                """
                Evaluation by LLM
                """
                llm_response = get_response_json(
                    sys_prompt=sys_prompt, inputs=user_input, model='gpt-3.5')

                """
                Get Evaluation Score
                """
                if questionnaire_name == '16Personalities':
                    llm_response['result'] = {k: float(str(v).strip(
                        "%")) for k, v in llm_response['result'].items()}

                    assert (sum(llm_response['result'].values()) == 100)
                    # use the score of dim_cls1
                    llm_response['result'] = llm_response['result'][dim_cls1]
                else:
                    if llm_response['result']:
                        try:
                            llm_response['result'] = float(
                                llm_response['result'])
                        except:
                            llm_response['result'] = (
                                questionnaire['range'][0] + questionnaire['range'][1]) / 2

                    else:
                        llm_response['result'] = (
                            questionnaire['range'][0] + questionnaire['range'][1]) / 2

                dim_results[dim] = {
                    'score': llm_response['result'],
                    # 'intra_std': std_score,  # will be None
                    'details': {'id': [r['id'] for r in dim_responses], 'dim': dim, 'responses': dim_responses, 'analysis': llm_response['analysis']},
                }

            """
            Prepare Data for Final Scoring
            """
            assessment_results = {
                'dims': {},
                'analysis': {},
                'code': ''
            }

            for dim in dimentions:

                a_results_keys = dim_results[dim].keys()  # score and details

                assessment_results['dims'][dim] = {
                    'score': dim_results[dim]['score'],
                    'all_scores': [dim_results[dim]['score']]  # TODO simple
                }

                if 'details' in a_results_keys:
                    assessment_results['dims'][dim]['details'] = [
                        dim_results[dim]['details']]

            """
            Prepare to Get Final Prediction
            """
            if questionnaire_name in ['BFI', '16Personalities']:
                label_settings = ['pdb', 'annotation']
            else:
                label_settings = ['annotation']

            if questionnaire_name in ['BFI', '16Personalities']:
                thresh = (questionnaire['range'][0] +
                          questionnaire['range'][1]) / 2

                if questionnaire_name == '16Personalities':
                    thresh = 50  # special
                    pos_tags = {dim: dim[0] for dim in dimentions}
                    neg_tags = {dim: dim[-1] for dim in dimentions}
                elif questionnaire_name == 'BFI':
                    pos_tags = {'Extraversion': 'S', 'Neuroticism': 'L',
                                'Conscientiousness': 'O', 'Agreeableness': 'A', 'Openness': 'I'}
                    neg_tags = {'Extraversion': 'R', 'Neuroticism': 'C',
                                'Conscientiousness': 'U', 'Agreeableness': 'E', 'Openness': 'N'}

                """
                Get Final Prediction
                """
                code = ''
                for dim in pos_tags.keys():
                    result = assessment_results['dims'][dim]
                    if result['score'] > thresh:
                        code += pos_tags[dim]
                    else:
                        code += neg_tags[dim]
                assessment_results['code'] = code

            all_assessment_results[(
                character_name_with_lang, agent_type)] = assessment_results

        preds = {rpa: {dim: result['dims'][dim]['all_scores']
                       for dim in result['dims']} for rpa, result in all_assessment_results.items()}

        if questionnaire_name in ['BFI', '16Personalities']:
            label_settings = ['annotation', 'pdb']
            labels_pdb = {rpa: {dim: character_labels['pdb'][rpa[0]][questionnaire_name][dim]
                                for dim in result['dims']} for rpa, result in all_assessment_results.items()}
        else:
            label_settings = ['annotation']
            labels_pdb = {rpa: {dim: character_labels['annotation'][rpa[0]][questionnaire_name][dim]
                                for dim in result['dims']} for rpa, result in all_assessment_results.items()}

        for label_setting in label_settings:
            labels = {rpa: {dim: character_labels[label_setting][rpa[0]][questionnaire_name][dim] for dim in result['dims']}
                      for rpa, result in all_assessment_results.items()}

            measured_alignment = calculate_measured_alignment(
                preds, labels, questionnaire_name, questionnaire, labels_pdb=labels_pdb, dims=dimentions)

            single_acc = measured_alignment['all']['single_acc']['all']
            single_mse = measured_alignment['all']['single_mse']['all']
            single_mae = measured_alignment['all']['single_mae']['all']
            full_acc = measured_alignment['all']['full_acc']

        return {"single_acc": single_acc, "single_mse": single_mse, "single_mae": single_mae, "full_acc": full_acc}


# ['BFI', '16Personalities', 'BSRI', 'DTDD', 'ECR-R', 'EIS', 'Empathy', 'EPQ-R', 'GSE', 'ICB', 'LMS', 'LOT-R', 'WLEIS', 'CABIN']
