# import json


# def count_unique_names(json_file, target_lang):
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     unique_names = set()

#     for item in data:
#         if 'meta' in item and isinstance(item['meta'], dict):
#             if item['meta'].get('lang') == target_lang:
#                 unique_names.add(item['meta'].get('name'))

#     return len(unique_names)


# # 示例调用
# json_file = '/Users/xiachongfeng/Github/rolecompass/data/socialbench/social_preference.json'  # 替换为实际文件路径
# target_lang = 'zh'  # 目标语言
# unique_count = count_unique_names(json_file, target_lang)
# print(f"在语言 '{target_lang}' 下，有 {unique_count} 个独特的 name。")


import json


def count_unique_names(json_file, target_lang, target_category):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_names = set()

    for item in data:
        if 'meta' in item and isinstance(item['meta'], dict):
            if item['meta'].get('lang') == target_lang and item['meta'].get('category') == target_category:
                unique_names.add(item['meta'].get('name'))

    return len(unique_names)


# 示例调用
json_file = '/Users/xiachongfeng/Github/rolecompass/data/socialbench/conversation_memory.json'  # 替换为实际文件路径
target_lang = 'en'  # 目标语言
target_category = 'Individual-MEM-Short'  # 目标类别
unique_count = count_unique_names(json_file, target_lang, target_category)
print(f"在语言 '{target_lang}' 和类别 '{target_category}' 下，有 {unique_count} 个独特的 name。")
