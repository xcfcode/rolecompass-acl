
import json


def filter_json_data(data):
    # 假设JSON结构为字典，且三级键值在第二层字典中
    for key in list(data.keys()):
        if isinstance(data[key], dict):
            for sub_key in list(data[key].keys()):
                if isinstance(data[key][sub_key], dict):
                    # 过滤三级键值
                    data[key][sub_key] = {k: v for k, v in data[key][sub_key].items() if k in [
                        "16Personalities", "BFI"]}
    return data


def main():
    # 读取JSON文件
    with open('/Users/xiachongfeng/Github/rolecompass/data/incharacter/characters_labels.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 过滤数据
    filtered_data = filter_json_data(data)

    # 写回JSON文件
    with open('characters_labels_filtered.json', 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
