import os
import json

# 创建输出文件夹
output_folder = "./divide_by_metric"
os.makedirs(output_folder, exist_ok=True)

# 读取 test.jsonl 数据
with open("./test_data.jsonl", "r", encoding="utf-8") as test_file:
    test_data = json.load(test_file)

# 读取 id2metric.jsonl 数据
with open("./id2metric.jsonl", "r", encoding="utf-8") as metric_file:
    id2metric = json.load(metric_file)

# 初始化分类字典
metric_to_data = {}

# 遍历 test 数据并根据 id2metric 进行分类
for entry in test_data:
    entry_id = str(entry["id"])  # 转为字符串以匹配 id2metric 的键
    if entry_id in id2metric:
        for metric in id2metric[entry_id]:
            metric_name = metric[0]  # 获取英文 metric 名称
            # entry['metric'] = metric[1]
            if metric_name not in metric_to_data:
                metric_to_data[metric_name] = []
            metric_to_data[metric_name].append(entry)

# 将分类结果写入不同的 JSON 文件
for metric_name, entries in metric_to_data.items():
    output_path = os.path.join(output_folder, f"{metric_name}.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(entries, output_file, ensure_ascii=False, indent=4)


print(f"分类完成，结果已保存到文件夹 {output_folder}")
