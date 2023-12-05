#根据score得分，筛选数据集
import json
import os
import glob

input_path = "/home/vot/votssd/code/ChenHu/HLLY/competition_kit/data/quality_classifier/01/classfier_data.jsonl"  # 替换为你的JSON文件所在的文件夹路径

# change-----------1----------------
output_path =  "/home/vot/votssd/code/ChenHu/HLLY/competition_kit/data/quality_classifier/01/dealed_data_0.98.jsonl"

# 获取指定目录下的所有json文件
json_files = glob.glob(os.path.join(input_path, '*.json'))

# 存储转换后的数据
transformed_data = []
scores=[]
for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        # 逐行读取并加载json数据
        for line in f:
            data = json.loads(line)
            
            # # 转换数据格式(已做3.1)
            # transformed_entry = {
            #     "meta": data["meta"],
            #     "text": data["text"],
            #     "input": data["input"],
            #     "output": data["output"],
            #     "instruction": data["instruction"],
            #     "__dj__hash": data["__dj__hash"],
            #     "__dj__stats__": data["__dj__stats__"],
            #     "__dj__simhash": data["__dj__simhash"]
            # }

             # 转换数据格式(未做3.1)
            transformed_entry = {
            "meta": data["meta"],
            "text": data["text"],
            "input": data["input"],
            "output": data["output"],
            "instruction": data["instruction"]
                }
            
            # 将转换后的数据添加到列表中
            # change-----------2----------------
            if data['doc_score']>0.98:
                transformed_data.append(transformed_entry)
                scores.append(data['doc_score'])
# 将转换后的数据保存到jsonl文件
with open(output_path, 'w', encoding='utf-8') as f:
    for entry in transformed_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')
