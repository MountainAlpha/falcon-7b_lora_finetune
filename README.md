## 核心方法

1. 先使用自带的质量分类器（`data-juicer/tools/quality_classifier/predict_en.py`或`data-juicer/tools/quality_classifier/predict_zh.py`）对训练集进行打分，得到数据**D1_en、D1_zh**
    - 对英文数据raw_data_en.json使用`data-juicer/tools/quality_classifier/predict_en.py`
    - 对中文数据raw_data_zh.json使用`data-juicer/tools/quality_classifier/predict_zh.py`（两者主要使用的质量分类器不同：gpt3/chinese）
2. 使用脚本（`./filter_based_quality_classifier.py`）对质量分类器产生的数据（**D1_en、D1_zh**）进行过滤得到（**D2_en、D2_zh**）。
    - 中文数据保留得分高于0.98的样本。
    - 英文数据保留得分高于0.99的样本
2. 对**D2_en、D2_zh**用 `data-juicer/tools/process_data.py` 进行数据清洗得到**D3_en、D3_zh**。
    - 中文配置文件路径`./alpaca-cot-zh-refine.yaml`
    - 英文配置文件路径`./alpaca-cot-en-refine.yaml`
3. 对**D3_en、D3_zh**利用 `lm-training/get_train_dataset_7b.py` 进行采样得到**D4**。
4. 利用**D4**进行训练（`lm-training/train_scripts/deepspeed_train_7b_lora.sh`）和测试（`lm-evaluation-harness/examples/challenge-7B-stage1.sh`）  
**备注：此处的D1到D4仅为描述流程方便，并不是真实的数据名称**
##代码流程：
1. 激活环境 
`conda activate dj_comp`

2. 路径声明（非必要）  
`export PYTHONPATH=/home/vot/votssd/code/ChenHu/HLLY/competition_kit/data-juicer/:$PYTHONPATH`  
`export PYTHONPATH=/home/vot/votssd/code/ChenHu/HLLY/competition_kit/:$PYTHONPATH`

3. 质量得分分类及依据得分过滤数据  
`cd data-juicer`  
`cd tools`  
`cd quality_classifier/`  
`python predict_zh.py "/home/vot/votssd/code/ChenHu/HLLY/competition_kit/data/raw_data/raw_data_zh.jsonl" "/home/vot/votssd/code/ChenHu/HLLY/competition_kit/data/quality_classifier/01/classfier_data.jsonl"`  
`python predict_en.py "/home/vot/votssd/code/ChenHu/HLLY/competition_kit/data/raw_data/raw_data_en.jsonl" "/home/vot/votssd/code/ChenHu/HLLY/competition_kit/data/quality_classifier/02/classfier_data.jsonl"`  
`cd ..`  
`python filter_based_quality_classifier.py`  (修改中文input_path 和 output_path、修改阈值0.98）  
`python filter_based_quality_classifier.py`  (修改英文input_path 和 output_path、修改阈值0.99）  

3. 数据清洗  
`cd data-juicer`  
`python tools/process_data.py --config ./alpaca-cot-en-refine.yaml` （修改dataset_path、export_path）  
`python tools/process_data.py --config ./alpaca-cot-zh-refine.yaml` （修改dataset_path、export_path）  

4. 数据采样  
`cd lm-training`
`python get_train_dataset_7b.py`（修改EN_DATA_DIR、ZH_DATA_DIR、OUTPUT_FILES）  

5. 训练  
`cd lm-training`
`sh train_scripts/deepspeed_train_7b_lora.sh {model_path} {data_path} {output_path}`

6. 评估  
`cd lm-evaluation-harness`
`sh examples/challenge-7B-stage1.sh{mode} {model_path} {output_path}`

## 文件解释

- `alpaca-cot-en-refine.yaml`: `data-juicer/tools/process_data.py` 进行数据清洗的配置文件
- `alpaca-cot-zh-refine.yaml`: `data-juicer/tools/process_data.py` 进行数据清洗的配置文件
- `processed_data_sampling_en_zh.jsonl`: 采样后用于模型训练的数据集（即 Dprocess）
- `lora`: 训练后保存的lora模型


## 注意事项
利用质量分类器进行数据打分和利用得分过滤数据集的时候，该步骤本身不具有随机性，但数据集内的数据顺序会发生变化，该步骤可能会导致最终训练得分的少量波动。如若想完整复现最优模型，可参考processed_data_sampling_en_zh.jsonl内的键值对顺序重新排序。  
