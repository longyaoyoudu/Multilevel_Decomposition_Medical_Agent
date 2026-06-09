#!/usr/bin/env python3
# coding: utf-8
"""
训练数据准备脚本
使用现有词典和问题模板生成意图识别和实体识别的训练数据
"""

import os
import json
import random
from typing import List, Dict, Tuple

from config import BertConfig


class DataPreparator:
    """数据准备类"""
    
    def __init__(self):
        self.config = BertConfig()
        self.dict_dir = self.config.get_dict_dir()
        self.data_dir = self.config.get_data_dir()
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 加载字典
        self.dictionaries = self._load_dictionaries()
        
        print(f"字典加载完成，共加载 {len(self.dictionaries)} 类实体")
        for name, entities in self.dictionaries.items():
            print(f"  - {name}: {len(entities)} 个实体")
    
    def _load_dictionaries(self) -> Dict[str, List[str]]:
        """加载所有字典文件"""
        dictionaries = {}
        
        # 字典文件映射
        dict_files = {
            'disease': 'disease.txt',
            'symptom': 'symptom.txt',
            'drug': 'drug.txt',
            'food': 'food.txt',
            'check': 'check.txt',
            'department': 'department.txt',
            'producer': 'producer.txt'
        }
        
        for entity_type, filename in dict_files.items():
            file_path = os.path.join(self.dict_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    entities = [line.strip() for line in f if line.strip()]
                dictionaries[entity_type] = entities
            else:
                print(f"警告: 字典文件不存在 {file_path}")
                dictionaries[entity_type] = []
        
        return dictionaries
    
    def _generate_intent_data(self, num_samples_per_intent: int = 100) -> List[Dict]:
        """
        生成意图识别训练数据
        """
        intent_data = []
        templates = self.config.QUESTION_TEMPLATES
        
        for intent, template_list in templates.items():
            # 确定需要的实体类型
            entity_type = None
            if 'disease' in intent:
                entity_type = 'disease'
            elif 'symptom' in intent:
                entity_type = 'symptom'
            elif 'drug' in intent:
                entity_type = 'drug'
            elif 'food' in intent:
                entity_type = 'food'
            elif 'check' in intent:
                entity_type = 'check'
            
            # 如果没有对应实体类型，跳过
            if entity_type not in self.dictionaries:
                continue
            
            entities = self.dictionaries[entity_type]
            if not entities:
                continue
            
            # 生成样本
            for _ in range(min(num_samples_per_intent, len(entities) * len(template_list))):
                # 随机选择模板和实体
                template = random.choice(template_list)
                entity = random.choice(entities)
                
                # 生成问题
                placeholder = '{' + entity_type + '}'
                question = template.replace(placeholder, entity)
                
                # 添加到数据集
                intent_data.append({
                    'text': question,
                    'intent': intent,
                    'entity': entity,
                    'entity_type': entity_type
                })
        
        # 打乱数据
        random.shuffle(intent_data)
        
        return intent_data
    
    def _generate_ner_data(self, intent_data: List[Dict]) -> List[Dict]:
        """
        生成命名实体识别训练数据
        使用BIO标注格式
        """
        ner_data = []
        
        for item in intent_data:
            text = item['text']
            entity = item['entity']
            entity_type = item['entity_type']
            
            # 生成BIO标注
            labels = self._bio_tagging(text, entity, entity_type)
            
            if labels:
                ner_data.append({
                    'text': text,
                    'tokens': list(text),  # 按字符拆分
                    'labels': labels,
                    'entity': entity,
                    'entity_type': entity_type
                })
        
        return ner_data
    
    def _bio_tagging(self, text: str, entity: str, entity_type: str) -> List[str]:
        """
        对文本进行BIO标注
        """
        labels = ['O'] * len(text)
        
        # 查找实体在文本中的位置
        start_idx = text.find(entity)
        if start_idx == -1:
            return None
        
        end_idx = start_idx + len(entity)
        
        # 标注B-开头
        labels[start_idx] = f'B-{entity_type}'
        
        # 标注I-中间
        for i in range(start_idx + 1, end_idx):
            labels[i] = f'I-{entity_type}'
        
        return labels
    
    def _split_data(self, data: List, train_ratio: float = 0.8, 
                    val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """
        划分数据集为训练集、验证集和测试集
        """
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    def prepare_all_data(self, num_samples_per_intent: int = 200):
        """
        准备所有训练数据
        """
        print("\n开始生成训练数据...")
        
        # 生成意图数据
        print("1. 生成意图识别数据...")
        intent_data = self._generate_intent_data(num_samples_per_intent)
        print(f"   共生成 {len(intent_data)} 条意图数据")
        
        # 生成NER数据
        print("2. 生成实体识别数据...")
        ner_data = self._generate_ner_data(intent_data)
        print(f"   共生成 {len(ner_data)} 条NER数据")
        
        # 划分数据集
        print("3. 划分数据集...")
        
        # 意图数据划分
        intent_train, intent_val, intent_test = self._split_data(intent_data)
        print(f"   意图数据: 训练集={len(intent_train)}, 验证集={len(intent_val)}, 测试集={len(intent_test)}")
        
        # NER数据划分
        ner_train, ner_val, ner_test = self._split_data(ner_data)
        print(f"   NER数据: 训练集={len(ner_train)}, 验证集={len(ner_val)}, 测试集={len(ner_test)}")
        
        # 保存数据
        print("4. 保存数据到文件...")
        
        # 保存意图数据
        intent_dir = os.path.join(self.data_dir, 'intent')
        os.makedirs(intent_dir, exist_ok=True)
        
        self._save_json(intent_train, os.path.join(intent_dir, 'train.json'))
        self._save_json(intent_val, os.path.join(intent_dir, 'val.json'))
        self._save_json(intent_test, os.path.join(intent_dir, 'test.json'))
        
        # 保存NER数据
        ner_dir = os.path.join(self.data_dir, 'ner')
        os.makedirs(ner_dir, exist_ok=True)
        
        self._save_json(ner_train, os.path.join(ner_dir, 'train.json'))
        self._save_json(ner_val, os.path.join(ner_dir, 'val.json'))
        self._save_json(ner_test, os.path.join(ner_dir, 'test.json'))
        
        # 保存标签映射
        print("5. 保存标签映射...")
        self._save_label_mappings()
        
        print("\n数据准备完成！")
        print(f"数据保存目录: {self.data_dir}")
        
        return {
            'intent_train': len(intent_train),
            'intent_val': len(intent_val),
            'intent_test': len(intent_test),
            'ner_train': len(ner_train),
            'ner_val': len(ner_val),
            'ner_test': len(ner_test)
        }
    
    def _save_json(self, data: List, file_path: str):
        """保存数据到JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_label_mappings(self):
        """保存标签映射文件"""
        # 意图标签
        intent2id = self.config.get_intent2id()
        id2intent = self.config.get_id2intent()
        
        intent_labels = {
            'intent2id': intent2id,
            'id2intent': {str(k): v for k, v in id2intent.items()}
        }
        
        intent_label_path = os.path.join(self.data_dir, 'intent_labels.json')
        self._save_json(intent_labels, intent_label_path)
        
        # 实体标签
        entity2id = self.config.get_entity2id()
        id2entity = self.config.get_id2entity()
        
        entity_labels = {
            'entity2id': entity2id,
            'id2entity': {str(k): v for k, v in id2entity.items()}
        }
        
        entity_label_path = os.path.join(self.data_dir, 'entity_labels.json')
        self._save_json(entity_labels, entity_label_path)
        
        print(f"   意图标签: {len(intent2id)} 类")
        print(f"   实体标签: {len(entity2id)} 类")


def main():
    """主函数"""
    print("=" * 60)
    print("BERT模型训练数据准备")
    print("=" * 60)
    
    preparator = DataPreparator()
    
    # 准备数据，每个意图生成200个样本
    stats = preparator.prepare_all_data(num_samples_per_intent=200)
    
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
