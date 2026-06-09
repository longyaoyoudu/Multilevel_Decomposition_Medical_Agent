#!/usr/bin/env python3
# coding: utf-8
"""
基于BERT的命名实体识别器（NER）
使用BIO标注格式
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from config import BertConfig


class NERDataset(Dataset):
    """命名实体识别数据集"""
    
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, 
                 entity2id: Dict[str, int], max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.entity2id = entity2id
        self.max_len = max_len
        self.id2entity = {v: k for k, v in entity2id.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']
        
        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # 对齐标签：处理BERT的subword问题
        # 这里简化处理：按字符级别的标签，假设每个字符对应一个token
        # 实际应用中需要更复杂的对齐策略
        
        # 初始化标签为O
        label_ids = [self.entity2id['O']] * self.max_len
        
        # 获取offsets mapping
        offsets = encoding['offset_mapping'][0].numpy()
        
        # 对齐标签
        for i, (start, end) in enumerate(offsets):
            if start == end:  # 特殊字符
                continue
            # 找到对应的字符位置
            # 这里简化：取第一个字符的标签
            if start < len(labels):
                label = labels[start]
                label_ids[i] = self.entity2id.get(label, self.entity2id['O'])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'text': text,
            'original_labels': labels
        }


class BertEntityRecognizer:
    """基于BERT的命名实体识别器"""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[BertConfig] = None):
        self.config = config or BertConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 加载标签映射
        self.entity2id = self.config.get_entity2id()
        self.id2entity = self.config.get_id2entity()
        self.num_labels = len(self.entity2id)
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            print(f"加载已训练模型: {model_path}")
            self.model = BertForTokenClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels
            )
        else:
            print(f"初始化新模型: {self.config.MODEL_NAME}")
            self.model = BertForTokenClassification.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=self.num_labels
            )
        
        self.model.to(self.device)
        
        print("实体识别器初始化完成")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据文件"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def train(self, train_data_path: str, val_data_path: str, 
              save_dir: Optional[str] = None) -> Dict:
        """
        训练模型
        """
        print("\n开始训练实体识别器...")
        
        # 加载数据
        train_data = self._load_data(train_data_path)
        val_data = self._load_data(val_data_path)
        
        print(f"训练数据: {len(train_data)} 条")
        print(f"验证数据: {len(val_data)} 条")
        
        # 创建数据集
        train_dataset = NERDataset(
            train_data, self.tokenizer, self.entity2id, self.config.MAX_SEQ_LENGTH
        )
        val_dataset = NERDataset(
            val_data, self.tokenizer, self.entity2id, self.config.MAX_SEQ_LENGTH
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        # 优化器和学习率调度器
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        total_steps = len(train_loader) * self.config.NUM_EPOCHS
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # 损失函数：忽略padding的标签
        # 注意：BERT的TokenClassification会自动处理attention_mask
        
        best_val_loss = float('inf')
        training_stats = []
        
        # 训练循环
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print("-" * 40)
            
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.model.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            # 计算训练损失
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"训练损失: {avg_train_loss:.4f}")
            
            # 验证阶段
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_val_loss += loss.item()
            
            # 计算验证损失
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"验证损失: {avg_val_loss:.4f}")
            
            # 保存训练统计
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_dir:
                    self.save(save_dir)
                    print(f"最佳模型已保存到: {save_dir}")
        
        print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
        
        return {
            'best_loss': best_val_loss,
            'training_stats': training_stats
        }
    
    def predict(self, text: str) -> List[Dict]:
        """
        预测单条文本的实体
        返回识别到的实体列表，每个实体包含：
        - text: 实体文本
        - type: 实体类型
        - start: 起始位置
        - end: 结束位置
        """
        self.model.eval()
        
        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config.MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offsets = encoding['offset_mapping'][0].cpu().numpy()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        
        # 解析预测结果
        entities = []
        current_entity = None
        
        for i, (pred_id, (start, end)) in enumerate(zip(predictions, offsets)):
            if start == end:  # 跳过特殊字符
                continue
            
            pred_label = self.id2entity.get(pred_id, 'O')
            
            if pred_label.startswith('B-'):
                # 新实体开始
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = pred_label[2:]  # 去掉'B-'前缀
                current_entity = {
                    'text': text[start:end],
                    'type': entity_type,
                    'start': start,
                    'end': end
                }
            elif pred_label.startswith('I-'):
                # 实体延续
                if current_entity:
                    entity_type = pred_label[2:]
                    if current_entity['type'] == entity_type:
                        current_entity['text'] = text[current_entity['start']:end]
                        current_entity['end'] = end
                    else:
                        # 类型不匹配，结束当前实体，开始新实体
                        entities.append(current_entity)
                        current_entity = {
                            'text': text[start:end],
                            'type': entity_type,
                            'start': start,
                            'end': end
                        }
            else:
                # O标签，结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        批量预测实体
        """
        results = []
        for text in texts:
            entities = self.predict(text)
            results.append(entities)
        return results
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取文本中的实体，按类型分组
        返回格式：{'disease': ['感冒', '发烧'], 'symptom': ['头痛'], ...}
        """
        entities = self.predict(text)
        
        result = {}
        for entity in entities:
            entity_type = entity['type']
            entity_text = entity['text']
            
            if entity_type not in result:
                result[entity_type] = []
            
            if entity_text not in result[entity_type]:
                result[entity_type].append(entity_text)
        
        return result
    
    def evaluate(self, test_data_path: str) -> Dict:
        """
        在测试集上评估模型
        """
        print("\n开始评估模型...")
        
        test_data = self._load_data(test_data_path)
        print(f"测试数据: {len(test_data)} 条")
        
        test_dataset = NERDataset(
            test_data, self.tokenizer, self.entity2id, self.config.MAX_SEQ_LENGTH
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        self.model.eval()
        
        # 收集所有预测和真实标签（忽略padding）
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)
                
                # 只考虑非padding的部分
                mask = attention_mask.bool()
                
                # 展平并过滤
                flat_predictions = predictions.masked_select(mask).cpu().numpy()
                flat_labels = labels.masked_select(mask).cpu().numpy()
                
                all_predictions.extend(flat_predictions)
                all_true_labels.extend(flat_labels)
        
        # 计算指标
        # 排除'O'标签的影响，只计算实体标签
        labels = [i for i, label in self.id2entity.items() if label != 'O']
        target_names = [self.id2entity[i] for i in labels]
        
        report = classification_report(
            all_true_labels,
            all_predictions,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # 计算总体准确率
        correct = sum(1 for p, t in zip(all_predictions, all_true_labels) if p == t)
        accuracy = correct / len(all_predictions) if all_predictions else 0
        
        print(f"\n测试准确率: {accuracy:.4f}")
        print("\n分类报告（实体标签）:")
        print(classification_report(
            all_true_labels,
            all_predictions,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    def save(self, save_dir: str):
        """
        保存模型
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        self.model.save_pretrained(save_dir)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # 保存标签映射
        label_info = {
            'entity2id': self.entity2id,
            'id2entity': {str(k): v for k, v in self.id2entity.items()}
        }
        
        label_path = os.path.join(save_dir, 'entity_labels.json')
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(label_info, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {save_dir}")
    
    @classmethod
    def load(cls, model_dir: str) -> 'BertEntityRecognizer':
        """
        加载已保存的模型
        """
        # 检查标签文件
        label_path = os.path.join(model_dir, 'entity_labels.json')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                label_info = json.load(f)
            # 这里可以覆盖默认的标签映射，但为了简单起见，我们使用配置中的
            pass
        
        # 创建识别器实例
        recognizer = cls(model_path=model_dir)
        
        return recognizer


def main():
    """测试函数"""
    print("=" * 60)
    print("BERT实体识别器测试")
    print("=" * 60)
    
    # 创建识别器
    recognizer = BertEntityRecognizer()
    
    # 测试预测
    test_texts = [
        "感冒有什么症状？",
        "头痛是什么病？",
        "高血压吃什么药？",
        "糖尿病不能吃什么？",
    ]
    
    print("\n测试预测功能:")
    for text in test_texts:
        entities = recognizer.predict(text)
        print(f"  文本: {text}")
        print(f"  识别到的实体:")
        for entity in entities:
            print(f"    - {entity['text']} ({entity['type']})")
        print()
        
        # 测试按类型分组
        grouped = recognizer.extract_entities(text)
        print(f"  按类型分组: {grouped}")
        print()


if __name__ == '__main__':
    main()
