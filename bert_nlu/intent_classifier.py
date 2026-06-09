#!/usr/bin/env python3
# coding: utf-8
"""
基于BERT的意图分类器
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from config import BertConfig


class IntentDataset(Dataset):
    """意图分类数据集"""
    
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, 
                 intent2id: Dict[str, int], max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.intent2id = intent2id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        intent = item['intent']
        
        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 转换标签
        intent_id = self.intent2id.get(intent, self.intent2id.get('others', 0))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(intent_id, dtype=torch.long),
            'text': text,
            'intent': intent
        }


class BertIntentClassifier:
    """基于BERT的意图分类器"""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[BertConfig] = None):
        self.config = config or BertConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 加载标签映射
        self.intent2id = self.config.get_intent2id()
        self.id2intent = self.config.get_id2intent()
        self.num_labels = len(self.intent2id)
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            print(f"加载已训练模型: {model_path}")
            self.model = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels
            )
        else:
            print(f"初始化新模型: {self.config.MODEL_NAME}")
            self.model = BertForSequenceClassification.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=self.num_labels
            )
        
        self.model.to(self.device)
        
        print("意图分类器初始化完成")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据文件"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def train(self, train_data_path: str, val_data_path: str, 
              save_dir: Optional[str] = None) -> Dict:
        """
        训练模型
        """
        print("\n开始训练意图分类器...")
        
        # 加载数据
        train_data = self._load_data(train_data_path)
        val_data = self._load_data(val_data_path)
        
        print(f"训练数据: {len(train_data)} 条")
        print(f"验证数据: {len(val_data)} 条")
        
        # 创建数据集
        train_dataset = IntentDataset(
            train_data, self.tokenizer, self.intent2id, self.config.MAX_SEQ_LENGTH
        )
        val_dataset = IntentDataset(
            val_data, self.tokenizer, self.intent2id, self.config.MAX_SEQ_LENGTH
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
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        best_val_accuracy = 0.0
        training_stats = []
        
        # 训练循环
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print("-" * 40)
            
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            train_predictions = []
            train_labels = []
            
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
                logits = outputs.logits
                
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # 收集预测结果
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                train_predictions.extend(preds)
                train_labels.extend(labels.cpu().numpy())
            
            # 计算训练指标
            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = accuracy_score(train_labels, train_predictions)
            
            print(f"训练损失: {avg_train_loss:.4f}")
            print(f"训练准确率: {train_accuracy:.4f}")
            
            # 验证阶段
            self.model.eval()
            total_val_loss = 0
            val_predictions = []
            val_labels = []
            
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
                    logits = outputs.logits
                    
                    total_val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_predictions.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
            
            # 计算验证指标
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            
            print(f"验证损失: {avg_val_loss:.4f}")
            print(f"验证准确率: {val_accuracy:.4f}")
            
            # 保存训练统计
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            })
            
            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_dir:
                    self.save(save_dir)
                    print(f"最佳模型已保存到: {save_dir}")
        
        print(f"\n训练完成！最佳验证准确率: {best_val_accuracy:.4f}")
        
        return {
            'best_accuracy': best_val_accuracy,
            'training_stats': training_stats
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        预测单条文本的意图
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
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
        
        intent = self.id2intent.get(predicted_class, 'others')
        
        return intent, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        批量预测意图
        """
        results = []
        for text in texts:
            intent, confidence = self.predict(text)
            results.append((intent, confidence))
        return results
    
    def evaluate(self, test_data_path: str) -> Dict:
        """
        在测试集上评估模型
        """
        print("\n开始评估模型...")
        
        test_data = self._load_data(test_data_path)
        print(f"测试数据: {len(test_data)} 条")
        
        test_dataset = IntentDataset(
            test_data, self.tokenizer, self.intent2id, self.config.MAX_SEQ_LENGTH
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(
            true_labels, 
            predictions,
            target_names=[self.id2intent[i] for i in range(len(self.id2intent))],
            labels=list(range(len(self.id2intent))),
            output_dict=True
        )
        
        print(f"\n测试准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(
            true_labels, 
            predictions,
            target_names=[self.id2intent[i] for i in range(len(self.id2intent))],
            labels=list(range(len(self.id2intent)))
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
            'intent2id': self.intent2id,
            'id2intent': {str(k): v for k, v in self.id2intent.items()}
        }
        
        label_path = os.path.join(save_dir, 'intent_labels.json')
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(label_info, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {save_dir}")
    
    @classmethod
    def load(cls, model_dir: str) -> 'BertIntentClassifier':
        """
        加载已保存的模型
        """
        # 检查标签文件
        label_path = os.path.join(model_dir, 'intent_labels.json')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                label_info = json.load(f)
            # 这里可以覆盖默认的标签映射，但为了简单起见，我们使用配置中的
            pass
        
        # 创建分类器实例
        classifier = cls(model_path=model_dir)
        
        return classifier


def main():
    """测试函数"""
    print("=" * 60)
    print("BERT意图分类器测试")
    print("=" * 60)
    
    # 创建分类器
    classifier = BertIntentClassifier()
    
    # 测试预测
    test_texts = [
        "感冒有什么症状？",
        "头痛是什么病？",
        "高血压吃什么药？",
        "糖尿病不能吃什么？",
        "你好"
    ]
    
    print("\n测试预测功能:")
    for text in test_texts:
        intent, confidence = classifier.predict(text)
        print(f"  文本: {text}")
        print(f"  意图: {intent}, 置信度: {confidence:.4f}")
        print()


if __name__ == '__main__':
    main()
