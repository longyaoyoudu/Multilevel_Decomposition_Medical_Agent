#!/usr/bin/env python3
# coding: utf-8
# File: question_classifier.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-10-4

import os
import ahocorasick
import logging
import traceback
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("question_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuestionClassifier")


class QuestionClassifier:
    def __init__(self):
        self.logger = logger
        self.logger.info("初始化问题分类器...")

        # 尝试加载字典文件
        try:
            cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.logger.info(f"项目根目录: {cur_dir}")

            # 字典文件路径
            dict_paths = {
                'disease': 'dict/disease.txt',
                'department': 'dict/department.txt',
                'check': 'dict/check.txt',
                'drug': 'dict/drug.txt',
                'food': 'dict/food.txt',
                'producer': 'dict/producer.txt',
                'symptom': 'dict/symptom.txt',
                'deny': 'dict/deny.txt'
            }

            # 加载字典
            self.dicts = {}
            for name, path in dict_paths.items():
                full_path = os.path.join(cur_dir, path)
                self.logger.info(f"加载字典: {full_path}")

                if not os.path.exists(full_path):
                    self.logger.warning(f"字典文件不存在: {full_path}")
                    self.dicts[name] = []
                    continue

                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        self.dicts[name] = [line.strip() for line in f if line.strip()]
                    self.logger.info(f"成功加载 {len(self.dicts[name])} 个{name}词汇")
                except Exception as e:
                    self.logger.error(f"加载字典 {full_path} 失败: {str(e)}")
                    self.dicts[name] = []

            # 提取字典数据
            self.disease_wds = self.dicts['disease']
            self.department_wds = self.dicts['department']
            self.check_wds = self.dicts['check']
            self.drug_wds = self.dicts['drug']
            self.food_wds = self.dicts['food']
            self.producer_wds = self.dicts['producer']
            self.symptom_wds = self.dicts['symptom']
            self.deny_words = self.dicts['deny']

            # 合并所有领域词汇
            self.region_words = set(
                self.department_wds +
                self.disease_wds +
                self.check_wds +
                self.drug_wds +
                self.food_wds +
                self.producer_wds +
                self.symptom_wds
            )

            # 构造领域actree
            self.logger.info("构建AC自动机...")
            self.region_tree = self.build_actree(list(self.region_words))

            # 构建词汇类型字典
            self.wdtype_dict = self.build_wdtype_dict()

            # 问句疑问词
            self.symptom_qwds = ['症状', '表征', '现象', '症候', '表现']
            self.cause_qwds = ['原因', '成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥',
                               '为何', '如何才会', '怎么才会', '会导致', '会造成']
            self.acompany_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生',
                                  '一同出现', '伴随发生', '伴随', '共现']
            self.food_qwds = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜', '忌口', '补品', '保健品', '食谱',
                              '菜谱', '食用', '食物', '补品']
            self.drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']
            self.prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止', '躲避', '逃避', '避开', '免得', '逃开', '避开',
                                 '避掉', '躲开', '躲掉', '绕开',
                                 '怎样才能不', '怎么才能不', '咋样才能不', '咋才能不', '如何才能不',
                                 '怎样才不', '怎么才不', '咋样才不', '咋才不', '如何才不',
                                 '怎样才可以不', '怎么才可以不', '咋样才可以不', '咋才可以不', '如何可以不',
                                 '怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不']
            self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时',
                                  '几个小时', '多少年']
            self.cureway_qwds = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法',
                                 '咋治', '怎么办', '咋办', '咋治']
            self.cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性',
                                  '能治', '可治', '可以治', '可以医']
            self.easyget_qwds = ['易感人群', '容易感染', '易发人群', '什么人', '哪些人', '感染', '染上', '得上']
            self.check_qwds = ['检查', '检查项目', '查出', '检查', '测出', '试出']
            self.belong_qwds = ['属于什么科', '属于', '什么科', '科室']
            self.cure_qwds = ['治疗什么', '治啥', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用',
                              '有何用', '用处', '用途',
                              '有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚', '需要', '要']

            self.logger.info('问题分类器初始化完成')

        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            # 设置默认值防止崩溃
            self.region_tree = None
            self.wdtype_dict = {}
            self.region_words = set()

    '''构造词对应的类型'''

    def build_wdtype_dict(self):
        wd_dict = defaultdict(list)

        # 疾病
        for wd in self.disease_wds:
            wd_dict[wd].append('disease')

        # 科室
        for wd in self.department_wds:
            wd_dict[wd].append('department')

        # 检查项目
        for wd in self.check_wds:
            wd_dict[wd].append('check')

        # 药品
        for wd in self.drug_wds:
            wd_dict[wd].append('drug')

        # 食物
        for wd in self.food_wds:
            wd_dict[wd].append('food')

        # 症状
        for wd in self.symptom_wds:
            wd_dict[wd].append('symptom')

        # 生产商
        for wd in self.producer_wds:
            wd_dict[wd].append('producer')

        return wd_dict

    '''构造actree，加速过滤'''

    def build_actree(self, wordlist):
        try:
            actree = ahocorasick.Automaton()
            for index, word in enumerate(wordlist):
                actree.add_word(word, (index, word))
            actree.make_automaton()
            return actree
        except Exception as e:
            self.logger.error(f"构建AC自动机失败: {str(e)}")
            return None

    '''问句过滤'''

    def check_medical(self, question):
        if not question:
            return {}

        if not self.region_tree:
            self.logger.warning("AC自动机未初始化，无法识别实体")
            return {}

        region_wds = []
        try:
            # 使用AC自动机识别实体
            for item in self.region_tree.iter(question):
                wd = item[1][1]  # 提取匹配到的词
                region_wds.append(wd)
        except Exception as e:
            self.logger.error(f"实体识别失败: {str(e)}")
            return {}

        # 过滤子串
        final_wds = []
        sorted_wds = sorted(region_wds, key=len, reverse=True)  # 按长度降序排序
        for wd in sorted_wds:
            # 检查当前词是否已被包含在已选词中
            if not any(wd in other for other in final_wds):
                final_wds.append(wd)

        # 构建结果字典
        final_dict = {}
        for wd in final_wds:
            if wd in self.wdtype_dict:
                final_dict[wd] = self.wdtype_dict[wd]
            else:
                self.logger.warning(f"词汇 '{wd}' 不在类型字典中")

        return final_dict

    '''基于特征词进行分类'''

    def check_words(self, wds, sent):
        if not sent:
            return False

        for wd in wds:
            if wd in sent:
                return True
        return False

    '''分类主函数'''

    def classify(self, question):
        if not question:
            self.logger.warning("收到空问题")
            return {}

        self.logger.info(f"分类问题: {question}")

        data = {}
        medical_dict = self.check_medical(question)

        if not medical_dict:
            self.logger.info("未识别到医疗实体")
            return {}

        data['args'] = medical_dict
        self.logger.info(f"识别到的实体: {medical_dict}")

        # 收集问句当中所涉及到的实体类型
        types = []
        for type_list in medical_dict.values():
            types.extend(type_list)
        types = list(set(types))  # 去重

        self.logger.info(f"实体类型: {types}")

        question_types = []

        # 症状
        if self.check_words(self.symptom_qwds, question):
            if 'disease' in types:
                question_types.append('disease_symptom')
            if 'symptom' in types:
                question_types.append('symptom_disease')

        # 原因
        if self.check_words(self.cause_qwds, question) and 'disease' in types:
            question_types.append('disease_cause')

        # 并发症
        if self.check_words(self.acompany_qwds, question) and 'disease' in types:
            question_types.append('disease_acompany')

        # 推荐食品
        if self.check_words(self.food_qwds, question) and 'disease' in types:
            deny_status = self.check_words(self.deny_words, question)
            question_types.append('disease_not_food' if deny_status else 'disease_do_food')

        # 已知食物找疾病
        if self.check_words(self.food_qwds + self.cure_qwds, question) and 'food' in types:
            deny_status = self.check_words(self.deny_words, question)
            question_types.append('food_not_disease' if deny_status else 'food_do_disease')

        # 推荐药品
        if self.check_words(self.drug_qwds, question) and 'disease' in types:
            question_types.append('disease_drug')

        # 药品治啥病
        if self.check_words(self.cure_qwds, question) and 'drug' in types:
            question_types.append('drug_disease')

        # 疾病接受检查项目
        if self.check_words(self.check_qwds, question) and 'disease' in types:
            question_types.append('disease_check')

        # 已知检查项目查相应疾病
        if self.check_words(self.check_qwds + self.cure_qwds, question) and 'check' in types:
            question_types.append('check_disease')

        # 症状防御
        if self.check_words(self.prevent_qwds, question) and 'disease' in types:
            question_types.append('disease_prevent')

        # 疾病医疗周期
        if self.check_words(self.lasttime_qwds, question) and 'disease' in types:
            question_types.append('disease_lasttime')

        # 疾病治疗方式
        if self.check_words(self.cureway_qwds, question) and 'disease' in types:
            question_types.append('disease_cureway')

        # 疾病治愈可能性
        if self.check_words(self.cureprob_qwds, question) and 'disease' in types:
            question_types.append('disease_cureprob')

        # 疾病易感染人群
        if self.check_words(self.easyget_qwds, question) and 'disease' in types:
            question_types.append('disease_easyget')

        # 若没有查到相关的外部查询信息
        if not question_types:
            if 'disease' in types:
                question_types = ['disease_desc']
            elif 'symptom' in types:
                question_types = ['symptom_disease']
            else:
                # 尝试根据问题关键词进行通用分类
                if self.check_words(self.symptom_qwds, question):
                    question_types = ['symptom_disease']
                elif self.check_words(self.cause_qwds, question):
                    question_types = ['disease_cause']
                elif self.check_words(self.cureway_qwds, question):
                    question_types = ['disease_cureway']
                else:
                    question_types = ['others']

        self.logger.info(f"问题类型: {question_types}")

        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types

        return data


if __name__ == '__main__':
    try:
        handler = QuestionClassifier()
        while True:
            question = input('请输入问题(输入"退出"结束):').strip()
            if question.lower() in ['退出', 'exit', 'quit']:
                print('再见！')
                break

            if not question:
                print("问题不能为空！")
                continue

            data = handler.classify(question)
            print(f"分类结果: {data}")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        logger.error(traceback.format_exc())