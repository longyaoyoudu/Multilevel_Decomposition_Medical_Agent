#!/usr/bin/env python3
# coding: utf-8
# File: test_llm_qa_system.py
# 测试脚本：对比原实现和 LLM 实现的问句分类和解析效果

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class QASystemTester:
    """
    问答系统测试器
    对比原实现和 LLM 实现的效果
    """
    
    def __init__(self, model_url=None):
        self.model_url = model_url
        
        # 测试问题集
        self.test_questions = [
            # 疾病症状查询
            {"question": "乳腺癌的症状有哪些？", "expected_type": "disease_symptom", "expected_entity": "乳腺癌"},
            
            # 症状对应疾病查询
            {"question": "流鼻涕可能是什么病？", "expected_type": "symptom_disease", "expected_entity": "流鼻涕"},
            
            # 疾病原因查询
            {"question": "为什么会失眠？", "expected_type": "disease_cause", "expected_entity": "失眠"},
            
            # 疾病并发症查询
            {"question": "失眠有哪些并发症？", "expected_type": "disease_acompany", "expected_entity": "失眠"},
            
            # 疾病忌口查询
            {"question": "失眠的人不要吃啥？", "expected_type": "disease_not_food", "expected_entity": "失眠"},
            
            # 疾病宜吃食物查询
            {"question": "耳鸣了吃点啥？", "expected_type": "disease_do_food", "expected_entity": "耳鸣"},
            
            # 食物忌口疾病查询
            {"question": "什么病最好不要吃蜂蜜？", "expected_type": "food_not_disease", "expected_entity": "蜂蜜"},
            
            # 食物宜吃疾病查询
            {"question": "鹅肉有什么好处？", "expected_type": "food_do_disease", "expected_entity": "鹅肉"},
            
            # 疾病用药查询
            {"question": "肝病要吃啥药？", "expected_type": "disease_drug", "expected_entity": "肝病"},
            
            # 药品治疗疾病查询
            {"question": "板蓝根颗粒能治啥病？", "expected_type": "drug_disease", "expected_entity": "板蓝根颗粒"},
            
            # 疾病检查项目查询
            {"question": "脑膜炎怎么检查？", "expected_type": "disease_check", "expected_entity": "脑膜炎"},
            
            # 检查项目对应疾病查询
            {"question": "全血细胞计数能查出啥？", "expected_type": "check_disease", "expected_entity": "全血细胞计数"},
            
            # 疾病预防查询
            {"question": "怎样才能预防肾虚？", "expected_type": "disease_prevent", "expected_entity": "肾虚"},
            
            # 疾病治疗周期查询
            {"question": "感冒要多久才能好？", "expected_type": "disease_lasttime", "expected_entity": "感冒"},
            
            # 疾病治疗方式查询
            {"question": "高血压要怎么治？", "expected_type": "disease_cureway", "expected_entity": "高血压"},
            
            # 疾病治愈概率查询
            {"question": "白血病能治好吗？", "expected_type": "disease_cureprob", "expected_entity": "白血病"},
            
            # 疾病易感人群查询
            {"question": "什么人容易得高血压？", "expected_type": "disease_easyget", "expected_entity": "高血压"},
            
            # 疾病描述查询
            {"question": "糖尿病是什么病？", "expected_type": "disease_desc", "expected_entity": "糖尿病"},
            
            # 复杂测试：多实体
            {"question": "感冒和发烧的症状有哪些不同？", "expected_type": "disease_symptom", "expected_entity": ["感冒", "发烧"]},
            
            # 口语化测试
            {"question": "我最近老是睡不着，这是咋回事啊？", "expected_type": "disease_cause", "expected_entity": "失眠"},
            
            # 同义词测试
            {"question": "肝癌的临床表现是什么？", "expected_type": "disease_symptom", "expected_entity": "肝癌"},
        ]
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化所有组件"""
        print("Initializing components...")
        
        # 原规则分类器
        self.rule_classifier = None
        try:
            from qa_traditional.question_classifier import QuestionClassifier
            self.rule_classifier = QuestionClassifier()
            print("✓ Rule classifier initialized")
        except ImportError as e:
            print(f"✗ Failed to initialize rule classifier: {e}")
        
        # 原解析器
        self.rule_parser = None
        try:
            from qa_traditional.question_parser import QuestionPaser
            self.rule_parser = QuestionPaser()
            print("✓ Rule parser initialized")
        except ImportError as e:
            print(f"✗ Failed to initialize rule parser: {e}")
        
        # LLM 分类器
        self.llm_classifier = None
        try:
            from qa_llm_enhanced.llm_question_classifier import LLMQuestionClassifier
            if self.model_url:
                self.llm_classifier = LLMQuestionClassifier(model_url=self.model_url)
            else:
                self.llm_classifier = LLMQuestionClassifier()
            print("✓ LLM classifier initialized")
        except Exception as e:
            print(f"✗ Failed to initialize LLM classifier: {e}")
        
        # LLM 解析器
        self.llm_parser = None
        try:
            from qa_llm_enhanced.llm_question_parser import LLMQuestionParser
            if self.model_url:
                self.llm_parser = LLMQuestionParser(model_url=self.model_url)
            else:
                self.llm_parser = LLMQuestionParser()
            print("✓ LLM parser initialized")
        except Exception as e:
            print(f"✗ Failed to initialize LLM parser: {e}")
        
        # 混合分类器
        self.hybrid_classifier = None
        try:
            from llm_question_classifier import HybridQuestionClassifier
            if self.model_url:
                self.hybrid_classifier = HybridQuestionClassifier(model_url=self.model_url)
            else:
                self.hybrid_classifier = HybridQuestionClassifier()
            print("✓ Hybrid classifier initialized")
        except Exception as e:
            print(f"✗ Failed to initialize hybrid classifier: {e}")
        
        print("=" * 50)
    
    def test_classifier(self, classifier, name):
        """测试分类器"""
        print(f"\n{'=' * 60}")
        print(f"Testing {name}")
        print(f"{'=' * 60}")
        
        if not classifier:
            print(f"✗ {name} not available, skipping test")
            return None
        
        results = []
        correct = 0
        total = 0
        
        for test_case in self.test_questions:
            question = test_case["question"]
            expected_type = test_case["expected_type"]
            expected_entity = test_case["expected_entity"]
            
            print(f"\n测试问题: {question}")
            
            try:
                result = classifier.classify(question)
                print(f"分类结果: {result}")
                
                # 检查结果
                args = result.get('args', {})
                question_types = result.get('question_types', [])
                
                # 检查实体识别
                entity_found = False
                if isinstance(expected_entity, list):
                    for entity in expected_entity:
                        if entity in args:
                            entity_found = True
                            break
                else:
                    if expected_entity in args:
                        entity_found = True
                
                # 检查问题类型
                type_found = expected_type in question_types
                
                # 计算是否正确
                is_correct = entity_found and type_found
                if is_correct:
                    correct += 1
                    print(f"✓ 分类正确")
                else:
                    print(f"✗ 分类不正确")
                    if not entity_found:
                        print(f"  - 未识别到预期实体: {expected_entity}")
                    if not type_found:
                        print(f"  - 未识别到预期类型: {expected_type}")
                
                total += 1
                
                results.append({
                    'question': question,
                    'expected_type': expected_type,
                    'expected_entity': expected_entity,
                    'actual_result': result,
                    'is_correct': is_correct
                })
                
            except Exception as e:
                print(f"✗ 分类出错: {e}")
                results.append({
                    'question': question,
                    'expected_type': expected_type,
                    'expected_entity': expected_entity,
                    'error': str(e),
                    'is_correct': False
                })
        
        # 输出统计
        print(f"\n{'=' * 60}")
        print(f"{name} 测试结果统计:")
        print(f"{'=' * 60}")
        print(f"总测试数: {total}")
        print(f"正确数: {correct}")
        if total > 0:
            print(f"准确率: {correct / total * 100:.2f}%")
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total * 100 if total > 0 else 0,
            'details': results
        }
    
    def test_parser(self, parser, classifier, name):
        """测试解析器"""
        print(f"\n{'=' * 60}")
        print(f"Testing {name}")
        print(f"{'=' * 60}")
        
        if not parser or not classifier:
            print(f"✗ {name} not available, skipping test")
            return None
        
        results = []
        
        for test_case in self.test_questions[:10]:  # 只测试前10个，节省时间
            question = test_case["question"]
            
            print(f"\n测试问题: {question}")
            
            try:
                # 先分类
                classify_result = classifier.classify(question)
                print(f"分类结果: {classify_result}")
                
                if not classify_result:
                    print("✗ 分类失败，跳过解析")
                    continue
                
                # 再解析
                parse_result = parser.parser_main(classify_result)
                print(f"解析结果: {parse_result}")
                
                results.append({
                    'question': question,
                    'classify_result': classify_result,
                    'parse_result': parse_result,
                    'success': True
                })
                
            except Exception as e:
                print(f"✗ 解析出错: {e}")
                results.append({
                    'question': question,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("开始运行所有测试...")
        print("=" * 60)
        
        # 测试分类器
        print("\n" + "#" * 60)
        print("# 分类器测试")
        print("#" * 60)
        
        rule_results = self.test_classifier(self.rule_classifier, "原规则分类器")
        llm_results = self.test_classifier(self.llm_classifier, "LLM 分类器")
        hybrid_results = self.test_classifier(self.hybrid_classifier, "混合分类器")
        
        # 测试解析器
        print("\n" + "#" * 60)
        print("# 解析器测试")
        print("#" * 60)
        
        print("\n--- 测试原解析器 + 原分类器 ---")
        self.test_parser(self.rule_parser, self.rule_classifier, "原解析器 + 原分类器")
        
        print("\n--- 测试原解析器 + LLM 分类器 ---")
        self.test_parser(self.rule_parser, self.llm_classifier, "原解析器 + LLM 分类器")
        
        # 输出对比
        print("\n" + "=" * 60)
        print("测试结果对比")
        print("=" * 60)
        
        comparison_data = []
        if rule_results:
            comparison_data.append(("原规则分类器", rule_results['accuracy']))
        if llm_results:
            comparison_data.append(("LLM 分类器", llm_results['accuracy']))
        if hybrid_results:
            comparison_data.append(("混合分类器", hybrid_results['accuracy']))
        
        if comparison_data:
            print(f"\n{'分类器名称':<20} {'准确率':<10}")
            print("-" * 30)
            for name, accuracy in comparison_data:
                print(f"{name:<20} {accuracy:.2f}%")
        
        return {
            'rule_classifier': rule_results,
            'llm_classifier': llm_results,
            'hybrid_classifier': hybrid_results
        }
    
    def interactive_test(self):
        """交互式测试"""
        print("\n" + "=" * 60)
        print("交互式测试模式")
        print("=" * 60)
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看帮助")
        
        while True:
            try:
                question = input("\n请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("退出交互式测试")
                    break
                
                if question.lower() in ['help', 'h']:
                    print("\n帮助信息:")
                    print("- 输入任何医疗问题进行测试")
                    print("- 系统将同时使用原分类器和 LLM 分类器进行分类")
                    print("- 输入 'quit' 或 'exit' 退出")
                    continue
                
                if not question:
                    continue
                
                print(f"\n问题: {question}")
                
                # 使用原分类器
                if self.rule_classifier:
                    print("\n--- 原规则分类器结果 ---")
                    try:
                        result = self.rule_classifier.classify(question)
                        print(f"分类结果: {result}")
                        
                        # 解析
                        if self.rule_parser and result:
                            parse_result = self.rule_parser.parser_main(result)
                            print(f"解析结果: {parse_result}")
                    except Exception as e:
                        print(f"错误: {e}")
                
                # 使用 LLM 分类器
                if self.llm_classifier:
                    print("\n--- LLM 分类器结果 ---")
                    try:
                        result = self.llm_classifier.classify(question)
                        print(f"分类结果: {result}")
                        
                        # 解析
                        if self.rule_parser and result:
                            parse_result = self.rule_parser.parser_main(result)
                            print(f"解析结果: {parse_result}")
                    except Exception as e:
                        print(f"错误: {e}")
                
                # 使用混合分类器
                if self.hybrid_classifier:
                    print("\n--- 混合分类器结果 ---")
                    try:
                        result = self.hybrid_classifier.classify(question)
                        print(f"分类结果: {result}")
                    except Exception as e:
                        print(f"错误: {e}")
                        
            except KeyboardInterrupt:
                print("\n\n退出交互式测试")
                break
            except Exception as e:
                print(f"错误: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试问答系统的分类和解析功能')
    parser.add_argument('--model-url', type=str, default=None, 
                        help='LLM 服务的 URL，如 http://127.0.0.1:11434/api/generate')
    parser.add_argument('--test', action='store_true', default=False,
                        help='运行自动化测试')
    parser.add_argument('--interactive', action='store_true', default=True,
                        help='运行交互式测试（默认）')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = QASystemTester(model_url=args.model_url)
    
    # 运行测试
    if args.test:
        tester.run_all_tests()
    
    if args.interactive or not args.test:
        tester.interactive_test()


if __name__ == '__main__':
    main()
