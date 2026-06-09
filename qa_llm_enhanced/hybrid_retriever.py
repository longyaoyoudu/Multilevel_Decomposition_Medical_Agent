# coding: utf-8
"""
多路召回融合模块
整合 Aho-Corasick实体识别 + 图查询 + 向量检索，通过RRF融合排序
"""
import os
import sys
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_traditional.question_classifier import QuestionClassifier
from kg_builder.build_medicalgraph import MedicalGraph
from kg_builder.vector_retriever import VectorRetriever
from utils.config import Config


class HybridRetriever:
    """多路召回融合器"""

    def __init__(self):
        """初始化多路召回器"""
        # 通道1: Aho-Corasick实体识别
        self.entity_parser = QuestionClassifier()

        # 通道2: 图谱查询（延迟初始化）
        self.graph = None

        # 通道3: 向量检索（延迟初始化）
        self.vector_retriever = None

        # RRF配置
        self.rrf_k = 60
        self.fusion_top_n = Config.RERANK_TOP_K
        self.vector_top_k = Config.VECTOR_TOP_K

        print("[HybridRetriever] 多路召回器初始化完成")
        print(f"   RRF配置: k={self.rrf_k}, fusion_top_n={self.fusion_top_n}")

    def _get_graph(self):
        """延迟获取图谱连接（带重连机制）"""
        if self.graph is None:
            try:
                self.graph = MedicalGraph()
                # 测试连接
                self.graph.g.run("MATCH (n) RETURN n LIMIT 1").data()
                print("    [通道2] Neo4j连接成功")
            except Exception as e:
                print(f"    [通道2] Neo4j连接失败: {e}")
                self.graph = None
        return self.graph

    def _test_graph_connection(self, graph):
        """测试图谱连接是否有效"""
        try:
            if graph is None:
                return False
            graph.g.run("RETURN 1").data()
            return True
        except:
            return False

    def _reconnect_graph(self):
        """重新连接Neo4j"""
        try:
            if self.graph is not None:
                del self.graph
        except:
            pass
        self.graph = None

        try:
            self.graph = MedicalGraph()
            # 测试连接
            self.graph.g.run("RETURN 1").data()
            print("    [通道2] Neo4j重连成功")
            return self.graph
        except Exception as e:
            print(f"    [通道2] Neo4j重连失败: {e}")
            self.graph = None
            return None

    def _get_vector_retriever(self):
        """延迟获取向量检索器"""
        if self.vector_retriever is None:
            self.vector_retriever = VectorRetriever()
        return self.vector_retriever

    def _channel1_recall(self, query: str) -> List[Dict]:
        """
        通道1: Aho-Corasick + 规则分类召回

        Args:
            query: 用户问题

        Returns:
            list: 召回结果 [{'triple': str, 'channel': 'rule', 'rank': int}, ...]
        """
        print("  [通道1] Aho-Corasick + 规则召回...")
        results = []

        # 实体识别
        entity_dict = self.entity_parser.check_medical(query)
        if not entity_dict:
            print("    ⚠️ 未识别到实体")
            return results

        print(f"    识别到实体: {entity_dict}")

        # 问题分类
        classify_result = self.entity_parser.classify(query)
        question_types = classify_result.get('question_types', [])

        if not question_types:
            print("    ⚠️ 未分类到问题类型")
            return results

        print(f"    问题类型: {question_types}")

        # 获取图谱连接
        graph = self._get_graph()

        # 根据问题类型执行图查询
        for q_type in question_types:
            cypher_sqls = self._build_cypher_query(q_type, entity_dict)

            for rank, sql in enumerate(cypher_sqls, 1):
                try:
                    records = graph.g.run(sql).data()
                    for record in records[:10]:  # 每个查询最多取10条
                        triple = self._record_to_triple(record, q_type)
                        if triple:
                            results.append({
                                'triple': triple,
                                'channel': 'rule',
                                'q_type': q_type,
                                'rank': rank
                            })
                except Exception as e:
                    print(f"    ⚠️ Cypher查询失败: {str(e)[:50]}")

        print(f"    通道1召回: {len(results)} 条")
        return results

    def _channel2_recall(self, query: str, entity_dict: Dict, depth: int = 2) -> List[Dict]:
        """
        通道2: 图谱多跳查询召回

        Args:
            query: 用户问题
            entity_dict: 实体字典
            depth: 跳数深度

        Returns:
            list: 召回结果
        """
        print(f"  [通道2] 知识图谱{depth}跳查询...")
        results = []

        # 动态调整深度
        depth = self._determine_depth(query, depth)
        print(f"    实际查询深度: {depth}")

        # 实体类型映射到Neo4j标签
        entity_type_map = {
            'disease': 'Disease',
            'drug': 'Drug',
            'symptom': 'Symptom',
            'food': 'Food',
            'check': 'Check',
            'department': 'Department',
            'producer': 'Producer'
        }

        graph = self._get_graph()

        for entity_name, types in entity_dict.items():
            for entity_type in types:
                neo4j_type = entity_type_map.get(entity_type, 'Disease')

                sql = f"""
                MATCH p=(m:{neo4j_type})-[r*..{depth}]-(n)
                where m.name = '{entity_name}'
                RETURN p
                """

                try:
                    # 如果图谱连接失效，先重连
                    if graph is None or not self._test_graph_connection(graph):
                        graph = self._reconnect_graph()

                    if graph is None:
                        print(f"    ⚠️ Neo4j连接不可用，跳过多跳查询")
                        break

                    records = graph.g.run(sql).data()
                    for path in records[:20]:  # 最多20条路径
                        p_data = path['p']
                        triple = self._path_to_triple(p_data)
                        if triple:
                            results.append({
                                'triple': triple,
                                'channel': 'graph',
                                'entity': entity_name,
                                'depth': depth
                            })
                except Exception as e:
                    print(f"    ⚠️ 多跳查询失败: {str(e)[:50]}")

        print(f"    通道2召回: {len(results)} 条")
        return results

    def _channel3_recall(self, query: str) -> List[Dict]:
        """
        通道3: 向量语义检索召回

        Args:
            query: 用户问题

        Returns:
            list: 召回结果
        """
        print("  [通道3] 向量语义检索...")
        results = []

        try:
            vector_retriever = self._get_vector_retriever()

            if not vector_retriever.exists():
                print("    ⚠️ 向量索引未构建")
                return results

            vector_results = vector_retriever.search(query, top_k=self.vector_top_k)

            for rank, res in enumerate(vector_results, 1):
                head = res['metadata'].get('head', '')
                rel = res['metadata'].get('rel', '')
                tail = res['metadata'].get('tail', '')

                results.append({
                    'triple': f"<{head},{rel},{tail}>",
                    'channel': 'vector',
                    'rank': rank,
                    'distance': res.get('distance', 0),
                    'metadata': res['metadata']
                })

            print(f"    通道3召回: {len(results)} 条")

        except Exception as e:
            print(f"    ⚠️ 向量检索失败: {str(e)[:50]}")

        return results

    def _rrf_fusion(self, channel_results: List[Dict]) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 多路召回融合
        结合 rank 和向量距离进行融合排序

        Args:
            channel_results: 各通道召回结果

        Returns:
            list: 融合后的结果，按RRF得分降序
        """
        print(f"  [融合] RRF融合，共 {len(channel_results)} 条候选...")
        scores = {}

        for result in channel_results:
            triple = result['triple']
            channel = result['channel']
            rank = result.get('rank', 1)

            if triple not in scores:
                scores[triple] = {
                    'triple': triple,
                    'total_score': 0,
                    'channels': [],
                    'details': []
                }

            # RRF公式: 1/(k + rank)
            rrf_score = 1.0 / (self.rrf_k + rank)

            # 向量通道额外加权：距离越小得分越高
            if channel == 'vector':
                distance = result.get('distance', 1.0)
                # 将距离转换为相似度分数 (距离越小越相似)
                # ChromaDB 使用 L2 距离，范围通常是 0-N
                # 转换为: 1 / (1 + distance)，使得距离小则分数高
                sim_score = 1.0 / (1.0 + distance * 10)  # 放大距离权重
                rrf_score = rrf_score * (1 + sim_score)

            scores[triple]['total_score'] += rrf_score

            if channel not in scores[triple]['channels']:
                scores[triple]['channels'].append(channel)

            scores[triple]['details'].append({
                'channel': channel,
                'rank': rank,
                'rrf_score': rrf_score
            })

        # 按RRF得分降序排列
        fused = sorted(scores.values(), key=lambda x: x['total_score'], reverse=True)

        # 取Top-N
        top_results = fused[:self.fusion_top_n]

        # 打印融合统计
        channel_stats = {}
        for r in channel_results:
            ch = r['channel']
            channel_stats[ch] = channel_stats.get(ch, 0) + 1

        print(f"    通道召回统计: {channel_stats}")
        print(f"    融合后候选: {len(top_results)} 条")

        return top_results

    def _determine_depth(self, query: str, default_depth: int) -> int:
        """根据问题复杂度动态调整查询深度"""
        # 简单问题：1跳
        simple_keywords = ['症状', '治疗', '药品', '药物', '检查', '食谱', '忌口']
        # 复杂问题：2跳
        complex_keywords = ['并发症', '预防', '原因', '关联', '导致']
        # 探索性问题：3跳
        explore_keywords = ['为什么', '如何会', '怎么会', '什么导致']

        for kw in explore_keywords:
            if kw in query:
                return 3

        for kw in complex_keywords:
            if kw in query:
                return 2

        for kw in simple_keywords:
            if kw in query:
                return 1

        return default_depth

    def _build_cypher_query(self, q_type: str, entity_dict: Dict) -> List[str]:
        """构建Cypher查询（参考question_parser逻辑）"""
        sqls = []

        # 疾病相关属性查询
        if q_type == 'disease_cause':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.cause")

        elif q_type == 'disease_prevent':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.prevent")

        elif q_type == 'disease_lasttime':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.cure_lasttime")

        elif q_type == 'disease_cureway':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.cure_way")

        elif q_type == 'disease_cureprob':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.cured_prob")

        elif q_type == 'disease_easyget':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.easy_get")

        elif q_type == 'disease_desc':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.desc")

        # 关系查询
        elif q_type == 'disease_symptom':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{disease}' return m.name, r.name, n.name")

        elif q_type == 'symptom_disease':
            for symptom in entity_dict.get('symptom', []):
                sqls.append(f"MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where n.name = '{symptom}' return m.name, r.name, n.name")

        elif q_type == 'disease_acompany':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where m.name = '{disease}' return m.name, r.name, n.name")
                sqls.append(f"MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where n.name = '{disease}' return m.name, r.name, n.name")

        elif q_type == 'disease_not_food':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name = '{disease}' return m.name, r.name, n.name")

        elif q_type == 'disease_do_food':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease)-[r:do_eat]->(n:Food) where m.name = '{disease}' return m.name, r.name, n.name")
                sqls.append(f"MATCH (m:Disease)-[r:recommand_eat]->(n:Food) where m.name = '{disease}' return m.name, r.name, n.name")

        elif q_type == 'disease_drug':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '{disease}' return m.name, r.name, n.name")
                sqls.append(f"MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '{disease}' return m.name, r.name, n.name")

        elif q_type == 'drug_disease':
            for drug in entity_dict.get('drug', []):
                sqls.append(f"MATCH (m:Disease)-[r:common_drug]->(n:Drug) where n.name = '{drug}' return m.name, r.name, n.name")
                sqls.append(f"MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where n.name = '{drug}' return m.name, r.name, n.name")

        elif q_type == 'disease_check':
            for disease in entity_dict.get('disease', []):
                sqls.append(f"MATCH (m:Disease)-[r:need_check]->(n:Check) where m.name = '{disease}' return m.name, r.name, n.name")

        elif q_type == 'check_disease':
            for check in entity_dict.get('check', []):
                sqls.append(f"MATCH (m:Disease)-[r:need_check]->(n:Check) where n.name = '{check}' return m.name, r.name, n.name")

        return sqls

    def _record_to_triple(self, record: Dict, q_type: str) -> str:
        """将查询记录转换为三元组字符串"""
        try:
            # 处理不同类型的结果
            if 'm.name' in record and 'n.name' in record:
                # 关系查询结果
                m_name = record.get('m.name', '')
                r_name = record.get('r.name', '')
                n_name = record.get('n.name', '')
                if m_name and n_name:
                    return f"<{m_name},{r_name},{n_name}>"
            elif 'm.name' in record:
                # 节点属性查询结果
                m_name = record.get('m.name', '')
                for key, value in record.items():
                    if key.startswith('m.') and key != 'm.name' and value:
                        attr_name = key[2:]  # 去掉 'm.' 前缀
                        return f"<{m_name},{attr_name},{value}>"
        except Exception as e:
            print(f"    ⚠️ 记录转换失败: {e}")
        return ""

    def _path_to_triple(self, path_data) -> str:
        """将路径数据转换为三元组字符串"""
        try:
            nodes = path_data.nodes
            rels = path_data.relationships

            if len(nodes) >= 2 and len(rels) >= 1:
                # 取起始节点和关系
                start_node = nodes[0]
                rel = rels[0]
                end_node = nodes[-1]

                start_name = start_node.get('name', '')
                rel_name = rel.get('name', '')
                end_name = end_node.get('name', '')

                if start_name and rel_name and end_name:
                    return f"<{start_name},{rel_name},{end_name}>"
        except Exception as e:
            print(f"    ⚠️ 路径转换失败: {e}")
        return ""

    def retrieve(self, query: str, use_vector: bool = True, use_graph: bool = True) -> List[Dict]:
        """
        多路召回主流程

        Args:
            query: 用户问题
            use_vector: 是否启用向量检索通道
            use_graph: 是否启用图谱多跳通道

        Returns:
            list: 融合后的Top-N候选列表
        """
        print(f"\n{'='*60}")
        print(f"[多路召回] Query: {query}")
        print(f"{'='*60}")

        all_results = []

        # 通道1: Aho-Corasick + 规则 (必须先执行，为通道2提供entity_dict)
        print("\n[Step 1] Aho-Corasick + 规则召回")
        ch1_results = self._channel1_recall(query)
        all_results.extend(ch1_results)

        # 获取实体用于后续通道
        entity_dict = self.entity_parser.check_medical(query)

        # 通道2和通道3并行执行
        parallel_results = []

        def run_channel2():
            if use_graph:
                print("\n[Step 2] 知识图谱多跳查询")
                return self._channel2_recall(query, entity_dict, depth=2)
            return []

        def run_channel3():
            if use_vector:
                print("\n[Step 3] 向量语义检索")
                return self._channel3_recall(query)
            return []

        # 使用线程池并行执行通道2和通道3
        with ThreadPoolExecutor(max_workers=2) as executor:
            future2 = executor.submit(run_channel2)
            future3 = executor.submit(run_channel3)

            ch2_results = future2.result()
            ch3_results = future3.result()

            parallel_results.extend(ch2_results)
            parallel_results.extend(ch3_results)

        all_results.extend(parallel_results)

        # RRF融合
        print("\n[Step 4] RRF融合排序")
        fused_results = self._rrf_fusion(all_results)

        # 打印最终结果
        print(f"\n{'='*60}")
        print(f"[多路召回] 最终候选: {len(fused_results)} 条")
        print(f"{'='*60}")

        for i, r in enumerate(fused_results[:5], 1):
            print(f"  {i}. [RRF:{r['total_score']:.4f}] {r['triple']}")

        return fused_results


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试多路召回")
    print("=" * 60)

    retriever = HybridRetriever()

    # 测试查询
    test_queries = [
        "糖尿病有什么症状",
        "糖尿病吃什么药",
        "高血压应该注意什么饮食"
    ]

    for query in test_queries:
        print(f"\n{'#'*60}")
        print(f"测试Query: {query}")
        print(f"{'#'*60}")

        try:
            results = retriever.retrieve(query, use_vector=True, use_graph=True)
            print(f"\n召回结果: {len(results)} 条")
        except Exception as e:
            print(f"⚠️ 召回失败: {e}")

    print("\n✅ 多路召回测试完成")