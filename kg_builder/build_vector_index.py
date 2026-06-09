# coding: utf-8
"""
向量索引构建脚本
从Neo4j导出所有知识三元组，构建Chroma向量索引
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_builder.build_medicalgraph import MedicalGraph
from kg_builder.vector_retriever import VectorRetriever


def export_triples_from_neo4j():
    """
    从Neo4j导出所有三元组

    Returns:
        list: 三元组列表 [(head, rel, tail), ...]
    """
    print("=" * 60)
    print("[Step 1] 从Neo4j导出三元组")
    print("=" * 60)

    kg = MedicalGraph()
    triples = []

    # =============================================
    # 导出 Disease 节点属性三元组
    # =============================================
    print("\n📤 导出 Disease 节点属性...")

    disease_attrs = [
        ('name', '名称'),
        ('desc', '疾病简介'),
        ('cause', '疾病病因'),
        ('prevent', '预防措施'),
        ('cure_lasttime', '治疗周期'),
        ('cure_way', '治疗方式'),
        ('cured_prob', '治愈概率'),
        ('easy_get', '易感人群'),
        ('cure_department', '治疗科室')
    ]

    # 构建属性查询
    attrs_str = ', '.join([f"m.{attr}" for attr, _ in disease_attrs])
    sql = f"MATCH (m:Disease) RETURN {attrs_str}"

    try:
        records = kg.g.run(sql).data()
        count = 0
        for record in records:
            name = record.get('m.name', '')
            if not name:
                continue
            for attr, attr_cn in disease_attrs:
                attr_key = f'm.{attr}'
                if attr_key in record and record[attr_key]:
                    value = record[attr_key]
                    if value and str(value).strip():
                        triples.append((name, attr_cn, str(value)))
                        count += 1
        print(f"   Disease属性三元组: {count} 条")
    except Exception as e:
        print(f"   ⚠️ 查询失败: {e}")

    # =============================================
    # 导出关系三元组
    # =============================================
    print("\n📤 导出关系三元组...")

    rel_configs = [
        ('has_symptom', 'Disease', 'Symptom', '疾病症状'),
        ('recommand_drug', 'Disease', 'Drug', '推荐药品'),
        ('common_drug', 'Disease', 'Drug', '常用药品'),
        ('do_eat', 'Disease', 'Food', '宜吃'),
        ('no_eat', 'Disease', 'Food', '忌吃'),
        ('recommand_eat', 'Disease', 'Food', '推荐食谱'),
        ('need_check', 'Disease', 'Check', '需要检查'),
        ('acompany_with', 'Disease', 'Disease', '并发症'),
        ('belongs_to', 'Disease', 'Department', '所属科室'),
        ('drugs_of', 'Drug', 'Producer', '生产药品')
    ]

    total_rel_count = 0
    for rel_name, source_type, target_type, rel_cn in rel_configs:
        print(f"   导出 {rel_name}...", end=" ")

        sql = f"MATCH (m:{source_type})-[r:{rel_name}]->(n:{target_type}) RETURN m.name, r.name, n.name"

        try:
            records = kg.g.run(sql).data()
            rel_count = 0
            for record in records:
                source_name = record.get('m.name', '')
                target_name = record.get('n.name', '')
                rel_value = record.get('r.name', rel_name)

                if source_name and target_name:
                    triples.append((source_name, rel_value, target_name))
                    rel_count += 1
                    total_rel_count += 1

            print(f"{rel_count} 条")
        except Exception as e:
            print(f"⚠️ 失败: {e}")

    # =============================================
    # 导出 Symptom/Drug/Food/Check 节点属性三元组
    # =============================================
    print("\n📤 导出其他节点属性...")

    node_types = [
        ('Symptom', ['name', 'desc']),
        ('Drug', ['name', 'desc']),
        ('Food', ['name', 'desc']),
        ('Check', ['name', 'desc']),
        ('Department', ['name'])
    ]

    for node_type, attrs in node_types:
        print(f"   导出 {node_type}节点属性...", end=" ")

        attrs_str = ', '.join([f"m.{attr}" for attr in attrs])
        sql = f"MATCH (m:{node_type}) RETURN {attrs_str}"

        try:
            records = kg.g.run(sql).data()
            node_count = 0
            for record in records:
                name = record.get('m.name', '')
                if not name:
                    continue
                for attr in attrs:
                    attr_key = f'm.{attr}'
                    if attr_key in record and record[attr_key]:
                        value = record[attr_key]
                        if value and str(value).strip() and attr != 'name':
                            triples.append((name, attr, str(value)))
                            node_count += 1
            print(f"{node_count} 条")
        except Exception as e:
            print(f"⚠️ 失败: {e}")

    # =============================================
    # 去重
    # =============================================
    print("\n📊 数据去重...")
    original_count = len(triples)
    triples = list(set(triples))
    print(f"   去重前: {original_count} 条")
    print(f"   去重后: {len(triples)} 条")

    print(f"\n✅ 总计导出 {len(triples)} 条三元组")
    return triples


def build_index(rebuild=False):
    """
    构建向量索引

    Args:
        rebuild: 是否重建索引
    """
    print("=" * 60)
    print("向量索引构建")
    print("=" * 60)

    # 检查是否需要重建
    retriever = VectorRetriever()

    if retriever.exists() and not rebuild:
        print(f"\n⚠️ 向量索引已存在 ({retriever.count()} 条)")
        print("   如需重建，请设置 rebuild=True")
        return

    if rebuild:
        print("\n🗑️  删除旧索引...")
        retriever.delete_index()

    # Step 1: 导出三元组
    triples = export_triples_from_neo4j()

    if not triples:
        print("\n❌ 未导出任何三元组，请检查Neo4j连接")
        return

    # Step 2: 构建向量索引
    print("\n" + "=" * 60)
    print("[Step 2] 构建Chroma向量索引")
    print("=" * 60)

    # 预先重置Embedding模型，确保加载最新配置
    from utils.embedding_model import embedding_model
    embedding_model.reset()

    retriever.build_index(triples, batch_size=100)

    print("\n" + "=" * 60)
    print("✅ 向量索引构建完成!")
    print("=" * 60)
    print(f"   存储路径: {retriever.persist_dir}")
    print(f"   索引记录: {retriever.count()} 条")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='构建医疗知识向量索引')
    parser.add_argument('--rebuild', action='store_true', help='强制重建索引')
    parser.add_argument('--check', action='store_true', help='仅检查索引状态')

    args = parser.parse_args()

    if args.check:
        retriever = VectorRetriever()
        print(f"索引状态: {'已构建' if retriever.exists() else '未构建'}")
        print(f"索引记录数: {retriever.count()}")
    else:
        build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()