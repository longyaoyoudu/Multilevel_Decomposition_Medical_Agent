from py2neo import Graph

def test_neo4j_connection(host, password):
    try:
        graph = Graph("bolt://" + host + ":7687", auth=("neo4j", password))
        result = graph.run("RETURN 'Connection successful' AS result")
        return True, result.data()
    except Exception as e:
        return False, str(e)

# 使用示例
success, response = test_neo4j_connection("127.0.0.1", "978626572")
print("连接结果:", "成功" if success else "失败")
print("响应:", response)