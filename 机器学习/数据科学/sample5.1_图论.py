import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()          # 定义一个图对象
plt.subplot(221)
G.add_edge(1, 2)        # 在两个节点之间添加边（图本身不带节点，它会自动创建）
nx.draw_networkx(G)     # 图的位置（节点）自动生成
# nx.draw_networkx(G, node_color='g', node_shape='*', edge_color='b')

plt.subplot(222)
# 增加节点
G.add_nodes_from([3, 4])
# 增加另外的边
G.add_edge(3, 4)
G.add_edges_from([(2, 3), (4, 1)])
nx.draw_networkx(G)

# 获得节点和边的集合
print(G.nodes())
print(G.edges())

# 列出每个节点的邻接节点-->邻接表
print(nx.to_dict_of_lists(G))

# 列出每条边
print(nx.to_edgelist(G))    # 每个元组的第三个因素是边的属性

# 将图描述为NumPy矩阵（矩阵(i,j)位置的值为1，则表示节点i,j相连通）
print(nx.to_numpy_matrix(G))

# 对于稀疏矩阵
print(nx.to_scipy_sparse_matrix(G))

plt.subplot(223)
# 添加一条新的边
G.add_edge(1, 3)
nx.draw_networkx(G)

# 计算各节点的度
print(nx.degree(G))

# 对于大规模的图，常常利用节点度的直方图来近似其分布
# 建立一个具有10000个节点，链接概率为1%的随机网络，提取并显示该网络图节点度的直方图
# plt.hist(nx.fast_gnp_random_graph(10000, 0.01).degree())
# plt.show()
plt.subplot(224)
# degree_list = []
# for i in nx.fast_gnp_random_graph(10000, 0.01).degree():
#     degree_list.append(i[1])
degree_list = [i[1] for i in nx.fast_gnp_random_graph(10000, 0.01).degree()]
print(degree_list)
plt.hist(degree_list)
plt.show()