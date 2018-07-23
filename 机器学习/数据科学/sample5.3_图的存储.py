import networkx as nx
import os
import matplotlib.pyplot as plt
import snowball_sampling


# 如果文件存在，则删除
def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


dump_file_base = 'dumped_graph'
G = nx.krackhardt_kite_graph()
# GML格式写入和读取
GML_file = dump_file_base + '.gml'
remove_file(GML_file)
nx.write_gml(G, GML_file)
G2 = nx.read_gml(GML_file)
# 确保两者是相同的
# assert(G.edges() == G2.edges())
nx.draw_networkx(G2)
plt.show()

my_social_network = nx.Graph()
snowball_sampling.snowball_sampling(my_social_network, 2, 'alberto')