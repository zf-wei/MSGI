### 聚类算法 输出NMI及 ECSim
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize


from sklearn.metrics import normalized_mutual_info_score
import clusim.sim as sim
from clusim.clustering import Clustering

def evaluate_embd(intr_list, intr_clus, evala): 
# 输入参数 的两个 intr 为内蕴聚类
# eval 的类型为向量 表示嵌入向量
    return_val = [] # 首先准备好返回值 
    ## 首先做 K Mean
    K = max(intr_list) + 1

    points = evala
    normalized_points = normalize(evala)

    euc_kmeans = KMeans(n_clusters=K, n_init=10)
    euc_kmeans.fit(points)

    cos_kmeans = KMeans(n_clusters=K, n_init=10)
    cos_kmeans.fit(normalized_points)

    evala_euclid_membership = euc_kmeans.labels_
    evala_cosine_membership = cos_kmeans.labels_
    

    ## 然后开始与内蕴聚类进行比较
    return_val.append(normalized_mutual_info_score(evala_euclid_membership, intr_list))
    return_val.append(normalized_mutual_info_score(evala_cosine_membership, intr_list))
    
    
    evala_euclid_clustering = Clustering(elm2clu_dict={i: [evala_euclid_membership[i]] for i in range(len(evala_euclid_membership))})
    evala_cosine_clustering = Clustering(elm2clu_dict={i: [evala_cosine_membership[i]] for i in range(len(evala_cosine_membership))})
    
    evala_euclid_similarity = sim.element_sim(intr_clus, evala_euclid_clustering, alpha=0.9)
    evala_cosine_similarity = sim.element_sim(intr_clus, evala_cosine_clustering, alpha=0.9)
    return_val.append(evala_euclid_similarity)
    return_val.append(evala_cosine_similarity)
    
    return return_val