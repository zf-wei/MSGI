### 聚类算法 输出NMI及 ECSim

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

from pyspark.ml.clustering import KMeans

from sklearn.metrics import normalized_mutual_info_score
import clusim.sim as sim
from clusim.clustering import Clustering

def evaluate_embedding(intr_list, intr_clus, evala): 
# 输入参数 的两个 intr 为内蕴聚类
# eval 的类型为向量 表示嵌入向量
    return_val = [] # 首先准备好返回值 
    ## 首先做 K Mean
    K = max(intr_list) + 1
    # Create a Spark DataFrame from the points
    # from pyspark.sql import SparkSession
    # from pyspark.ml.linalg import Vectors

    evala_spark = SparkSession.builder.getOrCreate()

    evala_vec = [Vectors.dense(row) for row in evala]
    
    evala_prep = SparkSession.builder.getOrCreate().\
                            createDataFrame([(vector,) for vector in evala_vec], ["embd"])

    # from pyspark.ml.clustering import KMeans

    # Create and fit the KMeans model
    euclid_kmeans = KMeans(k=K, featuresCol="embd")
    cosine_kmeans = KMeans(k=K, featuresCol="embd", distanceMeasure="cosine")
    evala_euclid_model = euclid_kmeans.fit(evala_prep)
    evala_cosine_model = cosine_kmeans.fit(evala_prep)


    # Add the cluster assignment to the DataFrame
    evala_euclid = evala_euclid_model.transform(evala_prep)
    evala_cosine = evala_cosine_model.transform(evala_prep)


    # Extract the cluster assignment and convert it to a list
    evala_euclid_membership = evala_euclid.select("prediction").rdd.flatMap(lambda x: x).collect()
    evala_cosine_membership = evala_cosine.select("prediction").rdd.flatMap(lambda x: x).collect()

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