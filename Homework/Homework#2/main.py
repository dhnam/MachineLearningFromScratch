import random                       # import random for random samping first centroids
import numpy as np                  # import numpy for 2d matrix processing
import pandas as pd                 # pandas for overall processing
import matplotlib.pyplot as plt     # for showing plots
import warnings                     # deprecation issue handler

try:
    from sklearn.cluster import KMeans  # check installation of sklearn
except:
    print("Not installed scikit-learn.")  # print general error  
    pass

LIMITION = 900  # Limit the maximum iteration(for get result much faster)
seed_num = 777  # set random seed
np.random.seed(seed_num) # seed setting
iteration = 300 # if value unchage untill 300 times

class kmeans_:
    def __init__(self, k, data, iteration): # initalize
        self.k = k # number of cluster
        self.data = data    # data
        self.iteration = iteration # set iteration [300]
        self.centroids = self.init_centroid()


    def init_centroid(self, ):  # Set initali centorids
        data = self.data.to_numpy()
        indices = np.random.randint(int(np.size(data, 0)), size=int(self.k))
        sampled_cen = data[indices, :]
        return sampled_cen

    def train(self, ):  # Train for get result and Processing overall kmeans workings
        prev_centroids = [[] for i in range(self.k)]
        iters = 0
        warnings.simplefilter(action="ignore", category=FutureWarning)
        result = None

        for t in range(self.iteration):
            clusters = self.assignment()
            self.update_centroids(clusters)
            result = clusters
            if np.array_equal(self.centroids, prev_centroids):
                break
            prev_centroids = self.centroids.copy()

        return result # return result

    def assignment(self): # 'assignment' part in psudocode. Assign cluster to each data.
        data = self.data.to_numpy()
        return self.get_cluster(data)

    def update_centroids(self, clusters): # 'update' part in psudocode. Update centroid coordinate.
        for i, next_cluster in enumerate(clusters.values()):
            self.centroids[i] = np.mean(next_cluster, axis=0).tolist()

    def get_cluster(self, data): # Clusters data based on Uclidian distance
        clusters = {}
        for ins in data:
            ud_list = []
            for i, _ in enumerate(self.centroids):
                ud_list.append((i, np.linalg.norm(ins - self.centroids[i]))) #||x_n - c_i||^2
            next_idx = min(ud_list, key=lambda t:t[1])[0]
            
            try:
                clusters[next_idx].append(ins)
            except KeyError:
                clusters[next_idx] = [ins]
        
        for result in clusters:
            if result is None: #give random point to void centroid.
                rand_idx = np.random.randint(0, len(data), size=1)
                result.append(data[rand_idx].flatten().tolist())

        return clusters     # return whole clustsers k sub-clusters

if __name__ == '__main__': # Start from main
    colorlist = ['r','c','k','g','m','b','y'] # Set color list (set this pallet because white and yellow is hard to congize)
    data = pd.read_csv('data.csv') # load data
    model1 = kmeans_(k=5, data=data, iteration=iteration) # implemented model init setting
    clusters = model1.train()
    result = []
    for i in range(int(model1.k)):
        result = np.array(clusters[i])
        result_x = result[:, 0]
        result_y = result[:, 1]
        plt.scatter(result_x,result_y,c=str((colorlist[i]))) #plt scatter for each clusters
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("implementaion") # set title
    plt.show() # show plot

    model2 = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data) # sklearn model init setting
    predict = pd.DataFrame(model2.predict(data)) # update predict label
    predict.columns = ["predict"] # Set col name
    data = pd.concat([data,predict],axis=1) # concat data
    predict.columns=['predict'] # Set col name
    plt.scatter(data['Sepal width'],data['Sepal length'],c=data['predict'],alpha=0.5) # scatter plot
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("from scikit-learn library") # set title
    plt.show() # show plot
