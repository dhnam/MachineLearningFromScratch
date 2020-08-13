import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# 직접적인 GMM 및 EM 라이브러리를 제외한 함수는 설치해서 자유롭게 import 가능

cluster_color = ['red','green','pink'] # 색깔은 자유롭게 설정


seed_num = 777
np.random.seed(seed_num) # seed setting


iteration = 100 # E-step과 M-step을 반복할 횟수

# 기본 변수만 제공, 자유롭게 구현 가능 (e.g. def 안에 def 작성 가능)

class EM_BASED_GMM_PARAM:
    def __init__(self, K, data, iteration):
        self.K = K
        self.data = data
        self.iteration = iteration

    def Initialization(self, ):  # 1. initialize mean, sigma, pi(initial probability)
        # your code here
        # mean : K * (feature)
        # sigma : K * (feature) * (feature) -> (feature) * (feature). 'tied' covariance.
        # pi : K
        self.mean = self.data[np.random.randint(len(self.data), size=self.K)]
        self.sigma = np.eye(self.data.shape[1]) * np.random.rand()
        self.pi = np.ones([self.K]) / self.K

    def Expectation(self, ):  # 2. Expectation step
        gamma = [] # positeriori probabilities
        for n, data in enumerate(self.data):
            gamma.append([])
            probs = [multivariate_normal.pdf(data, self.mean[i], self.sigma) * self.pi[i]
                     for i in range(self.K)] # pi_i * N(x_n | mu_k, sigma_k)
            denominator = np.sum(probs)
            for k in range(self.K):
                numerator = probs[k]
                gamma[n].append(numerator/denominator)

        return gamma

    def Maximization(self, gamma): # 3. Maximization step
        temp = []
        for n, data in enumerate(self.data):
            for k in range(self.K):
                diff = np.array([data - self.mean[k]]).T
                temp.append(gamma[n][k] * np.dot(diff, np.transpose(diff)))
        probs_sum = 0
        for k in range(self.K):
            probs = []
            for n, _ in enumerate(self.data):
                probs.append(gamma[n][k])
            probs = np.array(probs)
            self.mean[k] = np.sum(probs[:, None] * self.data, axis=0) / np.sum(probs)
            self.pi[k] = np.sum(probs) / len(self.data)
            probs_sum += np.sum(probs)
        self.sigma = np.sum(temp, axis=0) / probs_sum
    
    def Train(self, ): # 4. Clustering, 10 point
        self.Initialization()
        for _ in range(self.iteration):
            gamma = self.Expectation()
            self.Maximization(gamma)
        return gamma


    # 이외에, 자기가 원하는 util 함수 작성하여 사용가능
 
if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    idx_shuffle = np.int32(pd.read_csv('shuffle_idx.csv',header=None)).reshape(data.shape[0]) # for data shuffle
    data = data.loc[idx_shuffle,:]
    data = np.array(data)
    model = EM_BASED_GMM_PARAM(K=3, data=data, iteration=iteration)

    res = model.Train()
    labels = []
    for n, next_data in enumerate(data):
        labels.append(np.argmax(res[n]))


    colors = [cluster_color[label] for label in labels]

        

    for i in range(3):
        rv = multivariate_normal(model.mean[i], model.sigma)
        xx = np.linspace(min(data[:, 0] - 0.2), max(data[:, 0] + 0.2), 120)
        yy = np.linspace(min(data[:, 1] - 0.2), max(data[:, 1] + 0.2), 120)
        XX, YY = np.meshgrid(xx, yy)
        plt.contour(XX, YY, rv.pdf(np.dstack([XX, YY])))

    plt.scatter(data[:, 0], data[:, 1], c=colors) # plot cluster final result
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.title("Estimation")
    plt.show()

