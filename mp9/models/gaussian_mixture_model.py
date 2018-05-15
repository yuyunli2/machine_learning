"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from copy import deepcopy

class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=0.1):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        self._mu = np.random.rand(n_components, n_dims)  # np.array of size (n_components, n_dims)

        # Initialized with uniform distribution.
        self._pi = np.random.uniform(0,1,n_components).reshape(n_components, 1)  # np.array of size (n_components, 1)

        # Initialized with identity.
        self._sigma = np.zeros((n_components, n_dims, n_dims))  # np.array of size (n_components, n_dims, n_dims)

        for i in range(n_components):
            self._sigma[i,:,:] = 1000*np.identity(n_dims)

    def kmeans(self, x):
        data = x
        # print('data size', data.shape)
        length = data.shape[0]
        C = self._mu
        k = C.shape[0]
        m = C.shape[1]
        C_new = C
        C_old = np.zeros((k, m))
        count = 0

        for i in range(10):
            count += 1
            # print('count', count)
            cluster = {}
            for i in range(length):
                point = data[i, :m]
                # print('point', point)
                distance = []

                for j in range(k):
                    distance.append(sum((point - C_new[j, :]) ** 2))

                num = np.argmin(distance)
                if num not in cluster:
                    cluster[num] = [point]
                else:
                    cluster[num].append(point)
            # print('cluster', cluster)

            C_old = deepcopy(C_new)
            for ele in cluster:
                # print('ele', ele)
                C_new[ele, :] = np.mean(cluster[ele], axis=0)

            # print('C_new', C_new)

        C_final = C_new

        return C_final

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        # print('x[]', x[:self._n_components, :self._n_dims].shape)
        #
        # self._mu = x[:self._n_components, :self._n_dims]
        # # print('self mu', self._mu)
        # # self._mu = np.random.shuffle(self._mu)
        # # print('self mu', self._mu)
        #
        #
        # mean = np.mean(x,axis=0)
        # for i in range(self._n_components):
        #     self._mu[i, :] = mean
        # print(self._mu.shape)
        self._mu = self.kmeans(x)
        # self._sigma = np.zeros((self._n_components, self._n_dims, self._n_dims))  # np.array of size (n_components, n_dims, n_dims)
        #
        # for i in range(self._n_components):
        #     self._sigma[i, :, :] =  np.identity(self._n_dims)

        for i in range(self._max_iter):
            # print(i)
            z_ik = self._e_step(x)
            # z_ik = np.linalg.pinv(z_ik)
            self._m_step(x,z_ik)

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        # self._sigma = np.zeros((self._n_components, self._n_dims, self._n_dims))  # np.array of size (n_components, n_dims, n_dims)
        #
        # for i in range(self._n_components):
        #     self._sigma[i, :, :] = 1000*np.identity(self._n_dims)

        z_ik = self.get_posterior(x)

        return z_ik

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)

            _mu  # np.array of size (n_components, n_dims)
            pi np.array of size (n_components, 1)
        """
        # Update the parameters.
        N = z_ik.shape[0]
        ndims = x.shape[1]
        n_components = z_ik.shape[1]

        self._pi = (np.sum(z_ik, axis=0)/N).reshape(n_components, 1)
        self._mu = np.dot(z_ik.T,x)/(N*self._pi)
        # sigma = np.zeros((n_components, ndims, ndims))

        for i in range(n_components):
            ele1 =  np.dot((x-self._mu[i,:]).T,(z_ik[:,i].reshape(N,1)*(x-self._mu[i,:])))
            ele = ele1/(N*self._pi[i])

            # Add regularization
            self._sigma[i] = ele + self._reg_covar * np.eye(self._n_dims)
            # print('sigma', self._sigma[i])



    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N, n_components).

             ---------------------------
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        ret = []
        N = x.shape[0]
        ndims = self._n_dims
        n_components = self._n_components
        for i in range(n_components):
            mu_k = self._mu[i,:]
            # print('x', x.shape)
            # print('mu_k', mu_k.shape)
            sigma_k = self._sigma[i,:,:]
            # print('sigma_k', sigma_k.shape)
            ele = self._multivariate_gaussian(x, mu_k, sigma_k)
            ret.append(ele)

        ret = np.array(ret).T
        # print('ret shape', ret.shape)
        return ret

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        N = x.shape[0]
        n_component = self._n_components
        result = np.zeros((1,N))
        for i in range(n_component):
            mu_k = self._mu[i, :].T

            sigma_k = self._sigma[i, :, :]

            ele = self._multivariate_gaussian(x, mu_k, sigma_k)
            result += self._pi[i,:] * ele
            # result += (np.log(self._pi[i, :]) + np.log(ele))

        return result

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        N = x.shape[0]
        n_component = self._n_components
        z_ik = np.zeros((N,n_component))
        conditional = self.get_conditional(x)
        marginal = self.get_marginals(x)
        for i in range(n_component):
            # print('pi shape', self._pi.shape)
            # print('conditional', conditional.shape)
            z_ik[:,i] = self._pi[i,:] * conditional[:,i] / marginal
            # z_ik[:, i] = (np.log(self._pi[i, :]) + np.log(conditional[:, i])) / marginal

        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, 1)
        (N,1)
        """
        # print(multivariate_normal.pdf(x, mu_k, sigma_k))
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """

        self.cluster_label_map = []

        N = x.shape[0]
        n_components = self._n_components
        z_ik = self.get_posterior(x)
        print('zik', z_ik.shape)
        cluster_index = np.argmax(z_ik, axis=1)

        dict1 = {}
        cluster = np.unique(cluster_index)
        for ele in cluster:
            dict1[ele] = {}


        for i in range(N):
            if y[i] not in dict1[cluster_index[i]]:
                dict1[cluster_index[i]][y[i]] = 1
            else:
                dict1[cluster_index[i]][y[i]] += 1

        print('dict1', dict1)
        for i in range(n_components):
            if i not in dict1:
                self.cluster_label_map.append(0)
            else:
                my_dict = dict1[i]
                key_max = max(my_dict.keys(), key=(lambda k: my_dict[k]))
                self.cluster_label_map.append(key_max)


    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        z_ik = self.get_posterior(x)
        N = x.shape[0]
        y_hat = np.zeros((1,N)).reshape(N,)

        cluster_index = np.argmax(z_ik, axis=1)

        n_components = self._n_components
        print('cluster_index', cluster_index)
        print('cluster map', self.cluster_label_map)
        for i in range(N):
            y_hat[i] = self.cluster_label_map[cluster_index[i]]


        return np.array(y_hat)
