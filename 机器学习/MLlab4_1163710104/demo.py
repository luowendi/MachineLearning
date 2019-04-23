import matplotlib.pyplot as plt
import numpy as np

class PCA:
    def pca(self,X,n):


        X_avg = np.average(X,axis=0)

        X_core=X-X_avg


        X_cov= np.cov(X_core.T)



        eig_value,eig_vec = np.linalg.eig(np.mat(X_cov))
        eig_value_sort = np.argsort(eig_value)
        eig_max_index = eig_value_sort[-1:-(n+1):-1]


        W= eig_vec[:,eig_max_index]
        z= X_core * W

        recon_mat = z*W.T +X_avg
        return z,recon_mat

    def pca_pencent(self,X,percentage):

        X_avg = np.average(X, axis=0)
        X_core = X - X_avg


        X_cov = np.cov(X_core.T)



        eig_value, eig_vec = np.linalg.eig(np.mat(X_cov))
        n=self.percent_n(eig_value,percentage)
        eig_value_sort=np.argsort(eig_value)
        eig_max_index = eig_value_sort[-1:-(n+1):-1]



        W = eig_vec[:, eig_max_index]
        z = X_core * W

        recon_mat = z * W.T + X_avg
        return z, recon_mat

    def percent_n(self,eig_value,percentage):

        sort_eig_val=np.sort(eig_value)
        sort_eig_val = sort_eig_val[-1::-1]



        eig_sum=sum(sort_eig_val)
        tmp_sum=0.0
        num=0
        for eig in sort_eig_val:
            tmp_sum+=eig
            num+=1
            if tmp_sum>=eig_sum*percentage:
                return num

if __name__ == '__main__':
    raw_data=[[2.5,2,4],[0.5,0.7],[2.2,2.9],[1.9,2,2],[3.1,3.0],[2.3,2.7],[2,1.6],[1,1.1],[1.5,1.3]]

    data=np.array(raw_data)

    PCA = PCA()

    z,recon_data = PCA.pca(data,1)

    print(z)
    print(recon_data)


    z,recon_data=PCA.pca_pencent(data,0.95)
    print(z)
    print(recon_data)


    recon_data=np.array(recon_data.tolist())
    plt.scatter(data[:,0],data[:,-1],c='b')
    plt.scatter(recon_data[:,0],recon_data[:,1],c='r')
    plt.show()