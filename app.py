import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

df = pd.read_csv('diamonds.csv')

#TASK 1
#This function creates each unique permutation for cut,color and clarity
#It then extracts all features and targets relevant to it, from the original dataset, for each permutation
#Finally, it checks to see if it's combined datapoints sum up to 800. If so, it adds to a dictionary in which it's values are a nested dictionary
#The nested dictionary was used to help distinguish between it's features and labels for later on
def preprocess():
    feature_mapping = {}
    subset_list = []
    df = pd.read_csv('diamonds.csv')
    cut_qual = df['cut']
    color_grade = df['color']
    clarity_grade = df['clarity']

    for i in range(len(df)):
        subset = [cut_qual[i],color_grade[i],clarity_grade[i]]
        subset_list.append(tuple(subset))
    subset_arr = (list(set(subset_list)))#used set to only keep unique permutations of cut,color & clarity

    #Pulll features and target for each tuple key
    for s in subset_arr:
        s_data = df.loc[(df['cut'] == s[0]) & (df['color'] == s[1]) & (df['clarity'] ==s[2])]
        carat = s_data['carat']
        depth = s_data['depth']
        table_val = s_data['table']
        price = s_data['price']

        #This code block only includes datasets with 800+ data points
        count_data_points = len(carat)+len(depth)+len(table_val)+len(price)
        if(count_data_points>=800):
            ftr_vectors = list(zip(carat,depth,table_val))
            ftr_vectors = np.array(ftr_vectors)
            target = price.to_numpy()
            feature_mapping[s] = {"Features":ftr_vectors, "Target":target}
    return feature_mapping

datasets = preprocess()
x = next(iter(datasets))
print(x)

#This function is just a way to quickly pull feature vectors and targets from the mapping
def get_data(key,both=False):
    traverse = datasets[key]
    features = traverse['Features']
    if both==True:
        target=traverse['Target']
        return features,target
    else:
        return features

#TASK 2
def num_coeff(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        t+=1
    return t


ftr_v,target= get_data(('Premium', 'E', 'SI2'),both=True)

def model_calc(ftrs,p,deg):
    result = np.zeros(ftrs.shape[0])
    k=0
    for n in range(deg+1):
        for i in range(n+1):
            result += p[k]*(ftrs[:,0]**i)*(ftrs[:,1]**i)*(ftrs[:,2]**(n-i))
            k+=1
    return result

#TASK 3
def linearize(poly_degree,data,p0):
    f0 = model_calc(data,p0,poly_degree)
    J = np.zeros((len(f0),len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i]+=epsilon
        fi = model_calc(data,p0,poly_degree)
        p0[i] -= epsilon
        di = (fi-f0)/epsilon
        J[:,i] = di
    return f0,J


#TASK 4
def calc_update(y,f0,J):
    l = 1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)
    dp = np.linalg.solve(N,n)
    return dp

#TASK 5
def regression(poly_degree,ftrs,target):
    epoch = 10
    p0 = None
    for deg in range(poly_degree+1):
        if(deg==0):
            p0 = np.zeros(len(ftrs))
        else:
            p0 = np.zeros(num_coeff(deg))
        for i in range(epoch):
            f0,J= linearize(deg,ftrs,p0)
            dp = calc_update(target,f0,J)
            p0+= dp
    return p0

#TASK 6
def train(sp):
    kf = model_selection.KFold(n_splits=sp,shuffle=False)
    poly_mapping = {} #A dictionary that will map each dataset to its optimal degree of polynomial
    for ds in datasets:
        print("Training dataset "+str(ds))
        poly_mapping[ds] = [0,0,0,0]
        X,y = get_data(ds,both=True)
        for train_index,test_index in kf.split(X):
                current_i_train = X[train_index]
                current_i_test = X[test_index]
                ver = y[train_index]
                scores = []
                for d in range(0,4):
                    p = regression(d,current_i_train,ver)
                    model = model_calc(current_i_train,p,d)
                    diff = np.abs(model-ver)#absoloute difference between actual price and predicted price
                    score = np.mean(diff)#mean absoloute price difference
                    scores.append(score)
                best_poly_fit = (scores.index(min(scores)))#the index of the best result is the optimal degree of polynomial
                poly_mapping[ds][best_poly_fit]+=1#increment as the highest will be the best option
        poly_mapping[ds] = poly_mapping[ds].index(max(poly_mapping[ds]))
        print(poly_mapping)
    return poly_mapping

poly_mappings = train(10)

#TASK 7
def visualize(target,estimate):
    price_range = sorted(target)
    plt.scatter(price_range,target,c="r",label="Acutal Prices")
    plt.scatter(price_range,estimate,c="b",label="estimated prices")
    plt.xlabel("Price range")
    plt.legend(loc="upper left")
    plt.show()

def estimate_and_plot(ds):
    d = poly_mappings[ds]
    d = 3
    data,target = get_data(ds,both=True)
    p = regression(d,data,target)
    estimate = model_calc(data,p,d)
    visualize(target,estimate)

def main():
    for ds in datasets:
        estimate_and_plot(ds)

main()
