import numpy as np
import pandas as pd
from collections import defaultdict
import os
print(os.getcwd())

#load data
train=pd.read_csv('u1.base',sep='\t',names=['user','item,','rating','timestamp']);



test=pd.read_csv('u1.test',sep='\t',names=['user','item','rating','timestamp']);


R=defaultdict(dict)

#rating matrix


user_mean={}
#pearson similarity
#user-based prediction
#item-based similarity
#item-based prediction
#hybrid-prediction
#evalutate NMAE
for _,row in train.iterrows():
    
    
    R[row.user][row.item]=row.rating
    #user-rating
for u in R:
    
    
    user_mean[u]=np.mean(list(R[u].values()))
    
def sim_user(u,v):
    common=set(R[u]).intersection(R[V])
    
    if len(common)== 0:
       return 0
    
    
    
    num=sum((R[u][i]-user_mean[u])*(R[v][i]-user_mean[v]) for i in common)
    
    
    den1=np.sqrt(sum((R[u][i]-user_mean[u])**2 for i in common))
    
    
    den2=np.sqrt(sum((R[v][i]-user_mean[v])**2 for i in common))
    
    
    return num/(den1*den2) if den1 and den2 else 0
def pred_user(u,i,k=20):
    sims = []
    for v in R:
        if i in R[v] and v != u:
            sims.append((sim_user(u,v), v))
    
    sims.sort(reverse=True)
    sims = sims[:k]
    
    
    num, den = 0, 0
    for sim, v in sims:
        
        
        num += sim*(R[v][i]-user_mean[v])
        den += abs(sim)
        
        
    
    return user_mean[u] + num/den if den else user_mean[u]
def item_sim(i,j):
    users_i = {u for u in R if i in R[u]}
    
    
    users_j = {u for u in R if j in R[u]}
    common = users_i.intersection(users_j)
    
    if len(common)==0:
        return 0
    num=sum(R[u][i]*R[u][j] for u in common)
    
    
    den=np.sqrt(sum(R[u][i]**2 for u in common))*np.sqrt(sum(R[u][j]**2 for u in common))
    
    return num/den if den else 0

def item_pred(u,i,k=20):
    sims = []
    for j in R[u]:
        if j != i:
            sims.append((item_sim(i,j), j))
    
    
    sims.sort(reverse=True)
    
    sims = sims[:k]
    num, den = 0, 0
    
    for sim,j in sims:
        
        num += sim * R[u][j]
        den += abs(sim)
    
    return num/den if den else user_mean[u]

def pred_hybrid(u,i,alp=0.6):
    pu = pred_user(u,i)
   
   
    pi = item_pred(u,i)
    
    
    
    
    return alp*pu + (1-alp)*pi

errors = []

for _, row in test.iterrows():
    pred = pred_hybrid(row.user, row.item)
    
    
    errors.append(abs(row.rating - pred))

MAE = np.mean(errors)


NMAE = MAE / 4


print("Normalized mean absolute error is:-",NMAE)