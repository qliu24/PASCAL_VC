import math
import numpy as np
def comptScores(inp, obj_weight, logZ, mth):
    hi,wi,ci = inp.shape
    ho,wo,co = obj_weight.shape
    # assert(ci == co)
    if hi > ho:
        diff1 = math.floor((hi-ho)/2)
        diff2 = hi-ho-diff1
        obj_weight = np.pad(obj_weight, ((diff1, diff2),(0,0),(0,0)), 'constant', constant_values=np.log(1e-3/(1-1e-3)))
        logZ = np.pad(logZ, ((diff1, diff2),(0,0),(0,0)), 'constant', constant_values=np.log(1/(1-1e-3)))
    else:
        diff1 = math.floor((ho-hi)/2)
        obj_weight = obj_weight[diff1: diff1+hi, :, :]
        logZ = logZ[diff1: diff1+hi, :, :]
        
    
    if wi > wo:
        diff1 = math.floor((wi-wo)/2)
        diff2 = wi-wo-diff1
        obj_weight = np.pad(obj_weight, ((0,0),(diff1, diff2),(0,0)), 'constant', constant_values=np.log(1e-3/(1-1e-3)))
        logZ = np.pad(logZ, ((0,0),(diff1, diff2),(0,0)), 'constant', constant_values=np.log(1/(1-1e-3)))
        
    else:
        diff1 = math.floor((wo-wi)/2)
        obj_weight = obj_weight[:, diff1: diff1+wi, :]
        logZ = logZ[:, diff1: diff1+wi, :]
        
    # assert(inp.shape[1] == wo)
    
    # term1 = inp*obj_weight
    # score = np.sum(term1) - logZ
    if mth=='all':
        score = np.dot(inp.ravel().reshape(1,-1),obj_weight.ravel().reshape(-1,1)).squeeze() - np.sum(logZ)
    else:
        score = np.dot(inp.ravel().reshape(1,-1),(obj_weight-logZ).ravel().reshape(-1,1)).squeeze()
        
    return score


def comptScoresM(inp, all_weights, all_logZs, mth='all'):
    scores_i = []
    for kk in range(len(all_weights)):
        scores_i.append(comptScores(inp, all_weights[kk].T, all_logZs[kk].T, mth))
        
    return np.max(scores_i)
        

def predictLabel(inp, all_weights1, all_logZs1, all_weights2, all_logZs2):
    score1 = comptScoresM(inp, all_weights1, all_logZs1)
    score2 = comptScoresM(inp, all_weights2, all_logZs2)
    
    return int(score2>score1), score1-score2


