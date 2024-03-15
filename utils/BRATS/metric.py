import numpy as np
from scipy.spatial.distance import directed_hausdorff


def calculate_metrics(predict,gt):
    labels = ["TC","WT","ET"]
    results = []
    for i, label in enumerate(labels):
        if np.sum(gt[i])==0:
            return []
        preds_coords = np.argwhere(predict[i])
        targets_coords = np.argwhere(gt[i])
        haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]
        tp = np.sum((predict[i]==1)&(gt[i]==1))
        tn = np.sum((predict[i]==0)&(gt[i]==0))
        fp = np.sum((predict[i]==1)&(gt[i]==0))
        fn = np.sum((predict[i]==0)&(gt[i]==1))
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        dice = 2*tp/(2*tp+fp+fn)
        results.append([
            haussdorf_dist,
            sens,
            spec,
            dice
        ])
    return results