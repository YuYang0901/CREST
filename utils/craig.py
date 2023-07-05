# =============================================================================
# Code in this file is copied from https://github.com/baharanm/craig
# =============================================================================


import gc
import heapq
import math
import subprocess
import time

import numpy as np
import sklearn
import sklearn.metrics

SEED = 100
EPS = 1e-8

class FacilityLocation:
    def __init__(self, D, V, alpha=1.0, gamma=0.0):
        """
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        """
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1.0 / self.inc(V, [])
        self.gamma = gamma / len(self.D)  # encouraging diversity

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return (
                self.norm
                * math.log(
                    1
                    + self.f_norm
                    * (
                        np.maximum(self.curMax, self.D[:, ndx]).sum()
                        - self.gamma * self.D[sset + [ndx]][:, sset + [ndx]].sum()
                    )
                )
                - self.curVal
            )
        else:
            return (
                self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum())
                - self.curVal
            )

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(
            1
            + self.f_norm
            * (
                self.curMax.sum()
                - self.gamma * self.D[sset + [ndx]][:, sset + [ndx]].sum()
            )
        )
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


def lazy_greedy(F, ndx, B):
    """
    Args
    - F: FacilityLocation
    - ndx: indices of all points
    - B: int, number of points to select
    """
    TOL = 1e-6
    eps = 1e-15
    curVal = 0
    sset = []
    order = []
    vals = []
    for v in ndx:
        marginal = F.inc(sset, v) + eps
        heapq.heappush(order, (1.0 / marginal, v, marginal))

    not_selected = []
    while order and len(sset) < B:
        el = heapq.heappop(order)
        if not sset:
            improv = el[2]
        else:
            improv = F.inc(sset, el[1]) + eps

        # check for uniques elements
        if improv > 0 + eps:
            if not order:
                curVal = F.add(sset, el[1])
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = heapq.heappop(order)
                if improv >= top[2]:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    heapq.heappush(order, (1.0 / improv, el[1], improv))
                heapq.heappush(order, top)
        else:  # save the unselected items in order in a list
            not_selected.append(el[1])
    # if the number of item selected is less than desired, add items from the unselected item list
    if len(sset) < B:
        num_add = B - len(sset)
        sset.extend(not_selected[:num_add])
    return sset, vals



def similarity(X, metric):
    """Computes the similarity between each pair of examples in X.

    Args
    - X: np.array, shape [N, d]
    - metric: str, one of ['cosine', 'euclidean']

    Returns
    - S: np.array, shape [N, N]
    """
    # print(f'Computing similarity for {metric}...', flush=True)
    start = time.time()

    # 1. condensed distance matrix
    # 2. square distance matrix
    #    - this must happen BEFORE converting to similarity
    #      because squareform() always puts 0 on the diagonal
    # 3. convert from distance to similarity
    dists = sklearn.metrics.pairwise_distances(X, metric=metric, n_jobs=1)

    elapsed = time.time() - start

    L0 = 0

    if metric == "cosine":
        S = 1 - dists
    elif metric == "euclidean" or metric == "l1":
        m = np.max(dists)
        S = m - dists
        L0 = m * len(dists)
    else:
        raise ValueError(f"unknown metric: {metric}")

    return S, elapsed


def get_facility_location_submodular_order(
    S, B, c, smtk=0, no=0, stoch_greedy=0, weights=None, subset_size=128
):
    """
    Args
    - S: np.array, shape [N, N], similarity matrix
    - B: int, number of points to select

    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point
    """

    N = S.shape[0]
    no = smtk if no == 0 else no

    if smtk > 0:
        print(
            f"Calculating ordering with SMTK... part size: {len(S)}, B: {B}", flush=True
        )
        np.save(
            f"tmp/{no}/{smtk}-{c}", S
        )  # todo:try thread for greedi
        if stoch_greedy > 0:
            p = subprocess.check_output(
                f"/home/yuyang/smtk-master{no}/build/smraiz -sumsize {B} \
                 -stochastic-greedy -sg-epsilon {stoch_greedy} -flnpy tmp/{no}/{smtk}-{c}."
                f"npy -pnpv -porder -ptime".split()
            )
        else:
            p = subprocess.check_output(
                f"/home/yuyang/smtk-master{no}/build/smraiz -sumsize {B} \
                             -flnpy tmp/{no}/{smtk}-{c}.npy -pnpv -porder -ptime".split()
            )
        s = p.decode("utf-8")
        str, end = ["([", ",])"]
        order = s[s.find(str) + len(str) : s.rfind(end)].split(",")
        greedy_time = float(s[s.find("CPU") + 4 : s.find("s (User")])
        str = "f(Solution) = "
        F_val = float(s[s.find(str) + len(str) : s.find("Summary Solution") - 1])
    else:
        V = list(range(N))
        start = time.time()
        # encourage higher diversity
        F = FacilityLocation(S, V)
        order, _ = lazy_greedy(F, V, B)
        greedy_time = time.time() - start
        F_val = 0  # TODO

    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(B, dtype=np.float64)
    for i in range(N):
        max_loc = np.argmax(S[i, order])
        if weights is None:
            while sz[max_loc] == subset_size:
                S[i, order][max_loc] = 0
                if np.max(S[i, order]) == 0:
                    break
                max_loc = np.argmax(S[i, order])
            sz[max_loc] += 1
        else:
            sz[max_loc] += weights[i]
    collected = gc.collect()
    return order, sz, greedy_time, F_val


def faciliy_location_order(
    c, X, y, metric, num_per_class, smtk, no, stoch_greedy, weights=None
):
    # print("here -1")
    class_indices = np.where(y == c)[0]
    # print(f"here {len(class_indices)}")
    S, S_time = similarity(X[class_indices], metric=metric)
    # print("here 2")
    order, cluster_sz, greedy_time, F_val = get_facility_location_submodular_order(
        S, num_per_class, c, smtk, no, stoch_greedy, weights
    )
    return class_indices[order], cluster_sz, greedy_time, S_time


def get_orders_and_weights(
    B,
    X,
    metric,
    smtk=0,
    no=0,
    stoch_greedy=0,
    y=None,
    weights=None,
    equal_num=False,
    outdir=".",
):  # todo
    """
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    """
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    classes = np.unique(y)
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(
            np.ceil(np.divide([sum(y == i) for i in classes], N) * B)
        )
        # print("not equal_num")

    print(f'Greedy: selecting {num_per_class} elements')

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(
        *map(
            lambda c: faciliy_location_order(
                c, X, y, metric, num_per_class[c], smtk, no, stoch_greedy, weights
            ),
            classes,
        )
    )

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios))  # TODO

    order_mg_all = np.array(order_mg_all)
    cluster_sizes_all = np.array(cluster_sizes_all)
    for i in range(
        int(np.rint(np.max([len(order_mg_all[c]) / props[c] for c in classes])))
    ):
        for c in classes:
            ndx = slice(
                i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c]))
            )
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])
    order_mg = np.array(order_mg, dtype=np.int32)

    weights_mg = np.array(
        weights_mg, dtype=np.float32
    )  
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = [] 
    weights_sz = (
        []
    )  
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals
