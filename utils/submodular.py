import time

import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction


def faciliy_location_order(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    if mode == "dense":
        num_n = None

    start = time.time()
    obj = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)

    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz == 0)] = 1

    return class_indices[order], sz, greedy_time, S_time


def get_orders_and_weights(
    B,
    X,
    metric,
    y=None,
    weights=None,
    equal_num=False,
    outdir=".",
    mode="sparse",
    num_n=128,
):
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
    classes = classes.astype(np.int32).tolist()
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

    print(f"Greedy: selecting {num_per_class} elements")

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(
        *map(
            lambda c: faciliy_location_order(
                c[1], X, y, metric, num_per_class[c[0]], weights, mode, num_n
            ),
            enumerate(classes),
        )
    )
    print(
        f"time (sec) for computing facility location: {greedy_times} similarity time {similarity_times}",
    )

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios))  # TODO

    order_mg_all = np.array(order_mg_all, dtype=object)
    cluster_sizes_all = np.array(cluster_sizes_all, dtype=object)

    for i in range(
        int(
            np.rint(
                np.max([len(order_mg_all[c]) / props[c] for c, _ in enumerate(classes)])
            )
        )
    ):
        for c, _ in enumerate(classes):
            ndx = slice(
                i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c]))
            )
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])

    order_mg = np.array(order_mg, dtype=np.int32)

    weights_mg = np.array(
        weights_mg, dtype=np.float32
    )  # / sum(weights_mg) TODO: removed division!
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []  # order_mg_all[rows_selector, cluster_order].flatten(order='F')
    weights_sz = (
        []
    )  # cluster_sizes_all[rows_selector, cluster_order].flatten(order='F')
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals


def greedy_merge(X, y, B, part_num, metric):
    N = len(X)
    indices = list(range(N))
    part_size = int(np.ceil(N / part_num))
    part_indices = [
        indices[slice(i * part_size, min((i + 1) * part_size, N))]
        for i in range(part_num)
    ]
    print(f"GreeDi with {part_num} parts, finding {B} elements...", flush=True)

    order_mg_all, cluster_sizes_all, _, _, ordering_time, similarity_time = zip(
        *map(
            lambda p: get_orders_and_weights(
                int(B / 2), X[part_indices[p], :], metric, y=y[part_indices[p]],
            ),
            np.arange(part_num),
        )
    )

    order_mg_all = list(order_mg_all)
    order_mg = np.concatenate(order_mg_all, dtype=np.int32)
    weights_mg = np.concatenate(cluster_sizes_all, dtype=np.float32)
    print(
        f"GreeDi stage 1: found {len(order_mg)} elements in: {np.max(ordering_time)} sec",
    )

    (
        order,
        weights,
        order_sz,
        weights_sz,
        ordering_time_merge,
        similarity_time_merge,
    ) = get_orders_and_weights(
        B, X[order_mg, :], metric, y=y[order_mg], weights=weights_mg,
    )

    total_ordering_time = np.max(ordering_time) + ordering_time_merge
    total_similarity_time = np.max(similarity_time) + similarity_time_merge
    print(
        f"GreeDi stage 2: found {len(order)} elements in: {total_ordering_time + total_similarity_time} sec",
    )
    vals = (
        order,
        weights,
        order_sz,
        weights_sz,
        total_ordering_time,
        total_similarity_time,
    )
    return vals
