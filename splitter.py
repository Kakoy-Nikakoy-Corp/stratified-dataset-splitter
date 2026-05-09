from collections import defaultdict
from copy import deepcopy

POS_RATE = 0.49


def calculate_cost(candidate, target_samples, target_pos, weights):
    cost = 0
    for i in range(3):
        s = candidate[i]['total']
        p = candidate[i]['pos']

        ts = target_samples[i]
        tp = target_pos[i]

        cost += weights[0] * ((s - ts) / ts) ** 2

        if tp > 0:
            cost += weights[1] * ((p - tp) / tp) ** 2

    return cost


def greedy_loc_split(num_samples, location_data, split, weights):
    target_samples = []
    target_pos = []

    for i in range(3):
        sample = num_samples * split[i]
        target_samples.append(sample)
        target_pos.append(sample * POS_RATE)

    locations = list(location_data)
    locations.sort(key=lambda name: location_data[name]['pos'], reverse=True)

    mapping = defaultdict()
    state = [{'total': 0, 'pos': 0, 'count': 0} for _ in range(3)]
    for loc in locations:
        n_samples = location_data[loc]['total']
        n_pos = location_data[loc]['pos']

        best_fold = None
        best_cost = float('inf')

        for i in range(3):
            candidate = deepcopy(state)
            candidate[i]['total'] += n_samples
            candidate[i]['pos'] += n_pos

            # maintains location diversity inside split
            if n_samples > target_samples[i] * 0.25:
                continue

            cost = calculate_cost(candidate, target_samples, target_pos, weights)

            if cost < best_cost:
                best_cost = cost
                best_fold = i

        fold_name = ['train', 'val', 'test'][best_fold]
        mapping[loc] = fold_name

        state[best_fold]['total'] += n_samples
        state[best_fold]['pos'] += n_pos
        state[best_fold]['count'] += 1

    print(state)
    return dict(mapping)
