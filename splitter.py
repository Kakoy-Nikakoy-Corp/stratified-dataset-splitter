from copy import deepcopy


def calculate_cost(candidate, target_samples, target_pos, weights):
    cost = 0.0

    for i in range(3):
        s = candidate[i]["total"]
        p = candidate[i]["pos"]

        ts, tp = target_samples[i], target_pos[i]

        # Штрафуем несоответствие размеру фолда
        cost += weights[0] * ((s - ts) / ts) ** 2

        # Штрафуем дисбаланс позитивных сэмплов
        if tp > 0:
            cost += weights[1] * ((p - tp) / tp) ** 2

    return cost


def greedy_loc_split(total_samples, total_pos, location_data, split, weights):
    pos_rate = total_pos / total_samples

    target_samples = [total_samples * ratio for ratio in split]
    target_pos = [ts * pos_rate for ts in target_samples]

    # Сначала распределяем локации с наибольшим числом позитивных сэмплов
    locations = sorted(location_data.items(), key=lambda x: x[1]["pos"], reverse=True)

    folds = [{"total": 0, "pos": 0, "count": 0} for _ in range(3)]

    mapping = {}
    fold_names = ["train", "val", "test"]

    # Для каждой локации пробуем положить её по очереди в каждый фолд
    for loc_name, data in locations:
        n_samples = data["total"]
        n_pos = data["pos"]

        best_fold = None
        best_cost = float("inf")

        for i in range(3):
            # Ограничение: пропускаем фолд, если локация занимает более четверти его объёма
            if n_samples > target_samples[i] * 0.25:
                continue

            candidate = deepcopy(folds)
            candidate[i]["total"] += n_samples
            candidate[i]["pos"] += n_pos

            # Находим штраф при использовании такого варианта распределения
            cost = calculate_cost(candidate, target_samples, target_pos, weights)

            if cost < best_cost:
                best_cost = cost
                best_fold = i

        # Выбираем фолд, при попадании в который локация минимизирует штраф
        mapping[loc_name] = fold_names[best_fold]

        folds[best_fold]["total"] += n_samples
        folds[best_fold]["pos"] += n_pos
        folds[best_fold]["count"] += 1

    print(folds)
    return mapping
