from functools import reduce


def gain(current, delta, target):
    if target == 0:
        return 0

    return -delta * (2 * (current - target) + delta) / target**2


def greedy_loc_split(location_data, split, weights):
    total_samples, total_pos = reduce(
        lambda x, y: [x[0] + y[0], x[1] + y[1]], location_data.values()
    )

    # Сначала распределяем локации с наибольшим числом позитивных сэмплов
    locations = sorted(location_data.items(), key=lambda x: x[1][1], reverse=True)

    folds = [{"samples": 0, "pos": 0, "loc_count": 0} for _ in range(3)]

    mapping = {}
    fold_names = ["train", "val", "test"]

    # Для каждой локации пробуем положить её по очереди в каждый фолд
    for loc_name, data in locations:
        ds, dp = data

        best_fold = None
        best_gain = float("-inf")

        for i in range(3):
            target_s = total_samples * split[i]
            target_p = total_pos * split[i]

            # Ограничение: пропускаем фолд, если локация занимает более четверти его объёма
            if ds > target_s * 0.25:
                continue

            s = folds[i]["samples"]
            p = folds[i]["pos"]

            # Находим "прирост корректности" при использовании такого варианта распределения
            total_gain = (weights[0] * gain(s, ds, target_s)
                          + weights[1] * gain(p, dp, target_p))

            if total_gain > best_gain:
                best_gain = total_gain
                best_fold = i

        # Выбираем фолд, при попадании в который локация максимизирует прирост
        mapping[loc_name] = fold_names[best_fold]

        folds[best_fold]["samples"] += ds
        folds[best_fold]["pos"] += dp
        folds[best_fold]["loc_count"] += 1

    print(folds)
    return mapping
