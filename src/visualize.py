# get images from https://colab.research.google.com/drive/1aispWvWoYWY0DCZaM6UohnAg81SJDgot?usp=sharing
# download images into directory
import os
import matplotlib.pyplot as plt
import numpy as np

result_dir = "../images/"

raw_scores = []
human_scores = []
model_scores = []


def get_average(file):
    f = open(file, 'r')
    score_sum = 0
    i = -1
    for line in f:
        if i == -1:
            i += 1
            continue
        score = float(line)
        if 3.36 < score or score < 3.35:
            score_sum += score
            i += 1

    f.close()
    if i == 0:
        return -1
    else:
        return score_sum / i


sample_num = len(os.listdir(result_dir))
for i in range(0, sample_num):
    cur_dir = result_dir + str(i) + "/"
    human = cur_dir + "human/score.txt"
    raw = cur_dir + "raw/score.txt"
    model = cur_dir + "model/score.txt"

    h = get_average(human)
    r = get_average(raw)
    m = get_average(model)

    if h != -1 and r != -1 and m != -1:
        raw_scores.append(r)
        human_scores.append(h)
        model_scores.append(m)

print(raw_scores)
print(human_scores)
print(model_scores)

raw_scores.sort()
human_scores.sort()
model_scores.sort()


def prefix_sum(test_list):
    return [sum(test_list[ : i + 1]) for i in range(len(test_list))]


def get_distribution(array):
    dis = np.zeros(100)
    for element in array:
        dis[int(element * 10)] += 1
    return normalize(dis, 0, 1)

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def get_cdf(array):
    return normalize(prefix_sum(get_distribution(array)), 0, 1)


begin = 50
end = 90
xarray = np.arange(0, 10, 0.1)
plt.plot(xarray[begin:end], get_distribution(raw_scores)[begin:end], label="raw")
plt.plot(xarray[begin:end], get_distribution(human_scores)[begin:end], label="human")
plt.plot(xarray[begin:end], get_distribution(model_scores)[begin:end], label="model")
plt.xlabel("aesthetic score")
plt.ylabel("cumulative distribution")
plt.legend(loc="upper left")
plt.show()