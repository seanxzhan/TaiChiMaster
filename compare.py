import numpy as np
from scipy.spatial import distance


def penalty(x):
    return 3*x


def angle_similarity(real, truth):
    cos_dist = distance.cosine(real, truth)
    cos = 1 - cos_dist
    angle = np.arccos(cos)
    return 1 - penalty(angle / np.pi)


# gets accuracy for single real image compared to max similarity with truth images
def get_accuracy(real, truth):
    accuracies = np.array(
        [angle_similarity(real, t) for t in truth])
    return np.max(accuracies)

def compare(real, truth, window_size=10):
    accuracies = []
    for i, img in enumerate(real):
        accuracies.append(get_accuracy(img, truth[i: i + window_size]))
        # accuracies.append(get_accuracy(img, truth))
    return round(np.mean(accuracies) * 100)

def test():
    real = np.random.rand(100, 8)
    truth = np.random.rand(100, 8)

    accuracy = compare(real, truth)
    print(f'{accuracy}%')

def main():
    test()

if __name__ == '__main__':
    main()