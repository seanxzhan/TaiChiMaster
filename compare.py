import numpy as np

def angle_similarity(real, truth):
    distance = np.sqrt(np.sum(np.square(real - truth)))
    return 1 / (1 + distance)

# gets accuracy for single real image compared to max similarity with truth images
def get_accuracy(real, truth):
    accuracies = np.array([angle_similarity(real, t) for t in truth])
    return np.max(accuracies)

def compare(real, truth, window_size=10):
    accuracies = []
    for i, img in enumerate(real):
        accuracies.append(get_accuracy(img, truth[i: i + window_size]))
    return np.mean(accuracies)

def test():
    real = np.random.rand(100, 8)
    truth = np.random.rand(100, 8)

    accuracy = round(compare(real, truth) * 100)
    print(f'{accuracy}%')

def main():
    test()

if __name__ == '__main__':
    main()