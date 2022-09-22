import matplotlib.pyplot as plt
import json
import numpy as np

def plot_age_histogram(data):
    plt.title('Age distribution of catched boids (over 200 simulations)')
    plt.hist(data)
    plt.show()

if __name__ == "__main__":

    with open('./boids/output/Deleted_boids_setup1.json') as json_file:
        data = json.load(json_file)

    data = np.asarray(data)
    # data.flatten()
    # data = data.reshape(data.shape[0]*data.shape[1])

    plot_age_histogram(data)
