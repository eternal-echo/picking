import csv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # "0.34321117401123047,0.3471043109893799,0.362485408782959..."
    with open('run/detect.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)[0]
        print(data)
        data = [float(i) for i in data]
        print(np.mean(data))
        print(np.std(data))
        plt.hist(data, bins=100)
        plt.show()