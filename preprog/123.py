import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    A = cv2.imread('./19615.jpg')
    print(A)
    plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
    plt.show()
