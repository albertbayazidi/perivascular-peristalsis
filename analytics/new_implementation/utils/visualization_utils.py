import matplotlib.pyplot as plt

def simpleplot(A):
    bindary_data = (A!=0).astype(int)
    plt.imshow(bindary_data, cmap='viridis')
    plt.title("new imp")
    plt.show()

