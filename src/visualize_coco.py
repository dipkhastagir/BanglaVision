from PIL import Image
import matplotlib.pyplot as plt

def show_sample(image_path, caption):
    img = Image.open(image_path)

    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()