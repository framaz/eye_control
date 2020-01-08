from PIL import Image
from matplotlib import pyplot as plt

def pack_to_one_image(*args):
    sizes = list(map(lambda image: image.size, args))
    heigth = max(map(lambda size: size[1], sizes))
    length = sum(map(lambda size: size[0], sizes))
    mode = args[0].mode
    pointer = 0
    result = Image.new(mode, (length, heigth))
    for pic in args:
        result.paste(pic, (pointer, 0))
        pointer += pic.width
    return result

if __name__ == "__main__":
    first_img = Image.open("../chel.jpg")
    second = Image.open("../chel2.jpg")
    plt.imshow(pack_to_one_image(first_img, second))
    plt.show()
