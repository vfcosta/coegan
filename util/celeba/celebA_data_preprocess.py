import os
from torchvision.transforms import functional as F
from scipy.misc import imresize
from PIL import Image


def preprocess_celeba(crop_size=108, resize_size=32):
    root = 'data/CelebA/splits/train/'
    save_root = f"data/CelebA{resize_size}Cropped/train/"

    os.makedirs(save_root)
    img_list = os.listdir(root)

    for i in range(len(img_list)):
        img = Image.open(root + img_list[i]).convert('RGB')
        img = F.resize(img, crop_size)
        img = F.center_crop(img, 64)
        img = imresize(img, (resize_size, resize_size))
        Image.fromarray(img).save(save_root + img_list[i])

        if (i % 1000) == 0:
            print('%d images complete' % i)


if __name__ == '__main__':
    preprocess_celeba()
