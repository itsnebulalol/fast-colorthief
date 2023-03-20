import fast_colorthief
import numpy as np
import cv2
import colorthief
import PIL
import time


def test_some_output_returned():
    result = fast_colorthief.get_dominant_color('tests/veverka_lidl.jpg', 10)
    assert result == (133, 124, 90)


def test_same_output(image_path, quality=10):
    fast_palette = fast_colorthief.get_palette(image_path, 5, quality)
    colorthief_orig = colorthief.ColorThief(image_path)
    original_palette = colorthief_orig.get_palette(5, quality)
    if fast_palette != original_palette:
        print(f"Original {original_palette}")
        print(f"C++ {fast_palette}")

    #assert (fast_palette == original_palette)


def print_speed(image_path, iterations=10, quality=1):
    image = cv2.imread(image_path)
    image = PIL.Image.open(image_path)
    image = image.convert('RGBA')
    image = np.array(image).astype(np.uint8)

    start = time.time()

    for i in range(iterations):
        fast_colorthief.get_palette(image, quality=quality)

    print(f'CPP numpy {(time.time() - start) / iterations}')

    start = time.time()

    for i in range(iterations):
        fast_colorthief.get_palette(image_path, quality=quality)

    print(f'CPP image path {(time.time() - start) / iterations}')

    start = time.time()

    for i in range(iterations):
        colorthief_orig = colorthief.ColorThief(image_path)
        colorthief_orig.get_palette(quality=quality)

    print(f'Python image path {(time.time() - start) / iterations}')


if __name__ == '__main__':
    test_same_output('tests/veverka_lidl.jpg', quality=1)
    test_same_output('tests/monastery.jpg', quality=1)
    #print("Normal size image, bad quality")
    #print_speed('tests/veverka_lidl.jpg', iterations=10, quality=10)
    #print("\nNormal size image, best quality")
    #print_speed('tests/veverka_lidl.jpg', iterations=10)
    print("\nHuge image, best quality")
    print_speed('tests/monastery.jpg', iterations=1)


    if False:
        wrong_original = []
        exceptions = []

        from os import listdir
        from os.path import isfile, join

        path = '/data/logo_detection/dataset_version_10/train/images'

        for i, image_path in enumerate([join(path, f) for f in listdir(path)]):
            print(f"{i} {image_path}")

            try:
                fast_palette = fast_colorthief.get_palette(image_path, 5, 10)
                colorthief_orig = colorthief.ColorThief(image_path)
                original_palette = colorthief_orig.get_palette(5, 10)
            except RuntimeError:
                exceptions.append(image_path)
                continue

            wrong_original_output = False
            for rgb in original_palette:
                if max(rgb) >= 256:  # error in original colorthief
                    wrong_original_output = True
                    break

            if wrong_original_output:
                wrong_original.append(image_path)
                continue

            assert (fast_palette == original_palette)

        print(f'Wrong original output: {wrong_original}')
        print(f'Exception raised: {exceptions}')
