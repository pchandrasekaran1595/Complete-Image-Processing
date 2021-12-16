import re
import cv2
import pytest
import numpy as np  
from CLI.utils import image_handler, image_processor, ImageInferer

image_1 = image_handler.read_image("Files/Demo_Image_1.jpg")
image_2 = image_handler.read_image("Files/Demo_Image_2.jpg")

#######################################################################################################

@pytest.mark.parametrize("image, kernel_size, sigmaX",[(image_1, 5, 0), (image_2, 15, 3)])
def test_gaussian_blur_1(image, kernel_size, sigmaX):   
    assert image_processor.gauss_blur(image, kernel_size, sigmaX).all() == cv2.GaussianBlur(src=image, ksize=(kernel_size, kernel_size), sigmaX=sigmaX).all()


@pytest.mark.parametrize("image, kernel_size, sigmaX",[(image_1, 24, 0), (image_2, 26, 12)])
def test_gaussian_blur_2(image, kernel_size, sigmaX):   
    assert image_processor.gauss_blur(image, kernel_size, sigmaX).all() == cv2.GaussianBlur(src=image, ksize=(kernel_size+1, kernel_size+1), sigmaX=sigmaX).all()


@pytest.mark.parametrize("image, kernel_size",[(image_1, 24), (image_2, 51)])
def test_average_blur(image, kernel_size):
    assert image_processor.average_blur(image, kernel_size).all() == cv2.blur(src=image, ksize=(kernel_size, kernel_size)).all()


@pytest.mark.parametrize("image, kernel_size",[(image_1, 5), (image_2, 15)])
def test_median_blur_1(image, kernel_size):   
    assert image_processor.median_blur(image, kernel_size).all() == cv2.medianBlur(src=image, ksize=kernel_size).all()


@pytest.mark.parametrize("image, kernel_size",[(image_1, 24), (image_2, 26)])
def test_median_blur_2(image, kernel_size):   
    assert image_processor.median_blur(image, kernel_size).all() == cv2.medianBlur(src=image, ksize=kernel_size+1).all()

#######################################################################################################

@pytest.mark.parametrize("image, gamma",[(image_1, 1.8), (image_2, 0.5)])
def test_gamma(image, gamma): 
    image = image / 255
    image = np.clip(((image ** gamma) * 255), 0, 255).astype("uint8")
    assert image_processor.adjust_gamma(image, gamma).all() == image.all()


@pytest.mark.parametrize("image, linear",[(image_1, 100), (image_2, 75.625)])
def test_linear_contrast(image, linear): 
    assert image_processor.adjust_linear_contrast(image, linear).all() == np.clip((image + linear), 0, 255).astype("uint8").all()


@pytest.mark.parametrize("image, clipLimit, tileGridSize",[(image_1, 24, 1), (image_2, 2.5, 4)])
def test_adaptive_equalization(image, clipLimit, tileGridSize):  
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    if len(image.shape) == 3:
        for i in range(3):
            image[:, :, i] = clahe.apply(image[:, :, i])
    else:
        image = clahe.apply(image) 
    assert image_processor.adaptive_equalization(image, clipLimit, tileGridSize).all() == image.all()


@pytest.mark.parametrize("image",[(image_1), (image_2)])
def test_histogram_equalization(image):  
    if len(image.shape) == 3:
        for i in range(3):
            image[:, :, i] = cv2.equalizeHist(image[:, :, i])
    else:
        image = cv2.equalizeHist(image) 
    assert image_processor.histogram_equalization(image).all() == image.all()

#######################################################################################################

@pytest.mark.parametrize("image, hue",[(image_1, 2.5), (image_2, 75.625)])
def test_hue(image, hue): 
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
    feature = image[:, :, 0]
    feature = np.clip((hue * feature), 0, 179).astype("uint8")
    image[:, :, 0] = feature
    image = cv2.cvtColor(src=image, code=cv2.COLOR_HSV2BGR)
    assert image_processor.adjust_hue(image, hue).all() == image.all()


@pytest.mark.parametrize("image, saturation",[(image_1, 2.932), (image_2, 0.61225)])
def test_saturation(image, saturation): 
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
    feature = image[:, :, 1]
    feature = np.clip((saturation * feature), 0, 255).astype("uint8")
    image[:, :, 1] = feature
    image = cv2.cvtColor(src=image, code=cv2.COLOR_HSV2BGR)
    assert image_processor.adjust_saturation(image, saturation).all() == image.all()


@pytest.mark.parametrize("image, vibrance",[(image_1, 0.25), (image_2, 1.632349)])
def test_vibrance(image, vibrance): 
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
    feature = image[:, :, 2]
    feature = np.clip((vibrance * feature), 0, 255).astype("uint8")
    image[:, :, 2] = feature
    image = cv2.cvtColor(src=image, code=cv2.COLOR_HSV2BGR)
    assert image_processor.adjust_vibrance(image, vibrance).all() == image.all()

#######################################################################################################

@pytest.mark.parametrize("image, width, height",[(image_1, 640, 360), (image_2, 32, 32)])
def test_resize(image, width, height):  
    assert image_processor.resize_image(image, width, height).all() == cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_AREA).all()

#######################################################################################################

@pytest.mark.parametrize("image, sharpen",[(image_1, 3), (image_2, 7)])
def test_sharpen_1(image, sharpen): 
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(sharpen, sharpen)) * -1
    kernel[int(sharpen / 2), int(sharpen / 2)] = ((sharpen - 1) * 2) + 1

    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    image = np.clip(image, 0, 255).astype("uint8")
    assert image_processor.sharpen(image, sharpen).all() == image.all()


@pytest.mark.parametrize("image, sharpen",[(image_1, 4), (image_2, 12)])
def test_sharpen_2(image, sharpen): 
    sharpen += 1
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(sharpen, sharpen)) * -1
    kernel[int(sharpen / 2), int(sharpen / 2)] = ((sharpen - 1) * 2) + 1

    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    image = np.clip(image, 0, 255).astype("uint8")
    assert image_processor.sharpen(image, sharpen).all() == image.all()

#######################################################################################################

@pytest.mark.parametrize("image_1, image_2, alpha",[(image_1, image_2, 0.1), (image_2, image_1, 0.5)])
def test_alpha(image_1, image_2, alpha): 
    h1, w1, _ = image_1.shape
    h2, w2, _ = image_2.shape
    if w1 != w2 or h1 != h2:
        image_2 = cv2.resize(src=image_2, dsize=(w1, h1), interpolation=cv2.INTER_AREA)
    assert image_processor.alpha_blend(image_1, image_2, alpha).all() == cv2.addWeighted(image_1, alpha, image_2, 1-alpha, 0).all()

#######################################################################################################

@pytest.mark.parametrize("image, label",[(image_1, r"Sports Car"), (image_2, r"Airliner")])
def test_classifier(image, label): 
    image_inferer = ImageInferer(infer_type="classify")
    image_inferer.setup()

    assert re.match(label, image_inferer.infer(image), re.IGNORECASE)


@pytest.mark.parametrize("image, label",[(image_1, "Car"), (image_2, "Airplane")])
def test_detector(image, label): 
    image_inferer = ImageInferer(infer_type="detect")
    image_inferer.setup()

    assert re.match(label, image_inferer.infer(image, image.copy(), image.shape[1], image.shape[0]), re.IGNORECASE)


@pytest.mark.parametrize("image, label",[(image_1, "Car"), (image_2, "Aeroplane")])
def test_segmenter(image, label): 
    image_inferer = ImageInferer(infer_type="segment")
    image_inferer.setup()
    _, detected_label = image_inferer.infer(image, None, image.shape[1], image.shape[0])

    assert re.match(label, detected_label[0], re.IGNORECASE)

#######################################################################################################
