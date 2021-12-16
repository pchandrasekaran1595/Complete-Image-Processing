import sys
from os import listdir

from .utils import READ_PATH, image_handler, image_processor


def run():
    args_1 = ["--file1", "-f1"]
    args_2 = ["--file2", "-f2"]
    args_3  = ["--gauss-blur", "-gb"]
    args_4  = ["--avg-blur", "-ab"]
    args_5  = ["--median-blur", "-mb"]
    args_6  = ["--gamma", "-g"]
    args_7  = ["--linear", "-l"]
    args_8  = ["--clahe", "-ae"]
    args_9  = ["--hist-equ", "-he"]
    args_10  = ["--hue"]
    args_11 = ["--saturation", "-sat"]
    args_12 = ["--vibrance", "-v"]
    args_13 = ["--width", "-w"]
    args_14 = ["--height", "-h"]
    args_15 = ["--sharpen", "-sh"]
    args_16 = ["--posterize", "-post"]
    args_17 = ["--dither", "-dit"]
    args_18 = ["--alpha", "-a"]
    args_19 = ["--combine", "-c"]
    args_20 = ["--vertical", "-ver"]
    args_21 = ["--adapt-big", "-abig"]
    args_22 = ["--classify", "-cl"]
    args_23 = ["--detect", "-dt"]
    args_24 = ["--detect-all", "-dta"]
    args_25 = ["--segment", "-seg"]
    args_26 = ["--save", "-s"]
    args_27 = "-wf"

    filename_1 = None
    filename_2 = None
    do_gauss_blur = False
    do_average_blur = False
    do_median_blur = False
    do_gamma = False
    do_linear = False
    do_clahe = False
    do_histogram_equalization = False
    do_hue = False
    do_saturation = False
    do_vibrance = False
    width, height = None, None
    do_sharpen = False
    do_posterize = False
    do_dither = False
    alpha = None
    do_combine = False
    vertical = False
    adapt_small = True
    save = False
    workflow = False

    if args_1[0] in sys.argv: filename_1 = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: filename_1 = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: filename_2 = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: filename_2 = sys.argv[sys.argv.index(args_2[1]) + 1]

    if args_3[0] in sys.argv: 
        do_gauss_blur = True
        setup = sys.argv[sys.argv.index(args_3[0]) + 1] + ","
        gaussian_blur_kernel_size = setup.split(",")[0]
        gaussian_blur_sigmaX = setup.split(",")[1]
    if args_3[1] in sys.argv: 
        do_gauss_blur = True
        setup = sys.argv[sys.argv.index(args_3[1]) + 1] + ","
        gaussian_blur_kernel_size = setup.split(",")[0]
        gaussian_blur_sigmaX = setup.split(",")[1]
    
    if args_4[0] in sys.argv: 
        do_average_blur = True
        average_blur_kernel_size = int(sys.argv[sys.argv.index(args_4[0]) + 1])
    if args_4[1] in sys.argv: 
        do_average_blur = True
        average_blur_kernel_size = int(sys.argv[sys.argv.index(args_4[1]) + 1])
    
    if args_5[0] in sys.argv: 
        do_median_blur = True
        median_blur_kernel_size = sys.argv[sys.argv.index(args_5[0]) + 1]
    if args_5[1] in sys.argv: 
        do_median_blur = True
        median_blur_kernel_size = sys.argv[sys.argv.index(args_5[1]) + 1]
    
    if args_6[0] in sys.argv: 
        do_gamma = True
        gamma = float(sys.argv[sys.argv.index(args_6[0]) + 1])
    if args_6[1] in sys.argv: 
        do_gamma = True
        gamma = float(sys.argv[sys.argv.index(args_6[1]) + 1])
    
    if args_7[0] in sys.argv: 
        do_linear = True
        linear = float(sys.argv[sys.argv.index(args_7[0]) + 1])
    if args_7[1] in sys.argv: 
        do_linear= True
        linear = float(sys.argv[sys.argv.index(args_7[1]) + 1])
    
    if args_8[0] in sys.argv: 
        do_clahe = True
        setup = sys.argv[sys.argv.index(args_8[0]) + 1] + ","
        clipLimit = float(setup.split(",")[0])
        tileGridSize = setup.split(",")[1]
    if args_8[1] in sys.argv: 
        do_clahe = True
        setup = sys.argv[sys.argv.index(args_8[1]) + 1] + ","
        clipLimit = float(setup.split(",")[0])
        tileGridSize = setup.split(",")[1]
    
    if args_9[0] in sys.argv: 
        do_histogram_equalization = True
    if args_9[1] in sys.argv: 
        do_histogram_equalization = True
    
    if args_10[0] in sys.argv:
        do_hue = True
        hue = float(sys.argv[sys.argv.index(args_10[0]) + 1])
    
    if args_11[0] in sys.argv:
        do_saturation = True
        saturation = float(sys.argv[sys.argv.index(args_11[0]) + 1])
    if args_11[1] in sys.argv:
        do_saturation = True
        saturation = float(sys.argv[sys.argv.index(args_11[1]) + 1])
    
    if args_12[0] in sys.argv:
        do_vibrance = True
        vibrance = float(sys.argv[sys.argv.index(args_12[0]) + 1])
    if args_12[1] in sys.argv:
        do_vibrance = True
        vibrance = float(sys.argv[sys.argv.index(args_12[1]) + 1])
    
    if args_13[0] in sys.argv: width = int(sys.argv[sys.argv.index(args_13[0]) + 1])
    if args_13[1] in sys.argv: width = int(sys.argv[sys.argv.index(args_13[1]) + 1])
    
    if args_14[0] in sys.argv: height = int(sys.argv[sys.argv.index(args_14[0]) + 1])
    if args_14[1] in sys.argv: height = int(sys.argv[sys.argv.index(args_14[1]) + 1])
    
    if args_15[0] in sys.argv:
        do_sharpen = True
        sharpen_kernel_size = sys.argv[sys.argv.index(args_15[0]) + 1]
    if args_15[1] in sys.argv:
        do_sharpen = True
        sharpen_kernel_size = sys.argv[sys.argv.index(args_15[1]) + 1]
    
    if args_16[0] in sys.argv:
        do_posterize = True
        num_colors = int(sys.argv[sys.argv.index(args_16[0]) + 1])
    if args_16[1] in sys.argv:
        do_posterize = True
        num_colors = int(sys.argv[sys.argv.index(args_16[1]) + 1])
    
    if args_17[0] in sys.argv:
        do_dither = True
        num_colors = int(sys.argv[sys.argv.index(args_17[0]) + 1])
    if args_17[1] in sys.argv:
        do_dither = True
        num_colors = int(sys.argv[sys.argv.index(args_17[1]) + 1])
    
    if args_18[0] in sys.argv: alpha = float(sys.argv[sys.argv.index(args_18[0]) + 1])
    if args_18[1] in sys.argv: alpha = float(sys.argv[sys.argv.index(args_18[1]) + 1])

    if args_19[0] in sys.argv or args_19[1] in sys.argv: do_combine = True
    if args_20[0] in sys.argv or args_20[1] in sys.argv: 
        do_combine = True
        vertical = True
    if args_21[0] in sys.argv or args_21[1] in sys.argv: 
        do_combine = True
        adapt_small = False

    if args_26[0] in sys.argv or args_26[1] in sys.argv:  save = True
    if args_27 in sys.argv: workflow = True

    assert filename_1 is not None, "Enter argument for --file1 | -f1"
    assert filename_1 in listdir(READ_PATH), "File 1 Not Found"

    image =  image_handler.read_image(READ_PATH + "/" + filename_1)

    if do_gauss_blur: image = image_processor.gauss_blur(image=image, kernel_size=gaussian_blur_kernel_size, sigmaX=gaussian_blur_sigmaX)
    if do_average_blur: image = image_processor.average_blur(image=image, kernel_size=average_blur_kernel_size)
    if do_median_blur: image = image_processor.median_blur(image=image, kernel_size=median_blur_kernel_size)
    if do_gamma: image = image_processor.adjust_gamma(image=image, gamma=gamma)
    if do_linear: image = image_processor.adjust_linear_contrast(image=image, alpha=linear)
    if do_clahe: image = image_processor.adaptive_equalization(image=image, clipLimit=clipLimit, tileGridSize=tileGridSize)
    if do_histogram_equalization: image = image_processor.histogram_equalization(image=image)
    if do_hue: image = image_processor.adjust_hue(image=image, hue=hue)
    if do_saturation: image = image_processor.adjust_saturation(image=image, saturation=saturation)
    if do_vibrance: image = image_processor.adjust_vibrance(image=image, vibrance=vibrance)
    if do_sharpen: image = image_processor.sharpen(image=image, kernel_size=sharpen_kernel_size)

    if isinstance(width, int) and height is None:
        h, _, _ = image.shape
        image = image_processor.resize_image(image, width, h)

    if width is None and isinstance(height, int):
        _, w, _ = image.shape
        image = image_processor.resize_image(image, w, height)
    
    if isinstance(width, int) and isinstance(height, int):
        image = image_processor.resize_image(image, width, height)
    
    if do_posterize: image = image_processor.posterize_image(image=image, num_colors=num_colors)
    if do_dither: image = image_processor.dither_image(image=image, num_colors=num_colors)

    if isinstance(alpha, float):
        assert filename_2 is not None, "Enter argument for --file2 | -f2"
        assert filename_2 in listdir(READ_PATH), "File 2 Not Found"

        image_2 = image_handler.read_image(READ_PATH + "/" + filename_2)
        image = image_processor.alpha_blend(image, image_2, alpha)
    
    if do_combine: 
        assert filename_2 is not None, "Enter argument for --file2 | -f2"
        assert filename_2 in listdir(READ_PATH), "File 2 Not Found"

        image_2 = image_handler.read_image(READ_PATH + "/" + filename_2)
        image = image_processor.combine(image, image_2, vertical=vertical, adapt_small=adapt_small)

    if not save:
        if not workflow:
            image_handler.show(image)
    else:
        image_handler.save_image(image)
    