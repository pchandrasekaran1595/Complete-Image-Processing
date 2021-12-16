import os

BASE_CMD = "python main.py -f Demo_Image_1.jpg -wf "

def run(args: list):
    for arg in args:
        os.system(BASE_CMD + arg)

#######################################################################################################

def test_blurs():
    args = ["-gb 25", "-gb 25,5", "--gauss-blur 24", "--gauss-blur 1,12.25",
            "-ab 11", "-ab 6", "--avg-blur 12", "--avg-blur 7",
            "-mb 8", "-mb 11", "--median-blur 4", "--median-blur 9"]
    
    run(args)  
#######################################################################################################

def test_contrasts():
    args = ["--gamma 2.2", "-g 0.25", "--linear 100", "-l 255", 
            "--clahe 5", "--clahe 5.25,5", "-ae 4.75", "-ae 2.25,3",
            "--hist-equ", "-he"]
    
    run(args)

#######################################################################################################

def test_hsv():
    args = ["--hue 2.25", "--saturation 0.5", "-sat 1.1021", "--vibrance 1.65", "-v 0.61234567"]

    run(args) 

#######################################################################################################

def test_resize():
    args = ["--width 640", "-w 580", "--height 360", "-h 480", "--width 224 --height 224", "-w 480 -h 320"]

    run(args)

#######################################################################################################

def test_sharpen():
    args = ["--sharpen 11", "-sh 6"]

    run(args)  

#######################################################################################################

def test_posterize():
    args = ["-w 224 -h 224 --posterize 4", "-w 224 -h 224 -post 6"]

    run(args)  

#######################################################################################################

def test_dither():
    args = ["-w 224 -h 224 --dither 4", "-w 224 -h 224 -dit 6"]

    run(args)     

#######################################################################################################

def test_alpha():
    args = ["--file2 Demo_Image_2.jpg --alpha 0.25", "-f2 Demo_Image_2.jpg -a 0.5"]

    run(args)  

#######################################################################################################

def test_combine():
    args = ["--file2 Demo_Image_2.jpg --combine", "-f2 Demo_Image_2.jpg -c",
            "--file2 Demo_Image_2.jpg --combine --vertical", "-f2 Demo_Image_2.jpg -c -ver",
            "--file2 Demo_Image_2.jpg --combine --adapt-big", "-f2 Demo_Image_2.jpg -c -abig",
            "--file2 Demo_Image_2.jpg --combine --vertical --adapt-big", "-f2 Demo_Image_2.jpg -c -ver -abig"]

    run(args)  

#######################################################################################################

def test_save():
    args = ["-gb 5 -s", "-ab 9 --save"]

    run(args)      
#######################################################################################################