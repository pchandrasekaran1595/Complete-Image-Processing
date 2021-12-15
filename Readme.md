## **Image Processing from the Command Line**

- Prerequisites --> Python 3.8.x or 3.9.x
- Read Path &nbsp;--> `./Files`
- Write Path --> `./Processed`

<br>

## **CLI Arguments**

<br>

- `--file, -f` - Image Filename (including extension)

<br>

- `--gauss-blur, -gb` &nbsp; - Gaussian Blur Kernel Size,Gaussian SigmaX(Optional)
- `--avg-blur, -ab` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Average Blur Kernel Size 
- `--median-blur, -mb` - Median Blur Kernel Size

<br>

- `--gamma, -g` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Gamma Value
- `--linear, -l` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Linear Contrast Alpha
- `--clahe, -ae` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Cliplimit,Tile Grid Size(Optional)
- `--hist-equ, -he` - Histogram Equalization Flag

<br>

- `--hue` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Hue Multiplier
- `--saturation, -sat` - Saturation Multiplier
- `--vibrance, -v` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Vibrance Multiplier

<br>

- `--width, -w` &nbsp; - New Width
- `--height, -h` - New Height

<br>

- `--sharpen, -sh` - Sharpen Kernel Size

<br>

- `--posterize, -post` - Number of Colors in result
- `--dither, -dit` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Number of Colors in result

<br>

- `--save, -s` - Save Processed Image Flag