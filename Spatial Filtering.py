import numpy
import matplotlib.pyplot as mplplt
import matplotlib.image as mplimg

#Function for Creating any Size Gaussian Kernal based on Gaussian Formula
def createGaussian(len, sigma):
    gaussian = numpy.zeros((len, len))
    sideMax = numpy.int(numpy.floor(len / 2))
    sideMin = -sideMax
    total = 0
    for s in range(sideMin, sideMax+1):
        for t in range(sideMin, sideMax + 1):
            gaussian[s+sideMax][t+sideMax] = numpy.exp(-(s*s+t*t)/(2*sigma*sigma))
            total += gaussian[s+sideMax][t+sideMax]
    gaussian = gaussian/total
    return gaussian

##Upload Images to Test
#img = mplimg.imread(r'testImage.jpg')
img = mplimg.imread(r'Monarch.jpg')

##Convert Image Into Grayscale
def rgb2gray(image):
    imgNew = numpy.dot(image[...,:3], [0.2126, 0.7152, 0.0722])
    return imgNew

grayImg = numpy.round(rgb2gray(img))

#Display Grayscaled Iamge
mplplt.imshow(grayImg, cmap = 'gray')
mplplt.title("Original Image in Grayscale")
mplplt.show()

##Each individual element of the convoluted Image is computed with convMult() [used in my convolution() fuction]
def convMult(img, matrix, len, sideMax, x, y):
    imgFiltEntry = 0.0
    for i in range(0, len):
        for j in range (0, len):
            imgFiltEntry = imgFiltEntry + matrix[j,i]*img[x-sideMax+j, y-sideMax+i]
    return  imgFiltEntry

## This will produce the entire convoluted image. Applies the convMult() function.
## Can convolute for any size NxN matrix.
def convolution(matrix, len, img, imgXSize, imgYSize):
    imgFilt = numpy.zeros((imgXSize, imgYSize))*1.0 #creates the zeroed out "canvas" that our filtered Image will be on
    sideMax = numpy.int(numpy.floor(len / 2)) #figures out the max X and Y position of a centered kernel
    for x in range(sideMax,imgXSize-sideMax):
        for y in range (sideMax, imgYSize-sideMax):
            imgFilt[x,y] = convMult(img, matrix, len, sideMax, x, y) #each element of the convoluted Image is computed with convMult()
    return imgFilt

#Find the Size of Each Dimension of the Original Image
imgXSize, imgYSize = numpy.shape(grayImg)

#List of Kernals
average5x5 = numpy.ones((5,5))/25
sharpen3x3 = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])/9.0
laplacian5x5 = numpy.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1 ,-1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])/24.0
gaussLen = 7
sigma = 1
gauss7x7 = createGaussian(gaussLen, sigma) #arg[1] = length of one Side, arg[2] = sigma

#Compute and Display Sharpened Filtered Image
imgSharpen = convolution(sharpen3x3, 3, grayImg, imgXSize, imgYSize)
imgSharpen = imgSharpen.astype('uint8')
mplplt.imshow(imgSharpen, cmap='gray')
mplplt.title("3x3 Sharpen Kernal Filtered Image")
mplplt.show()

#Compute and Display Gaussian Filtered Image
imgGauss = convolution(gauss7x7, 7, grayImg, imgXSize, imgYSize)
imgGauss = imgGauss.astype('uint8')
mplplt.imshow(imgGauss, cmap='gray')
mplplt.title("7x7 Gaussian Kernal Filted Image")
mplplt.show()

#Compute and Display Laplacian of Gaussian Filtered Image
imgLoG = convolution(laplacian5x5, 5, imgGauss, imgXSize, imgYSize)
imgLoG = imgLoG.astype('uint8')
mplplt.imshow(imgLoG, cmap='gray')
mplplt.title("Image after Laplacian of Gaussian")
mplplt.show()


print("Hello World")
