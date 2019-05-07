import numpy
import matplotlib.pyplot as mplplt
import matplotlib.image as mplimg

#Create Gaussian Kernal based on Gaussian Formula
def createGaussian(K, len, sigma):
    gaussian = numpy.zeros((len, len))
    sideMax = numpy.int(numpy.floor(len / 2))
    sideMin = -sideMax
    for s in range(sideMin, sideMax+1):
        for t in range(sideMin, sideMax + 1):
            gaussian[s+sideMax][t+sideMax] = K*numpy.exp(-(s*s+t*t)/(2*sigma*sigma))
    return gaussian

gauss5x5 = createGaussian(0.15, 5, 1) #arg[1] = K, arg[2] = length of one Side, arg[3] = sigma
#No Need to Flip Kernel since symmetrical

img = mplimg.imread('testImage.jpg')  #Read the image from a file in the same folderspace
#img = mplimg.imread('LeafSmall.jpg')  #Read the image from a file in the same folderspace

def rgb2gray(image):
    imgNew = numpy.dot(image[...,:3], [0.2126, 0.7152, 0.0722])
    return imgNew

grayImg = numpy.round(rgb2gray(img))
mplplt.imshow(grayImg, cmap = 'gray')
mplplt.show()

imgXSize, _ = numpy.shape(grayImg)
_, imgYSize = numpy.shape(grayImg)

#print(imgXSize)

def convMult(img, matrix, len, sideMax, x, y):
    imgFiltEntry = 0.0
    for i in range(0, len):
        for j in range (0, len):
            imgFiltEntry = imgFiltEntry + matrix[j,i]*img[x-sideMax+j, y-sideMax+i]
    return  imgFiltEntry

def convolution(matrix, len, img, imgXSize, imgYSize):
    imgFilt = numpy.zeros((imgXSize, imgYSize))*1.0
    sideMax = numpy.int(numpy.floor(len / 2))
    for x in range(sideMax,imgXSize-sideMax):
        for y in range (sideMax, imgYSize-sideMax):
            imgFilt[x,y] = convMult(img, matrix, len, sideMax, x, y)
    return imgFilt

identity = numpy.array([[0,0,0],[0,1,0],[0,0,0]])
#gaussHard3x3 = numpy.array([[1,2,1], [2,4,2], [1,2,1]])/16
#gaussHard5x5 = numpy.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6] , [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256
average3x3 = numpy.ones((3,3))/9
average5x5 = numpy.ones((5,5))/25
sharpen3x3 = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])/9
sharpen5x5 = numpy.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1 ,-1, 25, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])/25
laplacian3x3 = numpy.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/8
laplacian5x5 = numpy.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1 ,-1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])/24


#print("GaussHard5x5:", gaussHard5x5)

#imgNew = convolution3(gaussHard3x3, KernalLen, grayImg, imgXSize, imgYSize)
#imgNew = convolution(average3x3, 3, grayImg, imgXSize, imgYSize)
#imgNew = convolution(average3x3, 3, grayImg, imgXSize, imgYSize)
#imgNew = convolution3(laplacian3x3, 3, grayImg, imgXSize, imgYSize)
#imgNew = convolution(laplacian5x5, 5, grayImg, imgXSize, imgYSize)
imgNew = convolution(gauss5x5, 5, grayImg, imgXSize, imgYSize)


#imgNew = grayImg+imgNew

print(imgNew)
imgNew = imgNew.astype('uint8')
print(numpy.shape(imgNew))
mplplt.imshow(imgNew, cmap='gray')
mplplt.show()

print("Original Gray Image")
print(grayImg[:,:])
print("Filtered Image")
print(imgNew[:,:])

print("Hello World")
