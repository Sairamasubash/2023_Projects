# Here we are importing all the required libraries for this assignment.
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as img

# Here we are defining the variable zero.
zero = 0

# Here we are defining the variable one.
one = 1

# Here we are defining the variable two.
two = 2

# Here we are defining the variable three.
three = 3

# Here we are defining the variable five.
five = 5

# Here we are defining the variable numberOne.
numberOne = 0.00001

# Here we are defining the variable numberTwo.
numberTwo = 10000000

# Here we are defining the variable numberThree.
numberThree = 0.2989

# Here we are defining the variable numberFour.
numberFour = 0.5870

# Here we are defining the variable numberFive.
numberFive = 0.1140

# Here we are creating a function called SSD.
def SSD(pictureOne, pictureTwo):

    # Here we are calculating the sum of squared differences between pictureOne and pictureTwo, raised to the power of two, and returning the result.
    return (np.sum((pictureOne - pictureTwo) ** two))

# Here we are creating a function called SAD.
def SAD(pictureOne, pictureTwo):

    # Here we are calculating the sum of absolute differences between pictureOne and pictureTwo using NumPy's np.sum and np.abs functions.
    return (np.sum(np.abs(pictureOne - pictureTwo)))

# Here we are creating a function called NC.
def NC(pictureOne, pictureTwo):

    # Here we are calculating the normalized cross-correlation between pictureOne and pictureTwo.
    return (one - ((np.sum((pictureOne - np.mean(pictureOne)) * (pictureTwo - np.mean(pictureTwo)))) / ((np.std(pictureOne) * np.std(pictureTwo)) + numberOne)))

# Here we are creating a function called binocularStereo.
def binocularStereo(pictureOne, pictureTwo, searchWindowSize, disparityRange, matchingFunction):

    # Here we are creating a function called binocularStereoHelper.
    def binocularStereoHelper(valueDR, eachOne, eachTwo):

        # Here we are checking if the sum of eachTwo and valueDR is less than zero.
        if eachTwo + valueDR < zero or eachTwo + valueDR >= pictureOne.shape[one] - searchWindowSize:

            return numberTwo   # Here we are returning numberTwo if the condition is met.

        else:   # Here we are checking if the sum of eachTwo and valueDR is not less than zero.

            # Here we are extracting the corresponding region from pictureTwo.
            pictureValueTwo = pictureTwo[eachOne:eachOne + searchWindowSize, eachTwo + valueDR:eachTwo + valueDR + searchWindowSize]

            if matchingFunction == "SSD":   # Here we are checking if the matchingFunction equals SSD.

                # Here we are calling the SSD function.
                return SSD(pictureValueOne, pictureValueTwo)

            elif matchingFunction == "SAD":   # Here we are checking if the matchingFunction equals SAD.

                # Here we are calling the SAD function.
                return SAD(pictureValueOne, pictureValueTwo)

            elif matchingFunction == "NC":   # Here we are checking if the matchingFunction equals NC.

                # Here we are calling the NC function.
                return NC(pictureValueOne, pictureValueTwo)

    # Here we are creating a result matrix filled with zeros, with dimensions based on pictureOne.
    result = np.zeros(shape = (pictureOne.shape[zero], pictureOne.shape[one]))

    # Here we are iterating through the rows of pictureOne with a sliding window of size searchWindowSize.
    for eachOne in range(pictureOne.shape[zero] - searchWindowSize + one):

        # Here we are iterating through the columns of pictureOne with a sliding window of size searchWindowSize.
        for eachTwo in range(pictureOne.shape[one] - searchWindowSize + one):

            # Here we are extracting a sub-image from pictureOne based on the current window position.
            pictureValueOne = pictureOne[eachOne:eachOne + searchWindowSize, eachTwo:eachTwo + searchWindowSize]

            # Here we are checking if the disparity value is not equal to zero, and assign it to the result matrix.
            if abs(((np.argmin((np.array([binocularStereoHelper(eachThree, eachOne, eachTwo) for eachThree in range(int((-disparityRange + one) / two), int((disparityRange + one) / two))])))) + int((-disparityRange + one) / two))) != zero:

                # Here we are calculating the disparity value using binocularStereoHelper function.
                result[eachOne, eachTwo] = five / abs(((np.argmin((np.array([binocularStereoHelper(eachThree, eachOne, eachTwo) for eachThree in range(int((-disparityRange + one) / two), int((disparityRange + one) / two))])))) + int((-disparityRange + one) / two)))

    return result   # Here we are returning the resulting matrix.

imageNumber = one   # Here we are assigning the value two to the variable imageNumber.

if imageNumber == one:   # Here we are checking if imageNumber is equal to one.

    # Here we are reading and storing the image tsukuba1.jpg in the colorOne variable.
    colorOne = img.imread('Data/tsukuba1.jpg')

    # Here we are reading and storing the image tsukuba2.jpg in the colorTwo variable.
    colorTwo = img.imread('Data/tsukuba2.jpg')

elif imageNumber == two:   # Here we are executing this block if imageNumber is not equal to one.

    # Here we are reading and storing the image moebius1.png in the colorOne variable.
    colorOne = img.imread('Data/moebius1.png')

    # Here we are reading and storing the image moebius2.png in the colorTwo variable.
    colorTwo = img.imread('Data/moebius2.png')

_, subplotOne = plt.subplots()   # Here we are creating a subplot named subplotOne.

subplotOne.set_aspect('equal')   # Here we are setting subplot's aspect to equal.

ST = time.time()   # Here we are recording the current time in ST.

# Here we are display an image using the binocularStereo function with specific arguments.
subplotOne.imshow((binocularStereo((numberThree * colorOne[:, :, zero] + numberFour * colorOne[:, :, one] + numberFive * colorOne[:, :, two]), (numberThree * colorTwo[:, :, zero] + numberFour * colorTwo[:, :, one] + numberFive * colorTwo[:, :, two]), searchWindowSize = 15, disparityRange = 25, matchingFunction = "SSD")), cmap='gray')

ET = time.time()   # Here we are recording the current time in ET.

# Here we are printing the runtime by subtracting ST from ET and appending Seconds.
print('Run Time:', ET - ST, 'Seconds')

plt.show()   # Here we are showing the plot.