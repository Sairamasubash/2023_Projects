# Here we are importing all the required libraries for this assignment.
import numpy as np
import cv2
from scipy.ndimage import gaussian_laplace
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Here we are defining a function to display detected circles and their orientations on the provided image.
def show_all_circles(image, cx, cy, rad, commonOrientation, circle_color = 'r'):
    """
     image: numpy array, representing the grayscsale image
     cx, cy: numpy arrays or lists, centers of the detected blobs
     rad: numpy array or list, radius of the detected blobs
     """

    figure, axis = plt.subplots()   # Here we are creating a new figure and axis for displaying the image.

    axis.set_aspect('equal')   # Here we are setting aspect ratio to 'equal' to avoid distortion of the circles.

    axis.imshow(image, cmap = 'gray')   # Here we are displaying the grayscale image.

    # Here we are looping through each circle's properties and plot them.
    for one, two, three, common in zip(cx, cy, rad, commonOrientation):

        # Here we are marking the center of the circle with an 'x'.
        axis.scatter(one, two, color = 'b', marker = 'x', s = 25)

        # Here we are drawing the circle around the blob.
        markCircle = Circle((one, two), three * 10, color = circle_color, fill = False, linewidth = 1)

        axis.add_patch(markCircle)   # Here we are adding this marking to the axis.

        oneEnd = int(one + 10 * three * np.cos(common))   # Here we are calculating first end point for the orientation arrow.

        twoEnd = int(two + 10 * three * np.sin(common))   # Here we are calculating second end point for the orientation arrow.

        # Here we are drawing an arrow indicating the orientation.
        markArrow = axis.arrow(one, two, oneEnd - one, twoEnd - two, head_width = 10, head_length = 10, fc = 'g', ec = 'g')

        axis.add_patch(markArrow)   # Here we are adding this marking to the axis.

    # Here we are adding title indicating the number of circles detected.
    plt.title('There are %i Circles In The Image Below' % len(cx))

    plt.show()   # Here we are displaying the image with marked circles.

# Here we are defining a function to convert an input picture to a normalized grayscale image.
def grayscale(picture):

    image = cv2.imread(picture)   # Here we are reading the input picture using OpenCV.

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # Here we are converting the image to grayscale.

    # Here we are normalizing the grayscale image values to the range [0,1].
    normalizeImage = grayImage.astype(np.float32) / 255.0

    return normalizeImage   # Here we are returning the normalized grayscale image.

# Here we are defining a function to detect corners in the given picture using the harris corners method.
def detectedCorners(picture, limitCorners=50, qualityOfCorners=0.01, limitDistance=10):

    # Here we are using OpenCV's goodFeaturesToTrack to detect corners, then convert the resulting corners to integer points.
    result = np.intp(cv2.goodFeaturesToTrack(picture, limitCorners, qualityOfCorners, limitDistance))

    return result   # Here we are returning the detected corners as a list of integer points.

# Here we are defining a function to apply the Laplacian of Gaussian (LoG) to an image at different scales and return the responses.
def logApplication(picture, startPoint = 2, endPoint = 10, kSize = 1.25):

    numberOfResponses = []   # Here we are using a list to store responses at each scale.

    # Here we are computing scales based on the starting point, end point, and scale factor.
    allScales = [startPoint * (kSize ** each) for each in range(endPoint)]

    # Here we are looping through each scale and compute the LoG response.
    for sigma in allScales:

        # Here we are downscaling the image by the scale factor.
        downscaled_picture = resize(picture, (int(picture.shape[0] / kSize), int(picture.shape[1] / kSize)), anti_aliasing = True)

        # Here we are computing the LoG response at the current scale and resize it to the original image size.
        eachResponse = resize(np.abs(gaussian_laplace(downscaled_picture, sigma = sigma)), picture.shape, mode = 'reflect', anti_aliasing = True)

        numberOfResponses.append(eachResponse)   # Here we are appending the computed response to the list.

    return numberOfResponses, allScales   # Here we are returning the list of responses and the scales used.

# Here we are defining a function to determine scales associated with each detected corner.
def cornerScales(picture, cornersDetected, numberOfResponses, allScales):

    # Here we are defining a function to determine the scale associated with a given corner.
    def get_scale(corner):

        horizontal, vertical = corner.ravel()   # Here we are extracting x and y coordinates from the corner point.

        # Here we are getting the scale corresponding to the maximum response for the given corner point.
        resultOne = allScales[np.argmax([resp[vertical, horizontal] for resp in numberOfResponses])]

        return resultOne   # Here we are returning the value of resultOne.

    # Here we are applying get_scale function for each detected corner to determine their associated scales.
    resultTwo = [get_scale(corner) for corner in cornersDetected]

    return resultTwo   # Here we are returning the value of resultTwo.

# Here we are defining a function that computes the most common gradient orientation in a given window around a pixel.
def computeCommonOrientation(picture, horizontal, vertical, allScales):

    gradientVertical, gradientHorizontal = np.gradient(picture)   # Here we are computing vertical and horizontal gradients of the picture using numpy.

    numberOfScales = int(allScales)   # Here we are converting the number of scales from float to integer.

    # Here we are computing the arctangent of the ratio of vertical to horizontal gradient for a window around the pixel.
    windows = np.arctan2(gradientVertical[vertical - numberOfScales:vertical + numberOfScales + 1, horizontal - numberOfScales:horizontal + numberOfScales + 1], gradientHorizontal[vertical - numberOfScales:vertical + numberOfScales + 1, horizontal - numberOfScales:horizontal + numberOfScales + 1])

    oneD = windows.ravel()   # Here we are flattening the windows to create a 1D array.

    # Here we are creating a histogram with 36 bins ranging from -π to π.
    histogram, numberOfBins = np.histogram(oneD, bins = 36, range = (-np.pi, np.pi))

    # Here we are finding the bin with the maximum count and compute its center orientation.
    computedCommonOrientation = numberOfBins[np.argmax(histogram)] + (numberOfBins[1] - numberOfBins[0]) / 2

    return computedCommonOrientation   # Here we are returning the computed common orientation.

# Here we are defining a function to get the new picture we are creating.
def newPicture(picture):

    allCorners = detectedCorners(picture)   # Here we are detecting all corners in the given picture.

    # Here we are applying Laplacian of Gaussian (LoG) to the picture and get the number of responses and all scales.
    numberOfResponses, allScales = logApplication(picture)

    # Here we are calculating scales for each corner detected in the picture.
    allCornerScales = cornerScales(picture, allCorners, numberOfResponses, allScales)

    # Here we are extracting horizontal and vertical coordinates from all detected corners.
    horizontal, vertical = zip(*[(corner[0][0], corner[0][1]) for corner in allCorners])

    # Here we are computing the common orientation for each corner using its coordinates and scale.
    allOrientations = [computeCommonOrientation(picture, h, v, s) for h, v, s in zip(horizontal, vertical, allCornerScales)]

    # Here we are displaying the picture with all detected corners marked by circles with their corresponding scales and orientations.
    show_all_circles(picture, horizontal, vertical, allCornerScales, allOrientations)

# Here we are defining a function that shifts the image left or right by a given percentage.
def shiftImageLR(picture, shiftPercent = 0.2, move = 'left'):

    _, c = picture.shape   # Here we are extracting the number of columns (width) from the picture's shape.

    shifting = int(c * shiftPercent)   # Here we are calculating the number of columns to shift based on the percentage.

    if move == 'left':   # Here we are checking if the shift direction is to the left.

        return picture[:, shifting:]   # Here we are returning the image excluding the first 'shifting' columns.

    if move == 'right':   # Here we are checking if the shift direction is to the right.

        return picture[:, :-shifting]   # Here we are returning the image excluding the last 'shifting' columns.

    return picture   # Here we are returning the value of the picture.

# Here we are defining a function that rotates an image clockwise or counter-clockwise based on the provided angle.
def rotateImageCWCCW(picture, rotation=90):

    r, c = picture.shape[:2]   # Here we are getting the number of rows (height) and columns (width) of the image.

    center = (c // 2, r // 2)   # Here we are calculating the center of the image for rotation purposes.

    # Here we are obtaining the rotation matrix for the specified angle.
    moving = cv2.getRotationMatrix2D(center, rotation, 1)

    # Here we are applying the rotation to the image using the rotation matrix.
    result = cv2.warpAffine(picture, moving, (c, r))

    return result   # Here we are returning the rotated image.

# Here we are defining a function to enlarge the given picture by the specified factor and then crop it to the original size.
def enlargedImage(picture, factor = 2.0):

    # Here we are calculating the new height and width based on the enlargement factor.
    enlargedHeight, enlargedWidth = int(picture.shape[0] * factor), int(picture.shape[1] * factor)

    # Here we are resizing the image to the enlarged dimensions with anti-aliasing.
    enlargedFactor = resize(picture, (enlargedHeight, enlargedWidth), anti_aliasing=True)

    # Here we are calculating vertical offset to center the enlarged image.
    verticalOffset = (enlargedHeight - picture.shape[0]) // 2

    # Here we are calculating horizontal offset to center the enlarged image.
    horizontalOffset = (enlargedWidth - picture.shape[1]) // 2

    # Here we are cropping the enlarged image back to the original size centered on the enlarged version.
    result = enlargedFactor[verticalOffset:verticalOffset + picture.shape[0], horizontalOffset:horizontalOffset + picture.shape[1]]

    return result   # Here we are returning the cropped, enlarged image.

# Here we are defining a function which will bring all the other code together.
def final(picture):

    image = grayscale(picture)   # Here we are converting the input picture to grayscale.

    # Here is the listing of tuples containing transformation name and corresponding transformed image.
    transformations = [

        ('original', image),   # Here we are printing the original image.

        ('shiftLeftCrop', shiftImageLR(image, move = 'left')),   # Here we are shifting image to the left.

        ('shiftRightCrop', shiftImageLR(image, move = 'right')),   # Here we are shifting image to the right.

        # Here we are rotating image 90 degrees counter-clockwise.
        ('rotateCCW', rotateImageCWCCW(image, rotation = 90)),

        # Here we are rotating image 90 degrees clockwise.
        ('rotateCW', rotateImageCWCCW(image, rotation = -90)),

        # Here we are enlarging the image by a factor of 2.0.
        ('enlarged', enlargedImage(image, factor = 2.0))
    ]

    # Here we are looping through each transformation and process the transformed image.
    for _, transformed_image in transformations:

        newPicture(transformed_image)   # Here we are calling the function called newPicture.

final('InitialImages/WINDOW.jpg')   # Here we are calling the final function for the first image.

final('InitialImages/HOTEL.jpg')   # Here we are calling the final function for the second image.

final('InitialImages/TOY.jpg')   # Here we are calling the final function for the third image.

final('InitialImages/ROSE.jpg')   # Here we are calling the final function for the fourth image.