# Here we are importing all the required libraries for this assignment.
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import ProjectiveTransform
from skimage.transform import warp

# Here we are defining a function called constructTestingArray.
def constructTestingArray(inlierMatches, allLines, testing):

    # Here we are initializing a 2D numpy array of zeros with shape (2 * testing, 9).
    testingArray = np.zeros((2 * testing, 9))

    # Here we are iterating over each line index and its corresponding position in the array.
    for each, one in enumerate(allLines):

        # Here we are extracting the inlier match corresponding to the current line.
        nowLine = inlierMatches[one]

        # Here we are unpacking the four coordinates of the inlier match.
        horizontalOne, verticalOne, horizontalTwo, verticalTwo = nowLine

        # Here we are setting values for the first row of the current line in the testing array.
        testingArray[2 * each] = [0, 0, 0, horizontalOne, verticalOne, 1, -verticalTwo * horizontalOne, -verticalTwo * verticalOne, -verticalTwo]

        # Here we are setting values for the second row of the current line in the testing array.
        testingArray[2 * each + 1] = [horizontalOne, verticalOne, 1, 0, 0, 0, -horizontalTwo * horizontalOne, -horizontalTwo * verticalOne, -horizontalTwo]

    return testingArray   # Here we are returning the constructed testing array.

# Here we are defining a function called calculateResidual.
def calculateResidual(inlierMatches, pairwiseDescriptorMatrix, limit):

    allMatches = len(inlierMatches)   # Here we are calculating the total number of inlier matches.

    allInliers = []   # Here we are initializing a list to store inliers below the residual limit.

    allResiduals = []   # Here we are initializing a list to store residuals of inliers below the limit.

    # Here we are iterating over each inlier match.
    for each in range(allMatches):

        # Here we are multiplying the pairwise descriptor matrix with the inlier match and append a 1 for homogenous coordinates.
        horizontal = np.dot(pairwiseDescriptorMatrix, np.array([inlierMatches[each][0], inlierMatches[each][1], 1]))

        horizontalOne = horizontal[0] / horizontal[2]   # Here we are calculating the x-coordinate in the target space.

        vertical = horizontal[1] / horizontal[2]   # Here we are calculating the y-coordinate in the target space.

        # Here we are extracting the target x and y coordinates from the inlier match.
        horizontalTwo, verticalTwo = inlierMatches[each][2], inlierMatches[each][3]

        # Here we are calculating the residual (Euclidean distance) between transformed source and target coordinates.
        oneResidual = np.sqrt((horizontalTwo - horizontalOne) ** 2 + (verticalTwo - vertical) ** 2)

        # Here we are checking if the residual is below the given limit.
        if oneResidual < limit:

            allInliers.append(each)   # Here we are adding the index of the inlier match to the list.

            allResiduals.append(oneResidual)   # Here we are adding the calculated residual to the list.

    return allInliers, allResiduals   # Here we are returning the lists of inliers and their residuals.

# Here we are defining a function called ransacImplementation.
def ransacImplementation(inlierMatches, numberOfLoops = 0, limit = 0):

    testing = 4   # Here we are initializing testing value to 4.

    testing2 = 4   # Here we are initializing testing2 value to 4.

    number = 1   # Here we are initializing number to 1.

    number2 = 1   # Here we are initializing number2 to 1.

    numberOfInliers = 15   # Here we are initializing numberOfInliers to 15.

    allMatches = inlierMatches.shape[0]   # Here we are getting the total number of inlier matches.

    # Here we are running the loop until the iteration counter is less than numberOfLoops.
    while number < numberOfLoops:

        # Here we are checking if the testing value is 4.
        if testing == testing2:

            # Here we are randomly selecting 'testing' lines from all matches.
            selectedLines = np.random.choice(np.arange(allMatches), size = testing)

        # Here we are constructing a testing array using the selected lines.
        testingArray = constructTestingArray(inlierMatches, selectedLines, testing)

        # Here we are performing singular value decomposition (SVD) on the testing array.
        U, s, V = np.linalg.svd(testingArray)

        # Here we are reshaping the last row of V to a 3x3 matrix.
        pairwiseDescriptorMatrix = np.reshape(V[-1], (3, 3))

        # Here we are calculating inliers and residuals using the pairwise descriptor matrix.
        allInliers, allResiduals = calculateResidual(inlierMatches, pairwiseDescriptorMatrix, limit)

        # Here we are checking if the number of inliers is numberOfInliers or more.
        if len(allInliers) >= numberOfInliers:

            testing = len(allInliers)   # Here we are updating testing value to the number of inliers.

            number += number2   # Here we are incrementing iteration counter.

        else:   # Here we are checking if the number of inliers is less than 15.

            testing = testing2   # Here we are resetting testing value to 4.

    averageOfResidual = np.mean(allResiduals)   # Here we are calculating the average of all residuals.

    # Here we are returning inliers, number of inliers, average residual, and the pairwise descriptor matrix.
    return allInliers, len(allInliers), averageOfResidual, pairwiseDescriptorMatrix

# Here we are defining a function called descriptorNormalizing.
def descriptorNormalizing(picture):

    pictureMin = np.min(picture)   # Here we are finding the minimum value in the picture array.

    pictureMax = np.max(picture)   # Here we are finding the maximum value in the picture array.

    # Here we are normalizing the picture values to a range of [0, 1].
    normalized_picture = (picture - pictureMin) / (pictureMax - pictureMin)

    return normalized_picture   # Here we are returning the normalized picture array.

# Here we are defining a function called grayscale.
def grayscale(*image_paths):

    # Here we are reading each image using the provided paths.
    images = [cv2.imread(path) for path in image_paths]

    # Here we are converting each image to grayscale.
    grayscales = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    return (*grayscales, *images)   # Here we are returning both grayscale images and original images.

# Here we are defining a function called computeDescriptors.
def computeDescriptors(picture):

    # Here we are creating a SIFT detector and compute keypoints and descriptors for the given image.
    return cv2.xfeatures2d.SIFT_create().detectAndCompute(picture, None)

# Here we are defining a function called computeDistances.
def computeDistances(leftTwo, rightTwo, limit):

    # Here we are calculating the squared Euclidean distance between the descriptors and return the indices of pairs with distance less than the given limit.
    return list(np.argwhere(scipy.spatial.distance.cdist(leftTwo, rightTwo, 'sqeuclidean') < limit))

# Here we are defining a function called get_harris_points.
def get_harris_points(leftOne, rightOne, limitActivation):

    # Here we are extracting the respective keypoints' coordinates for each pair in the activation list.
    return np.array([np.array([rightOne[each[1]].pt[0], rightOne[each[1]].pt[1], leftOne[each[0]].pt[0], leftOne[each[0]].pt[1]]) for each in limitActivation])

# Here we are defining a function called plot_inlier_matches.
def plot_inlier_matches(one, two, three, picture, inlierMatches, offsetOne, inlierMatchesTwo, offsetTwo, marker, markesize, title, both, offsetThree, offsetFour, vertical, bothTwo, markerTwo, markesizeTwo):

    # Here we are checking if the first condition is True.
    if one == True:

        plt.imshow(picture, cmap='gray')   # Here we are displaying the 'picture' in grayscale.

        # Here we are checking if the nested condition 'bothTwo' is True.
        if bothTwo == True:

            # Here we are iterating over each match in 'inlierMatchesTwo'.
            for each in inlierMatchesTwo:

                # Here we are plotting lines between corresponding match points.
                plt.plot([inlierMatches[each][offsetOne], inlierMatches[each][offsetThree] + vertical], [inlierMatches[each][offsetTwo], inlierMatches[each][offsetFour]], markerTwo, markersize = markesizeTwo)

        # Here we are plotting scatter points for matches based on offsets One and Two.
        plt.plot([inlierMatches[each][offsetOne] for each in inlierMatchesTwo], [inlierMatches[each][offsetTwo] for each in inlierMatchesTwo], marker, markersize = markesize)

        if both == True:   # Here we are checking if the nested condition 'both' is True.

            # Here we are plotting scatter points for matches based on offsets Three and Four, adjusted by 'vertical'.
            plt.plot([inlierMatches[each][offsetThree] + vertical for each in inlierMatchesTwo], [inlierMatches[each][offsetFour] for each in inlierMatchesTwo], marker, markersize=markesize)

        plt.title(title)   # Here we are setting the title of the plot.

        plt.axis('off')   # Here we are turning off the axis.

        plt.show()   # Here we are displaying the plot.

    if two == True:   # Here we are checking if the second condition is True.

        # Here we are displaying the sum of 'inlierMatches' and 'picture' in grayscale.
        plt.imshow(inlierMatches + picture, cmap='gray')

        plt.show()   # Here we are displaying the plot.

    if three == True:   # Here we are checking if the third condition is True.

        plt.imshow(picture, cmap='gray')   # Here we are displaying the 'picture' in grayscale.

        plt.show()   # Here we are displaying the plot.

# Here we are defining a function called stitching.
def stitching(grayScaleL, grayScaleR, pairwiseDescriptorMatrix):

    # Here we are concatenating transformed corner points of both images using the pairwise descriptor matrix.
    allParts = np.concatenate((stitching2(grayScaleL.shape), cv2.perspectiveTransform(stitching2(grayScaleR.shape), pairwiseDescriptorMatrix)), axis = 0)

    # Here we are calculating the translation needed to make sure the resulting stitched image fits the canvas.
    translation = [-np.int32(allParts.min(axis = 0).ravel() - 0.5)[0], -np.int32(allParts.min(axis = 0).ravel() - 0.5)[1]]

    # Here we are warping the right image using the translation and pairwise descriptor matrix to create the base for the final stitched image.
    final = cv2.warpPerspective(grayScaleR, np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]]).dot(pairwiseDescriptorMatrix), (np.int32(allParts.max(axis = 0).ravel() + 0.5)[0] - np.int32(allParts.min(axis = 0).ravel() - 0.5)[0], np.int32(allParts.max(axis = 0).ravel() + 0.5)[1] - np.int32(allParts.min(axis = 0).ravel() - 0.5)[1]))

    # Here we are overlaying the left grayscale image onto the final stitched image.
    final[translation[1]:grayScaleL.shape[0] + translation[1], translation[0]:grayScaleL.shape[1] + translation[0]] = grayScaleL

    return final   # Here we are returning the final stitched image.

# Here we are defining a function called stitching2.
def stitching2(shape):

    # Here we are extracting the horizontal and vertical dimensions of the shape.
    horizontal, vertical = shape[:2]

    # Here we are returning corner points of the shape in a specific format.
    return np.float32([[0, 0], [0, horizontal], [vertical, horizontal], [vertical, 0]]).reshape(-1, 1, 2)

# Here we are defining a function called colorPanorama.
def colorPanorama(final):

    horizontal3, vertical3 = np.shape(final)   # Here we are getting dimensions of the 'final' array.

    # Here we are initializing a zero array of same dimensions as 'final' but with an additional dimension for RGB channels.
    finalTwo = np.zeros((horizontal3, vertical3, 3))

    # Here we are looping over RGB channels, and stitch corresponding channels of 'leftImage' and 'rightImage'.
    for each, stitch in enumerate([2, 1, 0]):

        # Here we are doing this using the 'stitching' function.
        finalTwo[:, :, each] = stitching(leftImage[:, :, stitch], rightImage[:, :, stitch], pairwiseDescriptorMatrix)

    plt.imshow(descriptorNormalizing(finalTwo))   # Here we are displaying the stitched and normalized image.

    plt.show()   # Here we are showing the plotted image.

# Here we are setting the default figure size to 17 inches width and 10 inches height.
plt.rcParams['figure.figsize'] = (17, 10)

# Here we are loading grayscale images and original images from specified paths.
grayScaleL, grayScaleR, leftImage, rightImage = grayscale('Images/left.jpg', 'Images/right.jpg')

# Here we are computing SIFT keypoints and descriptors for the left grayscale images.
leftOne, leftTwo = computeDescriptors(grayScaleL)

# Here we are computing SIFT keypoints and descriptors for the right grayscale images.
rightOne, rightTwo = computeDescriptors(grayScaleR)

threshold = 10001   # Here we are creating a variable to hold the threshold value.

# Here we are computing and store pairs of descriptors that have a distance less than the given threshold.
limitActivation = computeDistances(leftTwo, rightTwo, threshold)

# Here we are extracting and storing the Harris corner points based on previously computed descriptor distances.
inlierMatches = get_harris_points(leftOne, rightOne, limitActivation)

# Here we are applying RANSAC algorithm to the inlier matches to identify robust point correspondences, estimated transformation model, etc.
inlierMatchesTwo, allLinesThree, averageOfResidual, pairwiseDescriptorMatrix = ransacImplementation(inlierMatches, numberOfLoops = 100, limit = 0.66)

# Here we are estimating the projective transformation matrix based on the pairwise descriptor matrix.
estimatedTransformation = ProjectiveTransform(pairwiseDescriptorMatrix)

# Here we are warping the left grayscale image using the estimated projective transformation.
warpImage = warp(grayScaleL, estimatedTransformation)

# Here we are calling the plot_inlier_matches function for plotting inlier matches in the left image.
plot_inlier_matches(True, 0, 0, grayScaleL, inlierMatches, 2, inlierMatchesTwo, 3, 'bx', 3, 'Inlier matches in the left image', False, 0, 0, 0, False, 0, 0)

# Here we are calling the plot_inlier_matches function for plotting inlier matches in the right image.
plot_inlier_matches(True, 0, 0, grayScaleR, inlierMatches, 0, inlierMatchesTwo, 1, 'bx', 3, 'Inlier matches in the right image', False, 0, 0, 0, False, 0, 0)

# Here we are retrieving the dimensions (height, width) of the 'grayScaleL' image.
horizontal, vertical = np.shape(grayScaleL)

# Here we are concatenating 'grayScaleL' and 'grayScaleR' horizontally to form 'imageMovementOne'.
imageMovementOne = np.hstack((grayScaleL, grayScaleR))

# Here we are setting the default figure size to 26 inches width and 21 inches height.
plt.rcParams['figure.figsize'] = (26, 21)

# Here we are calling the plot_inlier_matches function for plotting inlier matches in both images.
plot_inlier_matches(True, 0, 0, imageMovementOne, inlierMatches, 2, inlierMatchesTwo, 3, 'bx', 3, 'Inlier matches in both images', True, 0, 1, vertical, False, 0, 0)

# Here we are retrieving the dimensions (height, width) of the 'grayScaleL' image.
horizontal1, vertical1 = np.shape(grayScaleL)

# Here we are concatenating 'grayScaleL' and 'grayScaleR' horizontally to form 'imageMovementTwo'.
imageMovementTwo = np.hstack((grayScaleL, grayScaleR))

# Here we are setting the default figure size to 26 inches width and 21 inches height.
plt.rcParams['figure.figsize'] = (26, 21)

# Here we are calling the plot_inlier_matches function for plotting inlier matches in both images with lines.
plot_inlier_matches(True, 0, 0, imageMovementTwo, inlierMatches, 2, inlierMatchesTwo, 3, 'bx', 3, 'Inlier matches in both images with lines', True, 0, 1, vertical1, True, 'y-', 1)

# Here we are applying descriptor normalization to 'grayScaleR' and store the result in 'imageMovementThree'.
imageMovementThree = descriptorNormalizing(grayScaleR)

# Here we are setting all the pixels in 'imageMovementThree' to 0 where 'warpImage' is not equal to 0.
imageMovementThree[warpImage != 0] = 0

# Here we are getting the shape (dimensions) of the grayScaleL image, with horizontal2 as width and vertical2 as height.
horizontal2, vertical2 = grayScaleL.shape

# Here we are creating a 3D meshgrid of indices for the image and reshape it into a 3xN matrix, where N is the total number of pixels.
eachArray = np.array(np.meshgrid(np.arange(horizontal2), np.arange(vertical2), [1])).reshape(3, -1)

# Here we are multiplying the pairwiseDescriptorMatrix with eachArray to obtain a transformed set of points.
newEach = np.dot(pairwiseDescriptorMatrix, eachArray)

# Here we are homogenizing the coordinates by dividing the first two rows by the third row.
newEach[:2, :] /= newEach[2, :]

# Here we are extracting the maximum and minimum horizontal and vertical coordinates from the transformed points.
horizontalPlus, horizontalMinus, verticalPlus, verticalMinus = np.max(newEach[0, :]), np.min(newEach[0, :]), np.max(newEach[1, :]), np.min(newEach[1, :])

# Here we are merging two grayscale images using their descriptor matrix.
final = stitching(grayScaleL, grayScaleR, pairwiseDescriptorMatrix)

# Here we are calling the plot_inlier_matches function for shows the stitched image.
plot_inlier_matches(0, 0, True, final, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

colorPanorama(final)   # Here we are calling the colorPanorama function.

# Here we are printing the inliers for homography with a message.
print("Here is the homography inliers: " + str(allLinesThree))

# Here we are printing the average residual for inliers with a message.
print("Here is the average residual for the inliers: " + str(averageOfResidual))