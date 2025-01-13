# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
## Fundamental Matrix Estimation

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

##
## load images and match files for the first example
##

I1 = Image.open('MP4_part2_data/library1.jpg');
I2 = Image.open('MP4_part2_data/library2.jpg');
matches = np.loadtxt('MP4_part2_data/library_matches.txt');

# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image:
# matches(i,1:2) is a point in the first image
# matches(i,3:4) is a corresponding point in the second image

N = len(matches)

##
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
##

I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
I3[:,:I1.size[0],:] = I1;
I3[:,I1.size[0]:,:] = I2;
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I3).astype(int))
ax.plot(matches[:,0],matches[:,1],  '+r')
ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
plt.show()

# Here we are defining the variable zero.
zero = 0

# Here we are defining the variable one.
one = 1

# Here we are defining the variable two.
two = 2

# Here we are defining the variable three.
three = 3

# Here we are defining the variable four.
four = 4

# Here we are defining the variable eight.
eight = 8

# Here we are defining the variable twelve.
twelve = 12

# Here we are defining the variable numberOne.
numberOne = 100

# Here we are defining the variable numberTwo.
numberTwo = 15000

# Here we are defining the variable numberThree.
numberThree = 0.5

# Here we are creating a function called fit_fundamental.
def fit_fundamental(p1, p2 = True):

    # Here we are checking if the parameter 'p2' is True, to execute the normalization process.
    if p2:

        # Here we are calculating the mean of the points p1 along the axis 0 (column-wise mean).
        average = np.mean(p1, axis = zero)

        p1Average = p1 - average   # Here we are subtracting the mean from each point in p1 to normalize the data.

        # Here we are calculating the sum of squares of the normalized points, separately for the first two and last columns.
        addition = np.sum(p1Average[:, :two] ** two, axis = one) + np.sum(p1Average[:, two:] ** two, axis = one)

        # Here we are calculating the scaling factor for the first two columns.
        divisionPartOne = np.sqrt(np.sum(addition[:two]) / (two * len(p1)))

        # Here we are calculating the scaling factor for the last column.
        divisionPartTwo = np.sqrt(np.sum(addition[two:]) / (two * len(p1)))

        # Here we are normalizing the first two columns of p1Average with divisionPartOne.
        p1Average[:, :two] /= divisionPartOne

        # Here we are normalizing the last column of p1Average with divisionPartTwo.
        p1Average[:, two:] /= divisionPartTwo

        # Here we are creating a diagonal matrix to un-normalize the first two columns.
        resultOne = np.diag([one / divisionPartOne, one / divisionPartOne, one])

        # Here we are updating the last column of resultOne with the un-normalized averages of the first two columns.
        resultOne[zero:two, two] = -average[:two] / divisionPartOne

        # Here we are creating a diagonal matrix to un-normalize the last column.
        resultTwo = np.diag([one / divisionPartTwo, one / divisionPartTwo, one])

        # Here we are updating the last column of resultTwo with the un-normalized average of the last column.
        resultTwo[zero:two, two] = -average[two:] / divisionPartTwo

        p1 = p1Average   # Here we are assigning the normalized points to p1.

        # Here we are setting partTwo to the un-normalization matrix for the first two columns.
        partTwo = resultOne

        # Here we are setting partThree to the un-normalization matrix for the last column.
        partThree = resultTwo

    else:   # Here we are checking if p2 is False, do not normalize and set the un-normalization matrices to None.

        # Here we are setting both partTwo and partThree to None indicating no normalization matrices are created.
        partTwo = partThree = None

    # Here we are constructing a matrix from combinations of points' coordinates, for fundamental matrix computation.
    newN = np.array([[h1 * h2, v1 * h2, h2, h1 * v2, v1 * v2, v2, h1, v1, one] for h1, v1, h2, v2 in p1[:, :four]])

    # Here we are performing Singular Value Decomposition (SVD) and reshape the last row of V^T to a 3x3 matrix.
    valueOne = np.linalg.svd(newN)[two][-one].reshape(three, three)

    # Here we are decomposing the matrix from the previous step using SVD for further refinement.
    U, s, V = np.linalg.svd(valueOne)

    # Here we are forcing the smallest singular value to zero to impose rank 2 constraint on the fundamental matrix.
    s[-one] = zero

    # Here we are reconstructing the matrix with the modified singular values to get the fundamental matrix.
    result = U @ np.diag(s) @ V

    # Here we are checking if the second argument is true, adjust the fundamental matrix with the helper matrices; otherwise, return the result as is.
    return np.dot(partThree.T, result).dot(partTwo) if p2 else result

##
## display second image with epipolar lines reprojected
## from the first image
##

# first, fit fundamental matrix to the matches
F = fit_fundamental(matches, False); # this is a function that you should write
M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
L1 = np.matmul(F, M).transpose() # transform points from
# the first image to get epipolar lines in the second image

# find points on epipolar lines L closest to matches(:,3:4)
l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

# Here we are calculating the mean absolute distance of points from a line, store in residual.
finalR = np.mean(np.abs(pt_line_dist))

# Here we are setting the flag normalizationValue to False, indicating no normalization is applied.
normalizationValue = False

# Here we are outputting the residual and normalization values formatted in a string to the console.
print('\nResidual Value = {}, Normalization Value = {}'.format(finalR, normalizationValue))

# find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

# display points and segments of corresponding epipolar lines
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I2).astype(int))
ax.plot(matches[:,2],matches[:,3],  '+r')
ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')
plt.show()

## Camera Calibration
def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

# Here we are loading the text data from 'lab_matches.txt' into an array.
textOne = np.loadtxt('MP4_part2_data/lab_matches.txt')

# Here we are loading the text data from 'lab_3d.txt' into an array.
textTwo = np.loadtxt('MP4_part2_data/lab_3d.txt')

textResultOne = textOne[:, :two]   # Here we are extracting the first two columns from 'textOne'.

textResultTwo = textOne[:, two:]   # Here we are extracting the last two columns from 'textOne'.

# Here we are initializing a zero matrix with the same number of rows as 'textTwo' and 12 columns.
valueOne = np.zeros((len(textTwo), twelve))

# Here we are assigning to 'valueOne' the values from 'textTwo' with an added column of ones for even rows in columns 4 to 7.
valueOne[(np.arange(zero, (len(textTwo) // two) * two, two)), four:eight] = valueOne[(np.arange(one, (len(textTwo) // two) * two, two)), zero : four] = (np.hstack((textTwo[:(len(textTwo) // two)], np.ones(((len(textTwo) // two), one)))))

# Here we are assigning a scaled version of 'textTwo' with an added column of ones to even rows in columns 8 to 11, scaled by -textResultOne's second column.
valueOne[(np.arange(zero, (len(textTwo) // two) * two, two)), eight:twelve] = -textResultOne[:(len(textTwo) // two), one, None] * (np.hstack((textTwo[:(len(textTwo) // two)], np.ones(((len(textTwo) // two), one)))))

# Here we are assigning a scaled version of 'textTwo' with an added column of ones to odd rows in columns 8 to 11, scaled by -textResultOne's first column.
valueOne[(np.arange(one, (len(textTwo) // two) * two, two)), eight:twelve] = -textResultOne[:(len(textTwo) // two), zero, None] * (np.hstack((textTwo[:(len(textTwo) // two)], np.ones(((len(textTwo) // two), one)))))

# Here we are performing Singular Value Decomposition on 'valueOne' and unpacks the results into matrices _, _, and V.
_, _, V = np.linalg.svd(valueOne)

# Here we are reshaping the last row of V to a 3x4 matrix to obtain the first camera matrix.
cameraMatrixOne = V[-one].reshape((three, four))

# Here we are re-initializing 'valueOne' to a zero matrix with dimensions based on 'textTwo' length and a width of 12.
valueOne = np.zeros((len(textTwo), twelve))

# Here we are assigning to 'valueOne' the values from 'textTwo' with an added column of ones for even rows in columns 4 to 7.
valueOne[(np.arange(zero, (len(textTwo) // two) * two, two)), four:eight] = valueOne[(np.arange(one, (len(textTwo) // two) * two, two)), zero : four] = (np.hstack((textTwo[:(len(textTwo) // two)], np.ones(((len(textTwo) // two), one)))))

# Here we are assigning a scaled version of 'textTwo' with an added column of ones to even rows in columns 8 to 11, scaled by -textResultOne's second column.
valueOne[(np.arange(zero, (len(textTwo) // two) * two, two)), eight:twelve] = -textResultTwo[:(len(textTwo) // two), one, None] * (np.hstack((textTwo[:(len(textTwo) // two)], np.ones(((len(textTwo) // two), one)))))

# Here we are assigning a scaled version of 'textTwo' with an added column of ones to odd rows in columns 8 to 11, scaled by -textResultOne's first column.
valueOne[(np.arange(one, (len(textTwo) // two) * two, two)), eight:twelve] = -textResultTwo[:(len(textTwo) // two), zero, None] * (np.hstack((textTwo[:(len(textTwo) // two)], np.ones(((len(textTwo) // two), one)))))

# Here we are performing Singular Value Decomposition on 'valueOne' and unpacks the results into matrices _, _, and V.
_, _, V = np.linalg.svd(valueOne)

# Here we are reshaping the last row of V to a 3x4 matrix to obtain the second camera matrix.
cameraMatrixTwo = V[-one].reshape((three, four))

# Here we are computing and storing the residuals for the first camera matrix after evaluating the points.
_, cameraMatrixOneResidual = evaluate_points(cameraMatrixOne, textResultOne, textTwo)

# Here we are computing and storing the residuals for the second camera matrix after evaluating the points.
_, cameraMatrixTwoResidual = evaluate_points(cameraMatrixTwo, textResultTwo, textTwo)

# Here we are printing the first camera matrix.
print('\nCamera Matrix One = \n{}'.format(cameraMatrixOne))

# Here we are printing the second camera matrix.
print('\nCamera Matrix Two = \n{}'.format(cameraMatrixTwo))

# Here we are printing the residual value for the first camera matrix.
print('\nCamera Matrix One Residual = {}'.format(cameraMatrixOneResidual))

# Here we are printing the residual value for the second camera matrix.
print('\nCamera Matrix Two Residual = {}'.format(cameraMatrixTwoResidual))

## Camera Centers

# Here we are loading camera matrix data from a text file for library 1.
textThree = np.loadtxt('MP4_part2_data/library1_camera.txt')

# Here we are loading camera matrix data from a text file for library 2.
textFour = np.loadtxt('MP4_part2_data/library2_camera.txt')

# Here we are duplicating cameraMatrixOne for further independent processing.
cameraMatrixOne1 = cameraMatrixOne

# Here we are duplicating cameraMatrixTwo for further independent processing.
cameraMatrixOne2 = cameraMatrixTwo

# Here we are calculating camera center for lab camera one from SVD (Singular Value Decomposition) of the camera matrix.
labCameraCenterOne = (finalResult := np.linalg.svd(cameraMatrixOne1)[-one].T[:, -one]) / finalResult[-one]

# Here we are calculating camera center for lab camera two from SVD of the camera matrix.
labCameraCenterTwo = (finalResult := np.linalg.svd(cameraMatrixOne2)[-one].T[:, -one]) / finalResult[-one]

# Here we are calculating camera center for library camera one from SVD of the camera matrix loaded from text.
libraryCameraCenterOne = (finalResult := np.linalg.svd(textThree)[-one].T[:, -one]) / finalResult[-one]

# Here we are calculating camera center for library camera two from SVD of the camera matrix loaded from text.
libraryCameraCenterTwo = (finalResult := np.linalg.svd(textFour)[-one].T[:, -one]) / finalResult[-one]

# Here we are printing out the calculated camera center for lab camera one.
print("\nLab Camera Center One = {}".format(labCameraCenterOne))

# Here we are printing out the calculated camera center for lab camera two.
print("\nLab Camera Center Two = {}".format(labCameraCenterTwo))

# Here we are printing out the calculated camera center for library camera one.
print("\nLibrary Camera Center One = {}".format(libraryCameraCenterOne))

# Here we are printing out the calculated camera center for library camera two.
print("\nLibrary Camera Center Two = {}".format(libraryCameraCenterTwo))

## Triangulation

# Here we are loading lab matches data from text file into numpy array.
textFive = np.loadtxt('MP4_part2_data/lab_matches.txt')

# Here we are loading library matches data from text file into numpy array.
textSix = np.loadtxt('MP4_part2_data/library_matches.txt')

# Here we are appending a column of ones to the first two columns of lab data.
labCTOne = np.hstack((textFive[:, : two], (np.ones((len(textFive), one)))))

# Here we are appending a column of ones to the first two columns of library data.
libraryCTOne = np.hstack((textSix[:, : two], (np.ones((len(textSix), one)))))

# Here we are appending a column of ones to the last two columns of lab data.
labCTTwo = np.hstack((textFive[:, two:], (np.ones((len(textFive), one)))))

# Here we are appending a column of ones to the last two columns of library data.
libraryCTTwo = np.hstack((textSix[:, two:], (np.ones((len(textSix), one)))))

# Here we are initializing a 3D points array for lab with zeros.
labCT3D = np.zeros((len(textFive), four))

# Here we are initializing a 3D points array for library with zeros.
libraryCT3D = np.zeros((len(textSix), four))

# Here we are iterating over each pair of lab matches.
for each in range(len(textFive)):

    # Here we are performing Singular Value Decomposition (SVD) for the stacked matrices after multiplication with camera matrices.
    _, _, V = np.linalg.svd((np.vstack((((np.array([[zero, -labCTOne[each, two], labCTOne[each, one]], [labCTOne[each, two], zero, -labCTOne[each, zero]], [-labCTOne[each, one], labCTOne[each, zero], zero]])).dot(cameraMatrixOne1)),

                                        # Here we are creating two skew-symmetric matrices (one above and one here).
                                        ((np.array([[zero, -labCTTwo[each, two], labCTTwo[each, one]], [labCTTwo[each, two], zero, -labCTTwo[each, zero]], [-labCTTwo[each, one], labCTTwo[each, zero], zero]])).dot(cameraMatrixOne2))))))

    # Here we are normalizing the last row of V (homogeneous coordinates) and assign to lab 3D points array.
    labCT3D[each] = (V[len(V) - one]) / (V[len(V) - one])[- one]

# Here we are iterating over each pair of library matches.
for each in range(len(textSix)):

    # Here we are performing SVD for the stacked matrices after multiplication with library matrices.
    _, _, V = np.linalg.svd((np.vstack((((np.array([[zero, -libraryCTOne[each, two], libraryCTOne[each, one]], [libraryCTOne[each, two], zero, -libraryCTOne[each, zero]], [-libraryCTOne[each, one], libraryCTOne[each, zero], zero]])).dot(textThree)),

                                        # Here we are creating two skew-symmetric matrices (one above and one here).
                                        ((np.array([[zero, -libraryCTTwo[each, two], libraryCTTwo[each, one]], [libraryCTTwo[each, two], zero, -libraryCTTwo[each, zero]], [-libraryCTTwo[each, one], libraryCTTwo[each, zero], zero]])).dot(textFour))))))

    # Here we are normalizing the last row of V and assign to library 3D points array.
    libraryCT3D[each] = (V[len(V) -one]) / (V[len(V) - one])[-one]

# Here we are calculating and normalizing the residuals for lab calibration with the first camera matrix.
labCTResidualOne = (np.linalg.norm(((np.dot(cameraMatrixOne1, labCT3D.T).T) / (np.dot(cameraMatrixOne1, labCT3D.T).T)[:, -one][:, np.newaxis])[:, zero : two] - textFive[:, zero : two]) ** two) / textFive.shape[zero]

# Here we are calculating and normalizing the residuals for library calibration with the first text matrix.
libraryCTResidualOne = (np.linalg.norm(((np.dot(textThree, libraryCT3D.T).T) / (np.dot(textThree, libraryCT3D.T).T)[:, -one][:, np.newaxis])[:, zero : two] - textSix[:, zero : two]) ** two) / textSix.shape[zero]

# Here we are calculating and normalizing the residuals for lab calibration with the second camera matrix.
labCTResidualTwo = (np.linalg.norm(((np.dot(cameraMatrixOne2, labCT3D.T).T) / (np.dot(cameraMatrixOne2, labCT3D.T).T)[:, -one][:, np.newaxis])[:, zero : two] - textFive[:, two : four]) ** two) / textFive.shape[zero]

# Here we are calculating and normalizing the residuals for library calibration with the second text matrix.
libraryCTResidualTwo = (np.linalg.norm(((np.dot(textFour, libraryCT3D.T).T) / (np.dot(textFour, libraryCT3D.T).T)[:, -one][:, np.newaxis])[:, zero : two] - textSix[:, two : four]) ** two) / textSix.shape[zero]

# Here we are printing the residual error from lab calibration and triangulation for the first set of data.
print("\nLab Calibration And Triangulation Residual One = {}".format(labCTResidualOne))

# Here we are printing the residual error from lab calibration and triangulation for the second set of data.
print("\nLab Calibration And Triangulation Residual Two = {}".format(labCTResidualTwo))

# Here we are printing the residual error from library calibration and triangulation for the first set of data.
print("\nLibrary Calibration And Triangulation Residual One = {}".format(libraryCTResidualOne))

# Here we are printing the residual error from library calibration and triangulation for the second set of data.
print("\nLibrary Calibration And Triangulation Residual Two = {}".format(libraryCTResidualTwo))

# Here we are creating a 3D subplot for lab calibration and triangulation data.
labCalibrationAndTriangulation = plt.figure().add_subplot(projection = '3d')

# Here we are scatter plotting the lab camera center one position in 3D space.
labCalibrationAndTriangulation.scatter(labCameraCenterOne[zero], labCameraCenterOne[one], labCameraCenterOne[two])

# Here we are scatter plotting the lab camera center two position in 3D space.
labCalibrationAndTriangulation.scatter(labCameraCenterTwo[zero], labCameraCenterTwo[one], labCameraCenterTwo[two])

# Here we are scatter plotting the 3D points from lab calibration and triangulation.
labCalibrationAndTriangulation.scatter(labCT3D[:, zero], labCT3D[:, one], labCT3D[:, two])

# Here we are setting the x-axis label of the lab calibration and triangulation plot.
labCalibrationAndTriangulation.set_xlabel('x')

# Here we are setting the y-axis label of the lab calibration and triangulation plot.
labCalibrationAndTriangulation.set_ylabel('y')

# Here we are setting the z-axis label of the lab calibration and triangulation plot.
labCalibrationAndTriangulation.set_zlabel('z')

plt.show()   # Here we are displaying the plot for the lab calibration and triangulation.

# Here we are creating a 3D subplot for library calibration and triangulation data.
libraryCalibrationAndTriangulation = plt.figure().add_subplot(projection = '3d')

# Here we are scatter plotting the library camera center one position in 3D space.
libraryCalibrationAndTriangulation.scatter(libraryCameraCenterOne[zero], libraryCameraCenterOne[one], libraryCameraCenterOne[two])

# Here we are scatter plotting the library camera center two position in 3D space.
libraryCalibrationAndTriangulation.scatter(libraryCameraCenterTwo[zero], libraryCameraCenterTwo[one], libraryCameraCenterTwo[two])

# Here we are scatter plotting the 3D points from library calibration and triangulation.
libraryCalibrationAndTriangulation.scatter(libraryCT3D[:, zero], libraryCT3D[:, one], libraryCT3D[:, two])

# Here we are setting the x-axis label of the library calibration and triangulation plot.
libraryCalibrationAndTriangulation.set_xlabel('x')

# Here we are setting the y-axis label of the library calibration and triangulation plot.
libraryCalibrationAndTriangulation.set_ylabel('y')

# Here we are setting the z-axis label of the library calibration and triangulation plot.
libraryCalibrationAndTriangulation.set_zlabel('z')

plt.show()   # Here we are displaying the plot for the library calibration and triangulation.

# Here we are creating a function called computeMatrixHelperOne.
def computeMatrixHelperOne(p1):

    # Here we are concatenating p1 with a column of ones to the right and return the result.
    return np.concatenate((p1, (np.ones(((p1.shape[zero]), one)))), axis = one)

# Here we are creating a function called computeMatrixHelperTwo.
def computeMatrixHelperTwo(p1, p2):

    # Here we are computing the dot product of arrays "p1" and the transpose of "p2", then transpose the result.
    return np.dot(p1, p2.T).T

# Here we are creating a function called computeMatrixHelperThree.
def computeMatrixHelperThree(p1, p2):

    # Here we are calculating the absolute value of the dot product between p1 and p2, divided by the Euclidean norm of p2, for each row in the arrays, and return the result as a new matrix.
    return np.abs((np.sum(p1 * p2, axis = one)[:, np.newaxis])) / np.linalg.norm(p2, axis = one)[:, np.newaxis]

# Here we are creating a function called computeMatrix.
def computeMatrix(p1, p2):

    # Here we are calling the computeMatrixHelperThree, computeMatrixHelperTwo, computeMatrixHelperOne functions for the first time.
    return ((computeMatrixHelperThree((computeMatrixHelperOne(p1[:, zero:two])), (computeMatrixHelperTwo(p2.T, (computeMatrixHelperOne(p1[:, two:four])))))) +

            # Here we are calling the computeMatrixHelperThree, computeMatrixHelperTwo, computeMatrixHelperOne functions for the first time.
            (computeMatrixHelperThree((computeMatrixHelperOne(p1[:, two:four])), (computeMatrixHelperTwo(p2, (computeMatrixHelperOne(p1[:, zero:two]))))))) / two

# Here we are creating a function called computeDescriptors.
def computeDescriptors(picture):

    # Here we are creating a SIFT detector and compute keypoints and descriptors for the given image.
    return cv2.xfeatures2d.SIFT_create().detectAndCompute(picture, None)

# Here we are creating a function called ransacImplementation.
def ransacImplementation(p1, limit):

    numberOfInliers = zero   # Here we are initializing the number of inliers.

    averageInlierResidual = zero   # Here we are initializing the average inlier residual.

    for _ in range(numberOne):   # Here we are looping the RANSAC 100 times.

        # Here we are selecting all inliers based on a computed matrix and residual limit.
        allInliers = (p1[(np.where((computeMatrix(p1, (fit_fundamental(p1, p2 = True)))) < limit)[zero])])

        # Here we are checking if the current number of inliers is less than the total number of inliers.
        if numberOfInliers < len(allInliers):

            # Here we are creating a copy of the 'allInliers' list and store it in 'newInliers'.
            newInliers = allInliers.copy()

            # Here we are updating the 'numberOfInliers' variable with the new length of the 'allInliers' list.
            numberOfInliers = len(allInliers)

            # Here we are calculating the average residual of the new inliers.
            averageInlierResidual = (computeMatrix(p1, (fit_fundamental(p1, p2 = True))))[(np.where((computeMatrix(p1, (fit_fundamental(p1, p2 = True)))) < limit)[zero])].sum() / len(allInliers)

    # Here we are returning the best set of inliers and their average residual.
    return newInliers, averageInlierResidual

# Here we are creating a function called computeDistances.
def computeDistances(limit, p1, p2, p3, p4):

    # Here we are concatenating two arrays vertically using NumPy's concatenate function.
    return np.concatenate((

            # Here we are creating a NumPy array containing the points from p1 based on certain conditions.
            (np.array([p1[each].pt for each in (np.where((distance.cdist(p3, p4, 'sqeuclidean')) < limit)[zero])])),

            # Here we are creating a NumPy array containing the points from p2 based on certain conditions.
            (np.array([p2[each].pt for each in (np.where((distance.cdist(p3, p4, 'sqeuclidean')) < limit)[one])]))),

            axis = one)   # Here we are concatenating the arrays horizontally (along columns).

# Here we are creating a function called plot_inlier_matchesOne.
def plot_inlier_matchesHelperOne(p1):

    # Here we are opening an image file specified by 'p1' and converts it to grayscale, then returns the result.
    return Image.open(p1).convert('L')

# Here we are creating a function called plot_inlier_matchesTwo.
def plot_inlier_matchesHelperTwo(p1, p2):

    # Here we are creating an empty NumPy array to store the combined image, with dimensions determined by the maximum height and the sum of widths of p1 and p2.
    result = np.zeros(((max(p1.size[one], p2.size[one])), p1.size[zero] + p2.size[zero]))

    # Here we are placing image p1 on the left side of the result array.
    result[:p1.size[one], :p1.size[zero]] = p1

    # Here we are placing image p2 on the right side of the result array.
    result[:p2.size[one], p1.size[zero]:] = p2

    return result   # Here we are returning the combined image.

# Here we are creating a function called plot_inlier_matchesThree.
def plot_inlier_matchesHelperThree(p1, p2, p3):

    # Here we are plotting points from p2 on p1 as red crosses.
    p1.plot(p2[:, zero], p2[:, one], '+b')

    # Here we are plotting points from p2 and p3 combined on p1 as red crosses.
    p1.plot(p2[:, two] + p3, p2[:, three], '+b')

    # Here we are plotting lines connecting points from p2 and p3 on p1 in red.
    p1.plot([p2[:, zero], p2[:, two] + p3], [p2[:, one], p2[:, three]], 'y', linewidth = numberThree)

# Here we are creating a function called plot_inlier_matches.
def plot_inlier_matches(p1, p2):

    _, subplotTwo = plt.subplots()   # Here we are creating two subplots for visualization.

    subplotTwo.set_aspect('equal')   # Here we are setting the aspect ratio of the second subplot to be equal.

    # Here we are displaying the computed inlier matches as a grayscale image.
    subplotTwo.imshow((plot_inlier_matchesHelperTwo((plot_inlier_matchesHelperOne('MP4_part2_data/' + p2 + '1.jpg')), (plot_inlier_matchesHelperOne('MP4_part2_data/' + p2 + '2.jpg')))).astype(float), cmap = 'gray')

    # Here we are plotting additional inlier matches.
    plot_inlier_matchesHelperThree(subplotTwo, p1, (plot_inlier_matchesHelperOne('MP4_part2_data/' + p2 + '1.jpg')).size[zero])

    plt.axis('off')   # Here we are turning off axis labels and ticks.

    plt.show()   # Here we are displaying the final plot.

# Here we are creating a function called print_inlier_matches.
def print_inlier_matches(p1):

    # Here we are computing descriptors for the first image and assigning the results to partOne and partTwo.
    partOne, partTwo = computeDescriptors((np.array(Image.open('MP4_part2_data/' + p1 + '1.jpg').convert('L'))))

    # Here we are computing descriptors for the second image and assigning the results to partThree and partFour.
    partThree, partFour = computeDescriptors((np.array(Image.open('MP4_part2_data/' + p1 + '2.jpg').convert('L'))))

    # Here we are performing RANSAC and obtaining the number of inliers, and the average inlier residual for the images.
    numberOfInliersTwo, averageInlierResidualTwo = ransacImplementation(computeDistances(numberTwo, partOne, partThree, partTwo, partFour), numberThree)

    # Here we are printing the number of inliers for the images.
    print("\nHere is the number of inliers for the house image {}: ".format(len(numberOfInliersTwo)))

    # Here we are printing the average inlier residual for the images.
    print("\nHere is the average inlier residual for the house image: {}".format(averageInlierResidualTwo))

    # Here we are plotting inlier matches for the images.
    plot_inlier_matches(numberOfInliersTwo, p1)

# Here we are calling the print_inlier_matches function with the gaudi parameter.
print_inlier_matches('gaudi')

# Here we are calling the print_inlier_matches function with the house parameter.
print_inlier_matches('house')