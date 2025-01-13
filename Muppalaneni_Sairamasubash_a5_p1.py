# Here we are importing all the required libraries for this assignment.
import numpy as np
from numpy import linalg
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.image as img

zero = 0   # Here we are defining the variable zero.

one = 1   # Here we are defining the variable one.

two = 2   # Here we are defining the variable two.

three = 3   # Here we are defining the variable three.

valueOne = 202   # Here we are defining the variable valueOne.

valueTwo = 50   # Here we are defining the variable valueTwo.

valueThree = 100   # Here we are defining the variable valueThree.

textData = np.loadtxt('factorization_data/measurement_matrix.txt')   # Here we are loading data from a text file.

# Here we are computing the mean and reshape the data.
normalizeData = np.mean(textData, axis = one).reshape(valueOne, one)

textNormalizeData = textData - normalizeData   # Here we are subtracting the mean from the data.

# Here we are performing Singular Value Decomposition (SVD) on the normalized data.
U, s, V = linalg.svd(textNormalizeData)

# Here we are computing an approximation using a subset of the SVD components.
svdValue = ((U[:, zero:three]).dot((sqrtm((np.diag(s)[zero:three, zero:three]))))) @ ((sqrtm((np.diag(s)[zero:three, zero:three]))).dot((V[zero:three, :])))

# Here we are creating an identity matrix of a specific shape.
qMatrixHelper = np.identity((((U[:, zero:three]).dot((sqrtm((np.diag(s)[zero:three, zero:three]))))).shape[zero]))

# Here we are computing a Cholesky decomposition of a matrix.
qMatrix = np.linalg.cholesky((np.matrix(((U[:, zero:three]).dot((sqrtm((np.diag(s)[zero:three, zero:three])))))).I @ qMatrixHelper @ np.matrix(((U[:, zero:three]).dot((sqrtm((np.diag(s)[zero:three, zero:three]))))).T).I))

# Here we are printing the computed Q matrix.
print('\nQ Matrix Found To Eliminate The Affine Ambiguity:\n\n', qMatrix)

_, subplotOne = plt.subplots(subplot_kw = {'projection': '3d'})   # Here we are creating a 3D subplot.

# Here is the scatter plot in 3D space.
subplotOne.scatter3D(*((qMatrix.I @ sqrtm(np.diag(s)[zero:three, zero:three]) @ V[zero:three, :]).T)[:, :three].T, color = "green")

subplotOne.set_xlabel('x')   # Here we are setting x-axis label.

subplotOne.set_ylabel('y')   # Here we are setting y-axis label.

subplotOne.set_zlabel('z')   # Here we are setting z-axis label.

subplotOne.set_title('Display Of The 3D Structure')   # Here we are setting the plot title.

plt.show()   # Here we are displaying the 3D scatter plot.

# Here we are looping over a list of values.
for each in [one, valueTwo, valueThree]:

    _, subplotTwo = plt.subplots()   # Here we are creating a subplot.

    # Here we are displaying an image.
    subplotTwo.imshow(img.imread(f'factorization_data/frame00000{(str(each).zfill(three))}.jpg'), aspect = 'equal', cmap = 'gray')

    # Here we are plotting data points with 'x' markers in red.
    subplotTwo.plot(textData[two * each - two, :], textData[two * each - one, :], 'xr')

    # Here we are plotting data points with 'x' markers in green.
    subplotTwo.plot(svdValue[two * each - two, :] + normalizeData[two * each - two, zero], svdValue[two * each - one, :] + normalizeData[two * each - one, zero], 'xg')

    plt.title('Frame Value: ' + str(each))   # Here we are setting the plot title.

    plt.show()   # Here we are displaying the subplot.

# Here we are calculating the residual.
residualResult = np.sqrt(((textNormalizeData - svdValue)[:: two] ** two + (textNormalizeData - svdValue)[one :: two] ** two)).sum(axis = one, keepdims = True)

# Here we are printing the total residual.
print('\nTotal Residual Over All The Frames:\n\n', np.sum(residualResult))

plt.title("Per Frame Residual Plot")   # Here we are setting the plot title.

# Here we are plotting the residual values.
plt.plot(np.arange(one, len(residualResult) + one), residualResult.ravel(), color = "green")

plt.xlabel('Frame Numbers')   # Here we are setting x-axis label.

plt.ylabel('Residual Numbers')   # Here we are setting y-axis label.

plt.show()   # Here we are displaying the residual plot.