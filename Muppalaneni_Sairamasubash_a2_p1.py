# Here we are importing all the required libraries for this assignment.
import os
import time
import numpy as np
from scipy import ndimage
from PIL import Image
from matplotlib import pyplot as plt

PRE = True   # Here we are creating a global variable called PRE.

# Here we are implementing the shift function to move the images that are split.
def shift(picture, move, displacement):

    # Here is the mapping of move values to corresponding axis and direction.
    axis_map = {0: (0, -1), 2: (0, 1), 1: (1, 1), 3: (1, -1)}

    # Here we are fetching the corresponding axis and direction based on the move value.
    axis, direction = axis_map.get(move, (None, None))

    # Here we are checking to see if the axis is valid.
    if axis is not None:

        # Here we are performing the shift operation if the axis is valid.
        return np.roll(picture, direction * displacement, axis=axis)

    return picture   # Here we are returning the picture variable.

# Here we are implementing the alignment function to align the images that are split.
def alignment(picture, bestDisplacementOne, bestDisplacementTwo):

    # Here we are shifting the third channel of the picture using the best displacements.
    picture[2] = shift(shift(picture[2], 2, bestDisplacementOne[0]), 1, bestDisplacementOne[1])

    # Here we are shifting the second channel of the picture using the best displacements.
    picture[1] = shift(shift(picture[1], 2, bestDisplacementTwo[0]), 1, bestDisplacementTwo[1])

    return picture   # Here we are returning the picture variable.

# Here we are implementing the channel function to return an input channel.
def channel(channel_input):

    # Here we are returning a flipped and transposed version of the input channel.
    return np.flip(channel_input.transpose(1, 2, 0), axis=-1)


# Here we are implementing the norm function to normalize the RGB values in an image.
def norm(picture):

    shape = picture.shape   # Here we are getting the shape (dimensions) of the picture.

    # Here we are flattening the picture to have 3 rows (R, G, B) and auto-calculated columns.
    flattened = picture.reshape(3, -1)

    # Here we are calculating the mean of each RGB channel.
    mean_vector = np.mean(flattened, axis=1, keepdims=True)

    # Here we are finding the deviation of each RGB value from its channel's mean.
    deviation = flattened - mean_vector

    # Here we are calculating the normalization factor (Euclidean norm) for each RGB channel.
    normalization_factor = np.linalg.norm(deviation, axis=1, keepdims=True)

    # Here we are normalizing each RGB deviation by its channel's normalization factor.
    normalized = deviation / normalization_factor

    # Here we are reshaping the normalized values to the original picture's shape.
    return normalized.reshape(shape)

# Here we are implementing the FT function to perform Fourier Transform operations on two channels to determine displacement.
def FT(allChannels, channelOne, channelTwo):

    # Here we are implementing the get_channel_data function to retrieve and preprocess channel data if needed.
    def get_channel_data(channel, preprocessing=True):

        # Here we are checking to see if preprocessing is being used.
        if preprocessing:

            # Here we are applying the Gaussian Laplace filter for preprocessing.
            return ndimage.gaussian_laplace(allChannels[channel], 3)

        # Here we are returning the raw channel data.
        return allChannels[channel]

    # Here we are implementing the get_fft function to get the FFT of the data.
    def get_fft(data):

        # Here we are returning the 2D FFT of the given data, shifted to center.
        return np.fft.fftshift(np.fft.fft2(data))

    # Here we are recording the start time of the operation.
    start_time = time.time()

    # Here we are getting and preprocessing (if enabled) data from the first channel.
    one = get_channel_data(channelOne, PRE)

    # Here we are getting and preprocessing (if enabled) data from the second channel.
    two = get_channel_data(channelTwo, PRE)

    # Here we are obtaining the FFT of the data from the first channel.
    FT1 = get_fft(one)

    # Here we are obtaining the conjugate FFT of the data from the second channel.
    FT2Conjugate = np.conjugate(get_fft(two))

    # Here we are multiplying the two FFT results element-wise.
    product = np.multiply(FT1, FT2Conjugate)

    # Here is the inverse FFT of the product.
    inverse = np.fft.ifft2(product)

    # Here we are normalizing the result using logarithm for better visualization.
    normalize = np.log(np.abs(inverse))

    # Here we are identifying the index with the maximum value in the normalized data.
    max = np.unravel_index(normalize.argmax(), normalize.shape)

    # Here we are checking  if preprocessing is disabled.
    if not PRE:

        # Here we are calculating the total time taken for the operation.
        total_time = time.time() - start_time

        # Here we are printing the time taken.
        print(f"Here is the total time taken to process this image: {total_time}")

        # Here we are printing the displacements.
        print(f"Here are the best displacements for this image: {max}")

    # Here we are returning the best displacement and the normalized data.
    return max, normalize

# Here we are creating a list called lowImagePaths that holds the paths to the six low resolution images.
lowImagePaths = [

    "LowImages/00125v.jpg",

    "LowImages/00153v.jpg",

    "LowImages/00149v.jpg",

    "LowImages/00351v.jpg",

    "LowImages/01112v.jpg",

    "LowImages/00398v.jpg",

]

# Here we are creating a list called highImagePath that holds the paths to the three high resolution images.
highImagePath = [

    "HighImages/01657u.tif",

    "HighImages/01047u.tif",

    "HighImages/01861a.tif",

]

# Here we are looping through the list called lowImagePaths.
for each in lowImagePaths:

    allImages = plt.figure(figsize=(20, 12))   # Here we are creating a new figure with specified size.

    r, c = 2, 3   # Here we are defining the number of rows and columns for subplots.

    imageOne = Image.open(each)   # Here we are opening an image from the given path.

    imageTwo = np.array(imageOne)   # Here we are converting the opened image to a numpy array.

    # Here we are using a while loop to ensure that the height of the image is divisible by 3 by padding if needed.
    while imageTwo.shape[0] % 3 != 0:

        # Here we are using np.pad with a few parameters.
        imageTwo = np.pad(imageTwo, ((0, 1), (0, 0)), mode='constant', constant_values=255)

    channels = np.array(np.array_split(imageTwo, 3))   # Here we are splitting the image into 3 equal parts (channels).

    # Here we are looping through two conditions: without and with preprocessing.
    for r2, pre in enumerate([False, True]):

        PRE = pre   # Here we are setting the preprocessing flag.

        d1, p1 = FT(np.copy(channels), 0, 2)   # Here we are applying Fourier Transform for the first pair of channels.

        d2, p2 = FT(np.copy(channels), 0, 1)   # Here we are applying Fourier Transform for the second pair of channels.

        # Here we are aligning the channels based on the Fourier Transforms.
        newAlignment = alignment(channels, d1, d2)

        newImage = channel(newAlignment)   # Here we are constructing a new image from the aligned channels.

        # Here we are checking to see if we are using preprocessing.
        if PRE:

            # Here we are adding a subplot for the aligned image with preprocessing.
            allImages.add_subplot(r, c, c * r2 + 1)

            plt.title('Aligned Image')   # Here we are setting the title for the subplot.

            plt.axis('off')   # Here we are turning off the axis.

            plt.imshow(newImage)   # Here we are displaying the new image.

        # Here we are adding a subplot for the R to B alignment.
        allImages.add_subplot(r, c, c * r2 + 2)

        # Here we are creating a R to B alignment title.
        plt.title('R to B alignment ' + ('With' if PRE else 'Without') + ' Preprocessing')

        plt.set_cmap('jet')   # Here we are setting the colormap.

        plt.axis('off')   # Here we are turning off the axis.

        plt.imshow(p1)   # Here we are displaying the first Fourier Transform result.

        # Here we are adding a subplot for the G to B alignment.
        allImages.add_subplot(r, c, c * r2 + 3)

        # Here we are creating a G to B alignment title.
        plt.title('G to B alignment ' + ('With' if PRE else 'Without') + ' Preprocessing')

        plt.set_cmap('jet')   # Here we are setting the colormap.

        plt.axis('off')   # Here we are turning off the axis.

        plt.imshow(p2)   # Here we are displaying the second Fourier Transform result.

    # Here we are saving each of the created figures in a directory called LowColor.
    figureSavePath = os.path.join("LowColor", os.path.basename(each).replace('.jpg', '_plot.png'))

    # Saving the 'allImages' figure to the specified path with a resolution of 300 dpi.
    allImages.savefig(figureSavePath, dpi=300, bbox_inches='tight')

    # Adjusting the subplots of the current figure to fit within the figure area.
    plt.tight_layout()

    plt.show()   # Here we are showing the entire plot.

# Here we are looping through the list called highImagePath.
for each in highImagePath:

    allImages = plt.figure(figsize=(20, 12))   # Here we are creating a new figure with specified size.

    r, c = 2, 3   # Here we are defining the number of rows and columns for subplots.

    imageOne = Image.open(each)   # Here we are opening an image from the given path.

    imageTwo = np.array(imageOne)   # Here we are converting the opened image to a numpy array.

    # Here we are using a while loop to ensure that the height of the image is divisible by 3 by padding if needed.
    while imageTwo.shape[0] % 3 != 0:

        # Here we are using np.pad with a few parameters.
        imageTwo = np.pad(imageTwo, ((0, 1), (0, 0)), mode='constant', constant_values=255)

    channels = np.array(np.array_split(imageTwo, 3))   # Here we are splitting the image into 3 equal parts (channels).

    # Here we are looping through two conditions: without and with preprocessing.
    for r2, pre in enumerate([False, True]):

        PRE = pre   # Here we are setting the preprocessing flag.

        d1, p1 = FT(np.copy(channels), 0, 2)   # Here we are applying Fourier Transform for the first pair of channels.

        d2, p2 = FT(np.copy(channels), 0, 1)   # Here we are applying Fourier Transform for the second pair of channels.

        # Here we are aligning the channels based on the Fourier Transforms.
        newAlignment = alignment(channels, d1, d2)

        newImage = channel(newAlignment)   # Here we are constructing a new image from the aligned channels.

        # Here we are normalizing the new image values to be between 0 and 255.
        newImage = ((newImage / np.max(newImage)) * 255).astype(np.uint8)

        # Here we are checking to see if we are using preprocessing.
        if PRE:

            # Here we are adding a subplot for the aligned image with preprocessing.
            allImages.add_subplot(r, c, c * r2 + 1)

            plt.title('Aligned Image')   # Here we are setting the title for the subplot.

            plt.axis('off')   # Here we are turning off the axis.

            plt.imshow(newImage)   # Here we are displaying the new image.

        # Here we are adding a subplot for the R to B alignment.
        allImages.add_subplot(r, c, c * r2 + 2)

        # Here we are creating a R to B alignment title.
        plt.title('R to B alignment ' + ('With' if PRE else 'Without') + ' Preprocessing')

        plt.set_cmap('jet')   # Here we are setting the colormap.

        plt.axis('off')   # Here we are turning off the axis.

        plt.imshow(p1)   # Here we are displaying the first Fourier Transform result.

        # Here we are adding a subplot for the G to B alignment.
        allImages.add_subplot(r, c, c * r2 + 3)

        # Here we are creating a G to B alignment title.
        plt.title('G to B alignment ' + ('With' if PRE else 'Without') + ' Preprocessing')

        plt.set_cmap('jet')   # Here we are setting the colormap.

        plt.axis('off')   # Here we are turning off the axis.

        plt.imshow(p2)   # Here we are displaying the second Fourier Transform result.

    # Here we are saving each of the created figures in a directory called LowColor.
    figureSavePath = os.path.join("HighColor", os.path.basename(each).replace('.tif', '_plot.png'))

    # Saving the 'allImages' figure to the specified path with a resolution of 300 dpi.
    allImages.savefig(figureSavePath, dpi=300, bbox_inches='tight')

    # Adjusting the subplots of the current figure to fit within the figure area.
    plt.tight_layout()

    # Here we are showing the entire plot.
    plt.show()