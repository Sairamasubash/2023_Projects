# imports
import os
import sys
import glob
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import time   # Here I am importing the time library.


#####################################
### Provided functions start here ###
#####################################

# Image loading and saving

def LoadFaceImages(pathname, subject_name, num_images):
    """
    Load the set of face images.
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs

def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,0]*128+128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,1]*128+128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,2]*128+128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)


# Plot the height map

def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    surf = ax.plot_surface(
        H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)
    plt.show()   # Here I am showing the display_output images.


# Plot the surface normals

def plot_surface_normals(surface_normals):
    """
    surface_normals: h x w x 3 matrix.
    """
    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:,:,0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:,:,1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:,:,2])
    plt.show()   # Here I am showing the plot_surface_normals images.


#######################################
### Your implementation starts here ###
#######################################

# Here is the full definition of the preprocess function.
def preprocess(ambimage, imarray):
    """
    preprocess the data:
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """

    # Here we are subtracting ambimage from imarray, normalize by dividing by 255, and clip values to be in range [0, infinity].
    processed_imarray = np.clip((imarray - ambimage[:, :, np.newaxis]) / 255, 0, None)

    return processed_imarray   # Here we are returning the processed image array.


# Here is the full definition of the photometric_stereo function.
def photometric_stereo(imarray, light_dirs):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """

    # Here we are solving the system of linear equations to find the intensity values for each pixel.
    output = np.linalg.lstsq(light_dirs, imarray.reshape(imarray.shape[0] * imarray.shape[1], imarray.shape[2]).transpose())

    # Here we are computing the albedo image by finding the magnitude of the intensity values.
    albedo_image = (np.linalg.norm(output[0], axis = 0)).reshape(imarray.shape[0], imarray.shape[1])

    # Here we are normalizing the intensity values to find the surface normals and reshape them to the original image shape.
    surface_normals = (output[0] / (np.linalg.norm(output[0], axis = 0))).transpose().reshape(imarray.shape[0], imarray.shape[1], 3)

    return albedo_image, surface_normals   # Here we are returning the albedo image and surface normals.



# Here is the full definition of the get_surface function.
def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """

    startTime = time.time()   # Here we are starting to track the execution time.

    if integration_method == 'row':   # Here we are integrating using the row-based method.

        # Here is the value of height_map when the integration method is row.
        height_map = (np.cumsum((surface_normals[:, :, 0] / surface_normals[:, :, 2]), axis = 1))[0] + (np.cumsum((surface_normals[:, :, 1] / surface_normals[:, :, 2]), axis = 0))

    elif integration_method == 'column':   # Here we are integrating using the column-based method.

        # Here is the value of height_map when the integration method is column.
        height_map = (np.cumsum((surface_normals[:, :, 1] / surface_normals[:, :, 2]), axis = 0))[:, 0][:, np.newaxis] + (np.cumsum((surface_normals[:, :, 0] / surface_normals[:, :, 2]), axis = 1))

    elif integration_method == 'average':   # Here we are integrating using the average-based method.

        # Here is the value of height_map when the integration method is average.
        height_map = ((np.cumsum((surface_normals[:, :, 1] / surface_normals[:, :, 2]), axis = 0))[:, 0][:, np.newaxis] + (np.cumsum((surface_normals[:, :, 0] / surface_normals[:, :, 2]), axis = 1)) + (np.cumsum((surface_normals[:, :, 0] / surface_normals[:, :, 2]), axis = 1))[0] + (np.cumsum((surface_normals[:, :, 1] / surface_normals[:, :, 2]), axis = 0))) / 2

    elif integration_method == 'random':   # Here we are integrating using the random-based method.

        # Here we are extracting the dimensions of the surface normals matrix.
        height, width, _ = surface_normals.shape

        denominator = surface_normals[:, :, 2]   # Here we are calculating the denominator for the division operation.

        # Here we are calculating horizontal component of the normalized normal.
        horizontal = surface_normals[:, :, 0] / denominator

        # Here we are calculating vertical component of the normalized normal.
        vertical = surface_normals[:, :, 1] / denominator

        height_map = np.zeros((height, width))   # Here we are initializing height_map with zeros.

        # Here we are looping through every pixel except the top-left corner.
        for verticalOne, horizontalOne in [(flat, straight) for flat in range(height) for straight in range(width) if flat != 0 or straight != 0]:

            # Here we are performing 30 random walks, for each pixel.
            for each in range(30):

                # Here we are creating an array of directions (0 for horizontal, 1 for vertical).
                allParts = np.array([0] * horizontalOne + [1] * verticalOne)

                np.random.shuffle(allParts)   # Here we are randomly shuffling the directions.

                # Here we are initializing counters for current position and accumulated surface height.
                horizontalTwo = verticalTwo = eachPart = total = 0

                # Here we are performing the random walk until reaching the target pixel.
                while horizontalTwo < horizontalOne or verticalTwo < verticalOne:

                    # Here we are selecting which matrix to use for the walk (horizontal or vertical).
                    matrix = vertical if allParts[eachPart] else horizontal

                    # Here we are accumulating the surface height.
                    total += matrix[verticalTwo, horizontalTwo]

                    # Here we are checking to see if we are moving in one direction.
                    if allParts[eachPart]:

                        verticalTwo += 1   # Here we are incrementing verticalTwo.

                    else:   # Here we are checking to see if we are moving in another direction.

                        horizontalTwo += 1   # Here we are incrementing horizontalTwo.

                    eachPart += 1   # Here we are moving to the next direction in the array.

                # Here we are averaging the accumulated surface height over the 30 random walks.
                coordinates = (verticalOne, horizontalOne)

                # Here we are including the coordinates variable for the height_map.
                height_map[coordinates] += total

            # Here is the value of height_map when the integration method is random.
            height_map[verticalOne, horizontalOne] /= 30

    endTime = time.time()   # Here we are recording the end time of the execution.

    # Here we are printing the integration method used and the time taken for execution.
    print('The Integration Method Is: ' + integration_method + '. \nThe Execution Time Is: {} seconds.'.format(endTime - startTime))

    return height_map   # Here we are returning the computed height map.



# Main function
if __name__ == '__main__':
    root_path = 'croppedyale/'
    subject_name = 'yaleB01'
    integration_method = 'random'
    save_flag = True

    full_path = '%s%s' % (root_path, subject_name)
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path, subject_name,
                                                        64)

    processed_imarray = preprocess(ambient_image, imarray)

    albedo_image, surface_normals = photometric_stereo(processed_imarray,
                                                    light_dirs)

    height_map = get_surface(surface_normals, integration_method)   # Here I am changing 'average' into integration_method.

    if save_flag:
        save_outputs(subject_name, albedo_image, surface_normals)

    plot_surface_normals(surface_normals)

    display_output(albedo_image, height_map)