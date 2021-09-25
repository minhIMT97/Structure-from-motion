import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    #AF = 0
    A = np.zeros((points1.shape[0], 9))
    for i in range(A.shape[0]):
        A[i,:] = np.array([points1[i,0]*points2[i,0], 
                            points1[i,1]*points2[i,0],
                            points2[i,0],
                            points1[i,0]*points2[i,1],
                            points1[i,1]*points2[i,1],
                            points2[i,1],
                            points1[i,0],
                            points1[i,1], 1])
    b = np.zeros((A.shape[0],))
    # print(A)   
    # Use svd to find the lstsq solution for Wf = 0
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    # diag_s = np.diag(s)
    # diag_s_inv = np.linalg.inv(diag_s)
    # print(u.shape, s.shape, vh.shape)
    # f = np.dot(vh.dot(diag_s_inv).dot(u),b)
    f = vh[-1, :]
    F_t = f.reshape(3, 3)
    # Enforce F_t to F which is rank 2
    u, s, vh = np.linalg.svd(F_t, full_matrices=True)
    s[-1] = 0
    F = u.dot(np.diag(s)).dot(vh)
    return F
    # u1 = points1[:, 0]
    # v1 = points1[:, 1]
    # u1_p = points2[:, 0]
    # v1_p = points2[:, 1]
    # one = np.ones_like(u1)
    # W = np.c_[u1 * u1_p, v1 * u1_p, u1_p, u1 * v1_p, v1 * v1_p, v1_p, u1, v1, one]
    # # Use svd to find the lstsq solution for Wf = 0
    # u, s, vh = np.linalg.svd(W, full_matrices=True)
    # f = vh[-1, :]
    # F_t = f.reshape(3, 3)
    # # Enforce F_t to F which is rank 2
    # u, s, vh = np.linalg.svd(F_t, full_matrices=True)
    # s[-1] = 0
    # F = u.dot(np.diag(s)).dot(vh)
    # return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    d1 = np.max((points1[:,0] - centroid1[0])**2 + (points1[:,1] - centroid1[1])**2)
    d2 = np.max((points2[:,0] - centroid2[0])**2 + (points2[:,1] - centroid2[1])**2)
    
    s1 = 2/np.sqrt(d1)
    s2 = 2/np.sqrt(d2)
    
    T1 = np.array([[s1,0,-s1*centroid1[0]], [0, s1,-s1*centroid1[1]], [0,0,1]])
    T2 = np.array([[s2,0,-s2*centroid2[0]], [0, s2,-s2*centroid2[1]], [0,0,1]])
    
    q1 = T1.dot(points1.T).T
    q2 = T2.dot(points2.T).T
    # print(q2)
    
    
    A = np.zeros((q1.shape[0], 9))
    for i in range(A.shape[0]):
        A[i,:] = np.array([ q1[i,0]*q2[i,0], 
                            q2[i,0]*q1[i,1],
                            q2[i,0],
                            q2[i,1]*q1[i,0],
                            q1[i,1]*q2[i,1],
                            q2[i,1],
                            q1[i,0],
                            q1[i,1], 1])
    b = np.zeros((A.shape[0],))
    # print(A)   
    # Use svd to find the lstsq solution for Wf = 0
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    # diag_s = np.diag(s)
    # diag_s_inv = np.linalg.inv(diag_s)
    # print(u.shape, s.shape, vh.shape)
    # f = np.dot(vh.dot(diag_s_inv).dot(u),b)
    f = vh[-1, :]
    F_t = f.reshape(3, 3)
    # Enforce F_t to F which is rank 2
    u, s, vh = np.linalg.svd(F_t, full_matrices=True)
    s[-1] = 0
    Fq = u.dot(np.diag(s)).dot(vh)
    F = np.dot(T2.T, Fq).dot(T1)
    
    return F
    # mean1 = np.mean(points1, axis=0)
    # mean2 = np.mean(points2, axis=0)
    
    # scale1 = np.sqrt(2 / np.mean(np.sum((points1 - mean1) ** 2, axis=1)))
    # scale2 = np.sqrt(2 / np.mean(np.sum((points2 - mean2) ** 2, axis=1)))
    # T = np.array([[scale1, 0, -scale1 * mean1[0]], [0, scale1, -scale1 * mean1[1]], [0, 0 ,1]])
    # T_p = np.array([[scale2, 0, -scale2 * mean2[0]], [0, scale2, -scale2 * mean2[1]], [0, 0, 1]])
    # # q = T * p
    # points1 = T.dot(points1.T).T
    # points2 = T_p.dot(points2.T).T
    # Fq = lls_eight_point_alg(points1, points2)
    # # de-normalize
    # return T_p.T.dot(Fq).dot(T)

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # TODO: Implement this method!
    l = F.dot(points2.T)
    d = np.mean(np.abs(np.sum(l * points1.T, axis=0)) / np.sqrt(l[0, :] ** 2 + l[1, :] ** 2))

    return d
if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        # plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
