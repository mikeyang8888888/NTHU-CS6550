import os
import cv2
import math
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import skimage
import copy


# f = open("output2.txt", "w")
# rawString = ""

# def write_file(saveimg, filename, stage):
#     img = np.copy(saveimg)
#     img_folder_save_path = './process_results/'+stage
#     if not os.path.exists(img_folder_save_path):
#         os.makedirs(img_folder_save_path)

#     outputImg = img.astype(np.uint8)
#     save_name = filename + '.png'
#     # print(save_name)
#     save_img_path = os.path.join(img_folder_save_path, save_name)
#     cv2.imwrite(save_img_path, outputImg)

def homography_matrix(left, right):

    X_i = right.T
    x_prime_i = left.T
    row, col = X_i.shape

    # final_p = []
    final_p = np.array([])

    for i in range(col):
        X = X_i[0, i]
        Y = X_i[1, i]       
        u = x_prime_i[0, i]
        v = x_prime_i[1, i]        

        xX = np.dot(u, X_i[0, i])
        xY = np.dot(u, X_i[1, i])
        yX = np.dot(v, X_i[0, i])
        yY = np.dot(v, X_i[1, i])

        # p_flat = [X, Y, 1, 0, 0, 0, -xX, -xY, -u, 0, 0, 0, X, Y, 1, -yX, -yY, -v]
        p_flat = np.array([X, Y, 1, 0, 0, 0, -xX, -xY, -u, 0, 0, 0, X, Y, 1, -yX, -yY, -v])
        # print(p_flat)
        # print("------------------------")

        # p = np.reshape(p_flat, (2,-1))
        
        # if i == 0:
        #     # final_p = np.reshape(p_flat, (2,-1))
        #     final_p = p_flat
        # else:
        #     # final_p = np.append(final_p, p, axis = 0)
        
        final_p = np.concatenate((final_p, p_flat))
       
    #     final_p.append(p_flat)

    final_p = np.reshape(final_p, (-1, 9))

    # print(final_p)
    # f.writelines(["["+str(final_p)+"]", "\n"])

    eigenvalue, eigenvectors = np.linalg.eig(np.dot(final_p.T, final_p))
    # print('----------------eigenvalue----------------')
    # print(eigenvalue)
    # print('----------------eigenvectors----------------')
    # print(eigenvectors)
    # print("--------------------------")

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigenvalue[i]), eigenvectors[:,i]) for i in range(len(eigenvalue))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(reverse=False)
    P = eigen_pairs[0][1]
    H = np.reshape(P, (3,3))
    # f.writelines(["["+str(H)+"]", "\n"])
    return H

def get_warping_image_position(color, label_pos):

    row, col, _ = color.shape

    pos_1 = label_pos[0, :]
    pos_2 = label_pos[1, :]
    pos_3 = label_pos[2, :]
    pos_4 = label_pos[3, :]

    rectangle = np.array([pos_1, pos_2, pos_3, pos_4])
    # print(rectangle)

    space = np.zeros((row, col, 3), dtype=np.uint8)
    # warp_region = cv2.fillPoly(space, [rectangle], (255, 255, 255))
    cv2.fillPoly(space, [rectangle], (255, 255, 255))

    # fig, ax = plt.subplots()
    # ax.imshow(space)
    # plt.show()

    # cv2.fillPoly(full_img, [rectangle], 1)

    # print(warp_region)
# --------------------------------------------------------
    # ROI_y, ROI_x = np.asarray(np.where( warp_region == 1))
    ROI_pos = np.asarray(np.where( space == 255))
    
    tmp = np.copy(ROI_pos[1, :])
    ROI_pos[1, :] = ROI_pos[0, :]
    ROI_pos[0, :] = tmp
    # print(ROI_pos)

    # fig, ax = plt.subplots()
    # ax.plot(ROI_pos[0, :], ROI_pos[1, :], "r+", markersize=2)
    # # ax.imshow(cv2.cvtColor(corner_pos, cv2.COLOR_BGR2RGB))
    # ax.imshow(color)
    # # ax.imshow(color)
    # plt.show()

    return ROI_pos, space

# def forward_warping(source_region, target_region, img, h_matrix):
#     r, c, _ = source_region.shape
#     # print(source_region.shape)
#     color_img = np.copy(img)

#     # target_region_pos_stack = np.vstack(target_region_label)
#     # target_region_pos = np.insert(target_region_pos_stack, 2, values=1, axis=1) 

#     # target_pos = np.array(target_region_pos_stack[:, :2])
#     # cv2.fillPoly(color_img, [target_pos], (255, 255, 255))

#     # fig, ax = plt.subplots()
#     # ax.imshow(color_img)
#     # plt.show()

#     new_target = np.copy(target_region)

#     for i in range(r):
#         for j in range(c):
#             if source_region[i, j, 0] == 255 and source_region[i, j, 1] == 255 and source_region[i, j, 2]== 255:

#                 source = np.array([j, i, 1]).reshape(3, 1)

#                 target = np.dot(h_matrix, source)
#                 # f.writelines(["------------------"+str(i)+","+str(j)+"-------------------------", "\n"])
#                 # f.writelines(["["+str(h_matrix)+"]", "\n"])
#                 # f.writelines(["["+str(source)+"]", "\n"])

#                 x = np.round(target[0] / target[2]).astype('int64')
#                 y = np.round(target[1] / target[2]).astype('int64')                
#                 # # print(x,",",y)

#                 # f.writelines([" ------------------------------------- ", "\n"])
#                 # # f.writelines(["["+str(i)+","+str(j)+"]", "\n"])
#                 # f.writelines(["["+str(target)+"]", "\n"])
#                 # f.writelines(["***************************************", "\n"])

#                 new_target[y, x, :] = color_img[i, j, :]

#                 # print(pos)
    
#                 # new_target[x, y, :] = color_img[j, i, :]
#             # else:
#             #     new_target[i, j, :] = color_img[i, j, :]

#     # fig, ax = plt.subplots()
#     # # ax.imshow(cv2.cvtColor(new_target, cv2.COLOR_BGR2RGB))
#     # ax.imshow(new_target)
#     # plt.show()

#     return new_target

# def backward_warping(source_region, target_region, img, h_matrix):
#     r, c, _ = source_region.shape
#     # print(source_region.shape)
#     color_img = img
#     new_target = np.copy(target_region)

#     for i in range(r):
#         for j in range(c):
#             if source_region[i, j, 0] == 255 and source_region[i, j, 1] == 255 and source_region[i, j, 2]== 255:

#                 source = np.array([j, i, 1]).reshape(3, 1)

#                 target = np.dot(h_matrix, source)
#                 # f.writelines(["------------------"+str(i)+","+str(j)+"-------------------------", "\n"])
#                 # f.writelines(["["+str(h_matrix)+"]", "\n"])
#                 # f.writelines(["["+str(source)+"]", "\n"])

#                 x = np.round(target[0] / target[2]).astype('int64')
#                 y = np.round(target[1] / target[2]).astype('int64')
#                 # # print(x,",",y)

#                 # f.writelines([" ------------------------------------- ", "\n"])
#                 # # f.writelines(["["+str(i)+","+str(j)+"]", "\n"])
#                 # f.writelines(["["+str(target)+"]", "\n"])
#                 # f.writelines(["***************************************", "\n"])

#                 new_target[i, j, :] = color_img[y, x, :]
#                 # new_target[y, x, :] = color_img[i, j, :]
#             # else:
#             #     new_target[i, j, :] = color_img[i, j, :]

#     # fig, ax = plt.subplots()
#     # # ax.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
#     # ax.imshow(new_target)
#     # plt.show()

#     return new_target 

# -------------------------------------------------------------------- forward_warping

def forward_2warping(Left, Right, img, H_matrix):

    r, c, _ = img.shape
    color_img = np.copy(img)
    polly_img = np.copy(img)

    # new_target = np.copy(Left)
    source_region_R = np.copy(Right)

    for i in range(r):
        for j in range(c):
            if source_region_R[i, j, 0] == 255 and source_region_R[i, j, 1] == 255 and source_region_R[i, j, 2]== 255:

                sourceR = np.array([j, i, 1]).reshape(3, 1)

                target = np.dot(H_matrix, sourceR)
                x = np.round(target[0] / target[2]).astype('int64')
                y = np.round(target[1] / target[2]).astype('int64')

                polly_img[y, x, :] = color_img[i, j, :]
    
    source_region_L = np.copy(Left)
    H_inv = np.linalg.inv(H_matrix)

    for i in range(r):
        for j in range(c):
            if source_region_L[i, j, 0] == 255 and source_region_L[i, j, 1] == 255 and source_region_L[i, j, 2]== 255:

                sourceL = np.array([j, i, 1]).reshape(3, 1)

                target = np.dot(H_inv, sourceL)
                x = np.round(target[0] / target[2]).astype('int64')
                y = np.round(target[1] / target[2]).astype('int64')

                polly_img[y, x, :] = color_img[i, j, :]

    fig, ax = plt.subplots()
    plt.title('2_1.forward_screen',loc ='center')
    ax.imshow(polly_img)
    plt.show()

    # pass
    return polly_img

def Two_forward_2warping(Left, Right, img_L, img_R, H_matrix):

    r, c, _ = img_L.shape

    polly_img_L = copy.deepcopy(img_L)
    polly_img_R = copy.deepcopy(img_R)

    # polly_img = np.copy(img)

    # new_target = np.copy(Left)
    source_region_R = np.copy(Right)

    # fig, ax = plt.subplots()
    # ax.imshow(img_L)
    # plt.show()
    # ------------------------------------------------ right to left
    for i in range(r):
        for j in range(c):
            if source_region_R[i, j, 0] == 255 and source_region_R[i, j, 1] == 255 and source_region_R[i, j, 2]== 255:

                sourceR = np.array([j, i, 1]).reshape(3, 1)

                target = np.dot(H_matrix, sourceR)
                x = np.round(target[0] / target[2]).astype('int64')
                y = np.round(target[1] / target[2]).astype('int64')

                polly_img_L[y, x, :] = img_R[i, j, :]
    
    fig, ax = plt.subplots()
    plt.title('2_2.forward_left_screen',loc ='center')
    ax.imshow(polly_img_L)
    plt.show()
    # ------------------------------------------------ left to right
    source_region_L = np.copy(Left)
    H_inv = np.linalg.inv(H_matrix)

    for i in range(r):
        for j in range(c):
            if source_region_L[i, j, 0] == 255 and source_region_L[i, j, 1] == 255 and source_region_L[i, j, 2]== 255:

                sourceL = np.array([j, i, 1]).reshape(3, 1)

                target = np.dot(H_inv, sourceL)
                x = np.round(target[0] / target[2]).astype('int64')
                y = np.round(target[1] / target[2]).astype('int64')

                polly_img_R[y, x, :] = img_L[i, j, :]   

    fig, ax = plt.subplots()
    plt.title('2_2.forward_right_screen',loc ='center')
    ax.imshow(polly_img_R)
    plt.show()
    # pass
    return polly_img_L, polly_img_R

# -------------------------------------------------------------------- backward_warping

def backward_2warping(Left, Right, img, H_matrix):

    r, c, _ = img.shape
    color_img = np.copy(img)
    polly_img = np.copy(img)

    # new_target = np.copy(Left)
    source_region_R = np.copy(Right)

    for i in range(r):
        for j in range(c):
            if source_region_R[i, j, 0] == 255 and source_region_R[i, j, 1] == 255 and source_region_R[i, j, 2]== 255:

                sourceR = np.array([j, i, 1]).reshape(3, 1)

                target = np.dot(H_matrix, sourceR)
                x = np.round(target[0] / target[2]).astype('int64')
                y = np.round(target[1] / target[2]).astype('int64')

                polly_img[i, j, :] = color_img[y, x, :]
    
    source_region_L = np.copy(Left)
    H_inv = np.linalg.inv(H_matrix)

    for i in range(r):
        for j in range(c):
            if source_region_L[i, j, 0] == 255 and source_region_L[i, j, 1] == 255 and source_region_L[i, j, 2]== 255:

                sourceL = np.array([j, i, 1]).reshape(3, 1)

                target = np.dot(H_inv, sourceL)
                x = np.round(target[0] / target[2]).astype('int64')
                y = np.round(target[1] / target[2]).astype('int64')

                polly_img[i, j, :] = color_img[y, x, :]

    fig, ax = plt.subplots()
    plt.title('2_1.backward_screen',loc ='center')
    ax.imshow(polly_img)
    plt.show()

    return polly_img

def Two_backward_2warping(Left, Right, img_L, img_R, H_matrix):

    r, c, _ = img_L.shape

    polly_img_L = copy.deepcopy(img_L)
    polly_img_R = copy.deepcopy(img_R)

    # color_img = np.copy(img)
    # polly_img = np.copy(img)

    # new_target = np.copy(Left)
    source_region_R = np.copy(Right)

    # fig, ax = plt.subplots()
    # ax.imshow(polly_img_R)
    # plt.show()

    for i in range(r):
        for j in range(c):
            if source_region_R[i, j, 0] == 255 and source_region_R[i, j, 1] == 255 and source_region_R[i, j, 2]== 255:

                sourceR = np.array([j, i, 1]).reshape(3, 1)

                target = np.dot(H_matrix, sourceR)
                x = np.round(target[0] / target[2]).astype('int64')
                y = np.round(target[1] / target[2]).astype('int64')

                polly_img_R[i, j, :] = img_L[y, x, :]

    fig, ax = plt.subplots()
    plt.title('2_2.backward_right_screen',loc ='center')
    ax.imshow(polly_img_R)
    plt.show()
    
    source_region_L = np.copy(Left)
    H_inv = np.linalg.inv(H_matrix)

    for i in range(r):
        for j in range(c):
            if source_region_L[i, j, 0] == 255 and source_region_L[i, j, 1] == 255 and source_region_L[i, j, 2]== 255:

                sourceL = np.array([j, i, 1]).reshape(3, 1)

                target = np.dot(H_inv, sourceL)
                x = np.round(target[0] / target[2]).astype('int64')
                y = np.round(target[1] / target[2]).astype('int64')

                polly_img_L[i, j, :] = img_R[y, x, :]

    fig, ax = plt.subplots()
    plt.title('2_2.backward_left_screen',loc ='center')
    ax.imshow(polly_img_L)
    plt.show()

    return polly_img_L, polly_img_R
    # pass

# def scan_all(left, right, img):
#     r, c, _ = img.shape

#     mask = (left & right)
#     result = np.zeros((r,c,3))
#     for i in range(r):
#         for j in range(c):
#             # if left[i, j, :] == 0 or right[i, j, :] == 0
#             if mask[i][j] == False:
#                 result[i, j, :] = img[i, j, :]

#     fig, ax = plt.subplots()
#     # ax.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
#     ax.imshow(result)
#     plt.show()

#     pass

def runOne(raw_img, left_label, right_label):

    left_pos = np.insert(left_label, 2, values=1, axis=1)   
    # print(left_pos)

    # right
    right_pos = np.insert(right_label, 2, values=1, axis=1) 
    # print(right_pos)

    # # -----------------------------------  homography  -------------------------------
    # Get H matrix
    # homography_matrix(left_pos, right_pos)

    H_matrix = homography_matrix(left_pos, right_pos)
    # H_matrix.astype("float32")
    H_matrix = H_matrix.astype("float")
#     # H_matrix = np.array([[ 2.69633483e-03, 1.47173116e-04,-9.92060028e-01],[3.72990417e-04, 2.06352074e-03,-1.25715123e-01],[ 2.60644874e-06, -5.94548142e-09, 9.59498740e-04]])

# ------------------------------------- pollyfill image --------------------------

    # caculate warping image position
    # get_warping_image_position(gray_img, left_label)
    # get_warping_image_position(raw_img, left_label)
    left_polly_position, left_warp_region = get_warping_image_position(raw_img, left_label)
    right_polly_position, right_warp_region = get_warping_image_position(raw_img, right_label)

    
# --------------------------- forward warping ----------------------------------------
    print("# --------- begin forward --------- #")

    forward_img = forward_2warping(left_warp_region, right_warp_region, raw_img, H_matrix)
    # forward = scan_all(target_left, target_right, raw_img)
    # scan_all(target_left, target_right, raw_img)
# --------------------------- backward warping ----------------------------------------
    print("# --------- begin backward --------- #")

    backward_img = backward_2warping(left_warp_region, right_warp_region, raw_img, H_matrix)

    # backward_img = backward_2warping(left_warp_region, right_warp_region, raw_img, H_matrix)

#     b_target_right = backward_warping(left_warp_region, right_warp_region, raw_img, np.linalg.inv(H_matrix))

    pass

def runTwo(raw_img_L, raw_img_R, left_label, right_label):

    img_L = np.copy(raw_img_L)
    img_R = np.copy(raw_img_R)

    left_pos = np.insert(left_label, 2, values=1, axis=1)   
    # print(left_pos)

    # right
    right_pos = np.insert(right_label, 2, values=1, axis=1) 
    # print(right_pos)

    # # -----------------------------------  homography  -------------------------------
    # Get H matrix
    H_matrix = homography_matrix(left_pos, right_pos)
    # H_matrix.astype("float32")
    H_matrix = H_matrix.astype("float")
#     # H_matrix = np.array([[ 2.69633483e-03, 1.47173116e-04,-9.92060028e-01],[3.72990417e-04, 2.06352074e-03,-1.25715123e-01],[ 2.60644874e-06, -5.94548142e-09, 9.59498740e-04]])

# ------------------------------------- pollyfill image --------------------------

    left_polly_position, left_warp_region = get_warping_image_position(img_L, left_label)
    right_polly_position, right_warp_region = get_warping_image_position(img_R, right_label)

    
# --------------------------- forward warping ----------------------------------------
    # print("# --------- begin forward --------- #")
    # # forward_img = Two_forward_2warping(left_warp_region, right_warp_region, img_L, img_R, H_matrix)
    forward_left_screen, forward_right_screen = Two_forward_2warping(left_warp_region, right_warp_region, img_L, img_R, H_matrix)
    
# # --------------------------- backward warping ----------------------------------------
    print("# --------- begin backward --------- #")
#     backward_img = backward_2warping(left_warp_region, right_warp_region, raw_img, H_matrix)
    back_left_screen, back_right_screen = Two_backward_2warping(left_warp_region, right_warp_region, img_L, img_R, H_matrix)
    # Two_backward_2warping(left_warp_region, right_warp_region, img_L, img_R, H_matrix)

    pass

if __name__ == '__main__':
    
    img_path = './data2/b.jpg'
    # img_path = 'E:/NTHU COURSE/CV_hw/hw2/data3/pic_A_1.png'     
    img = skimage.io.imread(img_path)
    img = np.array(img, dtype=np.uint8)
    raw_img = np.copy(img)

# ------------------------------------   label compose  -------------------------
    # get2Dlabel = np.load('E:/NTHU COURSE/CV_hw/hw2/data3/homo_A.npy')
    get2Dlabel = np.load('./data2/Point2D_11.npy')

    # left
    left_label = get2Dlabel[0:4, :]  
    # print(left_pos)

    # right
    right_label = get2Dlabel[4:8, :]
    # print(right_pos)

    runOne(raw_img, left_label, right_label)

# #################------------------------2 image
    print("# ################################# #")

    print("# --------- run 2 image --------- #")
    img_left = skimage.io.imread('./data2/c_1.jpg')
    img_right = skimage.io.imread("./data2/c_2.jpg")  

    left_label = np.load("./data2/Point2D_21.npy")
    right_label = np.load("./data2/Point2D_22.npy")

    img_left = np.array(img_left, dtype=np.uint8)
    img_right = np.array(img_right, dtype=np.uint8)

    runTwo(img_left, img_right, left_label, right_label)

pass