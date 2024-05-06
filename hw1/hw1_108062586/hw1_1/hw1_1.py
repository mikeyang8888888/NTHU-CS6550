import os
import cv2
import math
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import skimage

def write_file(saveimg, filename, stage):
    img = saveimg
    # img_folder_save_path = 'E:/NTHU COURSE/CV_hw/hw1_1/results/'+stage
    img_folder_save_path = './process_results/'+stage
    if not os.path.exists(img_folder_save_path):
        os.makedirs(img_folder_save_path)

    outputImg = img.astype('float32') * 255.0
    save_name = filename + '.png'
    # print(save_name)
    save_img_path = os.path.join(img_folder_save_path, save_name)
    cv2.imwrite(save_img_path, outputImg)

# class Harris_Corner:
def gaussian_smooth(img, size, sigma=5):
    print("begin gaussian_smooth")

    sigma = sigma
    kernal_size = size
    # gaussian_kernel = np.exp(-(5**2+5**2))
    gaussian_kernel = np.zeros((kernal_size,kernal_size), dtype=np.float32)
    # print(gaussian_kernel)

    kernal_const = 1 / (2 * np.pi * (np.power(sigma,2)))
    # a = np.exp( -( (np.power(0,2) + np.power(0,2)) / (2 * (np.power(sigma,2)) )) )
    # print(a)
    # print(kernal_cost)

    i, j = 0, 0
    x,y = 0, 0

    if kernal_size%2 == 0:
         for x in range((-(kernal_size//2)), (kernal_size//2)):
            for y in range((-(kernal_size//2)), (kernal_size//2)):
                # print(x,y)
                # print(j)
                kernal_val = kernal_const * (np.exp( -( (np.power(x,2) + np.power(y,2)) / (2 * (np.power(sigma,2)) )) ))
                gaussian_kernel[i][j] = kernal_val
                # gaussian_kernel[x][y] = kernal_val
                j += 1

            i, j = i+1, 0
    else:
        for x in range((-(kernal_size//2)), (kernal_size//2)+1):
            for y in range((-(kernal_size//2)), (kernal_size//2)+1):
                # print(x,y)
                # print(j)
                kernal_val = kernal_const * (np.exp( -( (np.power(x,2) + np.power(y,2)) / (2 * (np.power(sigma,2)) )) ))
                gaussian_kernel[i][j] = kernal_val
                # gaussian_kernel[x][y] = kernal_val
                j += 1

            i, j = i+1, 0
    # print(gaussian_kernel)

    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # print(gaussian_kernel)
    outputImg = cv2.filter2D(img, -1, gaussian_kernel)
    # cv2.imshow("windows_name", outputImg)
    # cv2.waitKey (0) 

    return outputImg

def sobel_edge_detection(img, threshold_m=1e-7):
    print("begin sobel_edge_detection")

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.round((0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float), decimals=2)

    # masksize = 3
    # mask = np.zeros((masksize,masksize), dtype=int8)
    # # generate mask value
    # for i in range(masksize):
    #     for j in range(masksize):

    # sobel_mask_x = np.zeros((masksize,masksize), dtype=int8)
    # sobel_mask_y = np.zeros((masksize,masksize), dtype=int8)
# ----------------------------------------------------------------------

    sobel_mask_x1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float32)
    sobel_mask_y1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float32)

    sobel_mask_x = (1/8) * sobel_mask_x1
    sobel_mask_y = (1/8) * sobel_mask_y1

    # sobel_mask_x = sobel_mask_x1
    # sobel_mask_y = sobel_mask_y1

    # sobel_mask_8x = np.dot(sobel_mask_x,(1/8))
    # sobel_mask_8y = np.dot(sobel_mask_y, (1/8))

    outputImg_x = cv2.filter2D(gray_img, -1, sobel_mask_x)
    outputImg_y = cv2.filter2D(gray_img, -1, sobel_mask_y)

    # outputImg_8x = np.dot(outputImg_x,(1/8))
    # outputImg_8y = np.dot(outputImg_y, (1/8))

    # fig, ax = plt.subplots()
    # ax.imshow(outputImg_x)
    # # plt.show()

    # fig1, ax1 = plt.subplots()
    # ax1.imshow(outputImg_y)
    # plt.show()

    # cv2.imshow("windows_name1", outputImg_x)
    # cv2.waitKey (0) 

    # cv2.imshow("windows_name1", outputImg_y)
    # cv2.waitKey (0) 

    # sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    h, w = outputImg_x.shape

    magnitude_img = np.zeros((h,w), dtype=np.float32)
    # magnitude_img = np.sqrt( (np.power(outputImg_x, 2)) + (np.power(outputImg_y, 2)) )
    # threshold_m = (threshold_m/255.0)/8
    for i in range(h):
        for j in range(w):
            threshold = np.sqrt((np.power(outputImg_x[i][j], 2) + np.power(outputImg_y[i][j], 2)))
            if(threshold > threshold_m): # 80 0.314
                # magnitude_img[i][j] = 1.0
                magnitude_img[i][j] = threshold
            else:
                magnitude_img[i][j] = 0.0

    fig, ax = plt.subplots()
    plt.title('magnitude_img',loc ='center')
    ax.imshow(magnitude_img, cmap = "gray")
    plt.show()

    # print(magnitude_img)
    # cv2.imshow("windows_name1", magnitude_img)
    # cv2.waitKey (0) 

    direction_img = np.zeros((h,w), dtype=np.float32)
    direction_img = np.arctan2( outputImg_y, outputImg_x )

    hsv = np.zeros((h, w, 3))
    hsv[..., 0] = (np.arctan2(outputImg_x, outputImg_y) + np.pi) / (2 * np.pi)
    hsv[..., 1] = np.ones((h,w))
    hsv[..., 2] = (magnitude_img - magnitude_img.min()) / (magnitude_img.max() - magnitude_img.min())
    hsv_colormap = skimage.color.hsv2rgb(hsv)

    fig, ax = plt.subplots()
    plt.title('direction',loc ='center')
    ax.imshow(hsv_colormap, cmap = "rainbow")
    plt.show()

    # magnitude_img = np.zeros((h,w), dtype=np.float)
    # magnitude_img = np.sqrt( (np.power(outputImg_x, 2)) + (np.power(outputImg_y, 2)) )

    # print(h)
    # print(w)

    # magnitude_img = np.zeros((h,w), dtype=np.float)
    # magnitude_img = np.sqrt( (np.power(outputImg_x, 2)) + (np.power(outputImg_y, 2)) )

    # magnitude_img = np.sqrt(np.power((outputImg_x + outputImg_y), 2))
    # for i in range(h):
    #     for j in range(w):
    #         magnitude_img[i][j] = np.sqrt(np.power(outputImg_x[i][j], 2)+np.power(outputImg_y[i][j], 2))
    #         # print(np.power(outputImg_x[i][j])
    #         # magnitude_img = outputImg_x[i][j]

    # cv2.imshow("windows_name", outputImg_x)
    # cv2.waitKey (0)

    # cv2.imshow("windows_name1", magnitude_img)
    # cv2.waitKey (0) 
    # print(magnitude_img)

# -----------------------------------------------

    return outputImg_x, outputImg_y, magnitude_img, hsv_colormap

def structure_tensor(original_img, Ix, Iy, size=3, threshold_k=0.05):
    print("begin structure_tensor")
# [ [Ix*Ix] , [Ix*Iy]
#   [Iy*Ix], [Iy*Iy] ]

    # # ad_Ix = Ix * (1/8)
    # # ad_Iy = Iy * (1/8)
    # struct_tensor = np.zeros((2,2), dtype=np.float)
    # # struct_tensor[0][0] = np.power(Ix, 2)
    # # struct_tensor[0][1] = Ix*Iy
    # # struct_tensor[1][0] = Iy*Ix
    # # struct_tensor[1][1] = Iy*Iy
    # # lamda1 = np.linalg.det(struct_tensor)

    # # lamda2 = struct_tensor[0][0] + struct_tensor[1][1]
    # print(M_tensor)

    kernel_size = size
    Ixx = gaussian_smooth(Ix * Ix, kernel_size)
    Ixy = gaussian_smooth(Ix * Iy, kernel_size)
    Iyy = gaussian_smooth(Iy * Iy, kernel_size)
    
    k = threshold_k

    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    struct_r = det - (k * trace * trace)
    # ------------ corner map without nms ---------

    h, w = Ix.shape
    struct_map = np.zeros((h,w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            if struct_r[i][j] > 1e-7:
                # struct_map[i][j] = original_img[i][j]
                struct_map[i][j] = 1.0
    # print(struct_r)
    fig, ax = plt.subplots()
    plt.title('structure_tensor_kernel:'+str(size)+"x"+str(size),loc ='center')
    ax.imshow(struct_map, cmap = "gray")
    plt.show()

    return struct_r, struct_map

def nms(img, R, r_threshold=1e-7, kernel_size=30):
    print("begin nms")
    mask1 = (R > r_threshold)
    # avoid float error
    mask2 = (np.abs(scipy.ndimage.maximum_filter(R, size=kernel_size) - R) < 1e-8)

    mask = (mask1 & mask2)
    r, c = np.where(mask == True)
    corner_pos_num = r.shape[0]

    corner_pos = np.copy(img).astype("float32")
    corner_pos_gray = np.copy(R).astype("float32")
    # print(corner_pos.shape)
    h, w, d = img.shape
    # corner_pos = np.zeros((h,w), dtype=np.float32)
    # count_point = 0
    for i in range(h):
        for j in range(w):
            # corner_pos[i][j] = 1
            if mask[i][j] == True:
                # map to original image
                corner_pos[i, j, 0] = 0.0
                corner_pos[i, j, 1] = 0.0
                corner_pos[i, j, 2] = 1.0
                
                # count_point += 1

    # count_point2 = 0
    for i in range(h):
        for j in range(w):
            # corner_pos[i][j] = 1
            if mask[i][j] == True:
                # gray
                corner_pos_gray[i][j] = 1.0
                # count_point2 += 1
            else:
                corner_pos_gray[i][j] = 0.0
    # r, c = np.where(R > 0.001)
    # print(mask)
    # r, c = mask.shape
    fig, ax = plt.subplots()
    plt.title('corner_after_nms_color #corner:'+str(corner_pos_num),loc ='center')
    plt.plot(c, r, "ro", markersize=3)
    ax.imshow(cv2.cvtColor(corner_pos, cv2.COLOR_BGR2RGB))
    # ax.imshow(corner_pos, cmap="rainbow")
    plt.show()

    # fig, ax = plt.subplots()
    # plt.title('corner_after_nms_gray/'+'#corner:'+str(count_point2),loc ='center')
    # ax.imshow(corner_pos_gray, cmap = "gray")
    # plt.show()

    return corner_pos, corner_pos_gray
    # return R

def run(img, process, gaussian_size = 10, s_tensor_windowSize = 30):

    gaussian_kernel_size = gaussian_size
    structure_tensor_kernel_size = s_tensor_windowSize

    img = gaussian_smooth(img, gaussian_kernel_size)

    fig, ax = plt.subplots()
    plt.title('gaussian_smooth_kernel:'+str(gaussian_size)+"x"+str(gaussian_size) ,loc ='center')
    ax.imshow(cv2.cvtColor(np.copy(img).astype("float32"), cv2.COLOR_BGR2RGB))
    plt.show()

    write_file(img, "1. gaussian_smooth_image_ksizeis"+str(gaussian_kernel_size), process)

    g_x, g_y, sobel_img, hsv_img = sobel_edge_detection(img)
    # write_file(g_x, "2.1. sobel_x")
    # write_file(g_y, "2.2. sobel_y")
    write_file(sobel_img, "2. magnitude", process)
    write_file(hsv_img, "3. hsv_gradient_colormap", process)

    R, struct_map = structure_tensor(sobel_img, g_x, g_y, structure_tensor_kernel_size)
    write_file(struct_map, "4. struct_tensor", process)
    # nms_img = nms(img, structure_img)
    # nms(img, R)
    corner_img, corner_img_gray = nms(img, R)
    write_file(corner_img, "5. corner_afterNms", process)
    # write_file(corner_img_gray, "5.1. corner_afterNms_gray", process)
    print("------------ "+ process +"_finish ----------")

if __name__ == '__main__':

    # img_path = 'E:/NTHU COURSE/CV_hw/hw1_1/original.jpg'
    img_path = './original.jpg'
    # img_folder_save_path = 'E:/NTHU COURSE/CV_hw/hw1_1/results'
    # print(img_path)

    img = cv2.imread(img_path) 
    img_gt = img
    img = np.array(img, dtype=np.uint8)
    # np.array(img, dtype='uint8')
    #### Normalize the value from [0, 255] to [0, 1]
    img = img.astype('float32') / 255.0

    process_img = np.copy(img)
    rotate_img = np.copy(img)
    scale_img = np.copy(img)
    # rotate 30
    rotate_img = skimage.transform.rotate(rotate_img, 30)
    # scale 0.5x
    scale_img = skimage.transform.rescale(scale_img, 0.5, multichannel=True)

    # #########################################
    #                process
    # #########################################
    h, w, depth = process_img.shape

    run(process_img, "original")
    run(rotate_img, "rotate")
    run(scale_img, "scale")

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian_kernel_size = 5
    # structure_tensor_kernel_size = 5

    # img = gaussian_smooth(img, gaussian_kernel_size)
    # write_file(img, "1. gaussian_smooth_image_ksizeis"+str(gaussian_kernel_size))

    # g_x, g_y, sobel_img, hsv_img = sobel_edge_detection(img)
    # write_file(g_x, "2_1. sobel_x")
    # write_file(g_y, "2_2. sobel_y")
    # write_file(sobel_img, "2. magnitude")
    # write_file(hsv_img, "3. hsv_gradient_colormap")

    # R, struct_map = structure_tensor(sobel_img, g_x, g_y, structure_tensor_kernel_size)
    # write_file(struct_map, "4. struct_tensor")
    # # nms_img = nms(img, structure_img)
    # # nms(img, R)
    # corner_img = nms(img, R)
    # write_file(corner_img, "5. corner_afterNms")

    # outputImg = corner_img

    # x = cv2.Sobel(img,cv2.CV_16S,1,0)
    # y = cv2.Sobel(img,cv2.CV_16S,0,1)    
    # absX = cv2.convertScaleAbs(x)   # 转回uint8
    # absY = cv2.convertScaleAbs(y)    
    # dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    # outputImg = grady_y.astype('float32') * 255.0
    # outputImg = img

    # #########################################
    #                save pic
    # #########################################

    # save_name = 'result5.png'
    # save_img_path = os.path.join(img_folder_save_path, save_name)
    # cv2.imwrite(save_img_path, outputImg)
    print("------- finish ---------")

pass
