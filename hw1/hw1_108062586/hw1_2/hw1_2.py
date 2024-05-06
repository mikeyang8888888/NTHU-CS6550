import os
import cv2
import math
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from tone_mapping import tone_mapping
from color_correction import BGR2RGB, RGB2BGR, RGB2XYZ, XYZ2RGB, color_correction
from demosaic_and_mosaic import mosaic, demosaic
from white_balance import generate_wb_mask

# def save_process(img, path, step):
#     # img = 
#     print(path)
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_step_img_path = os.path.join(path, step+'.png')
#     cv2.imwrite(save_step_img_path, img)

def calculate_psnr(img1, img2):
    '''
    Input:
        img1, img2: H*W*3 numpy array
    Output:
        psnr: the peak signal-to-noise ratio value
    '''
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr


def process_one_image(img_path, pattern, step_img_folder_save_path):

    ##############################################################
    #                                                            #
    #                     Camera  Parameters                     #
    #                                                            # 
    ##############################################################

    # curve_path = 'E:/NTHU COURSE/CV_hw/hw1_2/'
    # curve_path = 'E:/NTHU COURSE/CV_hw/hw1_2/curves'
    curve_path = './curves'
    curve_name = os.path.join(curve_path, 'tone_curves.mat')
    # print(curve_name)
    curve_inv_name = os.path.join(curve_path, 'tone_curves_inv.mat')
    tone_curves = scipy.io.loadmat(curve_name)
    tone_curves_inv = scipy.io.loadmat(curve_inv_name)
    
    I = tone_curves['I'] # Irradiance
    B = tone_curves['B'] # Brightness
    I_inv = tone_curves_inv['invI']
    B_inv = tone_curves_inv['invB']

    ccm = np.array([1.0234, -0.2969, -0.2266, 
                    -0.5625, 1.6328, -0.0469, 
                    -0.0703, 0.2188, 0.6406])

    # [[ 2.04720944 -0.59391878 -0.45329066]
    # [-0.54963846  1.59546609 -0.04582763]
    # [-0.08908884  0.27727791  0.81181092]]

    ccm = np.reshape(ccm, (3, 3))
    # ccm = ccm / np.tile(np.sum(ccm, axis=1), [3, 1]).T
    ccm = (ccm / np.tile(np.sum(ccm, axis=1), [3, 1]).T).T
    ccm_inv = np.linalg.inv(np.copy(ccm))

    #### Not Fixed Parameters
    #### You can use the following random code to fit other images

    # tone_index = random.randint(0, 200)
    # fr = random.uniform(0.75, 1)
    # fb = random.uniform(0.75, 1)
    
    # print("****tone,fr,fb*****")
    # print(tone_index)
    # print(fr)
    # print(fb)
    # print("*******************")

    tone_index = 170  
    fr = 0.7715567349551743
    fb = 0.9068480239589546
    

    ##############################################################
    #                                                            #
    #                    Load Image by OpenCV                    #
    #                                                            # 
    ##############################################################

    img = cv2.imread(img_path) 
    img_gt = img
    np.array(img, dtype='uint8')
    #### Normalize the value from [0, 255] to [0, 1]
    img = img.astype('double') / 255.0

    #### Remember that the image store in OpenCV is BGR instead of RGB
    #### We should transfer to RGB first before ISP
    img = BGR2RGB(img)
    # print(step_img_folder__save_path)    
    save_step_img_path = os.path.join(step_img_folder_save_path, '6.origin.png')
    cv2.imwrite(save_step_img_path, np.copy(img).astype("float32")* 255.0)

    # save_process(step_img_folder__save_path, img, "origin")

    # cv2.imshow("windows_name", img)
    # cv2.waitKey (0) 

    ##############################################################
    #                                                            #
    #                     Inverse ISP Process                    #
    #                                                            # 
    ##############################################################

    # print("1. Inverse Tone Mapping")
    img = tone_mapping(img, I_inv, B_inv, index=tone_index, inv=True)

    # fig1, ax1 = plt.subplots()
    # ax1.imshow(cv2.cvtColor(np.copy(img).astype("float32"), cv2.COLOR_BGR2RGB))
    # plt.show()

    save_step_img_path1 = os.path.join(step_img_folder_save_path, '5.tone_mapping.png')
    cv2.imwrite(save_step_img_path1, np.copy(img).astype("float32")* 255.0)

    # cv2.imshow("windows_name1", img)
    # cv2.waitKey (0)

    # print("2. from RGB to CIE XYZ")
    img = RGB2XYZ(img)

    save_step_img_path2 = os.path.join(step_img_folder_save_path, '4.RGB2XYZ.png')
    cv2.imwrite(save_step_img_path2, np.copy(img).astype("float32")* 255.0)
    # cv2.imshow("windows_name2", img)
    # cv2.waitKey (0)

    # print("3. Color Correction")
    img = color_correction(img, ccm)

    save_step_img_path3 = os.path.join(step_img_folder_save_path, '3.color_correction.png')
    cv2.imwrite(save_step_img_path3, np.copy(img).astype("float32")* 255.0)
    # cv2.imshow("windows_name3", img)
    # cv2.waitKey (0)

    # print("4. Mosaic")
    img = mosaic(img, pattern=pattern)

    save_step_img_path4 = os.path.join(step_img_folder_save_path, '2.mosaic.png')
    cv2.imwrite(save_step_img_path4, np.copy(img).astype("float32")* 255.0)

    # print(img.shape, img.dtype)
    # cv2.imshow("windows_name4", img)
    # cv2.waitKey (0)

    #print("5: Inverse AWB")
    wb_mask = generate_wb_mask(img, pattern, fr, fb)

    # print(wb_mask)
    # cv2.imshow("windows_name5", img)
    # cv2.waitKey (0)

    img = img * wb_mask
    save_step_img_path5 = os.path.join(step_img_folder_save_path, '1.Inverse AWB.png')
    cv2.imwrite(save_step_img_path5, np.copy(img).astype("float32")* 255.0)

    # cv2.imshow("windows_name6", img)
    # cv2.waitKey (0)

    
    ##############################################################
    #                                                            #
    #                         ISP Process                        #
    #                                                            # 
    ##############################################################
    
    #print("1. AWB")
    wb_mask = generate_wb_mask(img, pattern, 1/fr, 1/fb)
    img = img * wb_mask
    img = np.clip(img, 0, 1)

    # #print("2. Demosaic")
    img = demosaic(img, pattern=pattern)

    # #print("3. Color Correction")
    img = color_correction(img, ccm_inv)

    # #print("4. from XYZ to RGB")
    img = XYZ2RGB(img)

    # #print("5. Tone Mapping")
    img = tone_mapping(img, I, B, index=tone_index, inv=False)
    
    img = RGB2BGR(img) * 255.0
    
    return img_gt, img
    

if __name__ == '__main__':
    #### If you want to test more image, remember to change the settings here
    img_num = 1
    # img_folder_path = 'images'
    # img_folder_save_path = 'outputs'

    # img_folder_path = 'E:/NTHU COURSE/CV_hw/hw1_2/images'
    # img_folder_save_path = 'E:/NTHU COURSE/CV_hw/hw1_2/outputs'

    img_folder_path = './images'
    img_folder_save_path = './outputs'

    step_img_folder_save_path = './stepouts'
    if not os.path.exists(step_img_folder_save_path):
        os.makedirs(step_img_folder_save_path)
    
    for idx in range(img_num):
        img_path = os.path.join(img_folder_path, str(idx+1)+'.png')
        # print(img_path)

        for pattern_index in range(4):
            ##############################################################
            #                                                            #
            #              Processed by Each Bayer Pattern               #
            #                                                            # 
            ##############################################################
            print(pattern_index)
            print("--------------begin----------")
            if pattern_index == 0:
                pattern = 'GRBG'
            elif pattern_index == 1:
                pattern = 'RGGB'
            elif pattern_index == 2:
                pattern = 'GBRG'
            elif pattern_index == 3:
                pattern = 'BGGR'
            
            gt, pred = process_one_image(img_path, pattern, step_img_folder_save_path)

            # process_one_image(img_path, pattern)
        
            ##############################################################
            #                                                            #
            #                  Save and Calculate PSNR                   #
            #                                                            # 
            ##############################################################
        
            #### Save Results
            save_name = str(idx+1)+'_'+pattern+'.png'
            if not os.path.exists(img_folder_save_path):
                os.makedirs(img_folder_save_path)
            save_img_path = os.path.join(img_folder_save_path, save_name)
            cv2.imwrite(save_img_path, pred)
            
            #### Calculate PSNR
            print("{}: {}".format(pattern ,calculate_psnr(pred, gt)))

        print('') # for next image