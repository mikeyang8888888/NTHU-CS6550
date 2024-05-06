
import numpy as np

def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################

    h, w= img.shape

    mask = np.empty((h,w), dtype=np.float)

    # if pattern == "GRBG":
    #     for i in range(w):
    #         for j in range(h):
    #             if (i%2 == 0 and j%2 == 0) or (i%2 != 0 and j%2 != 0):
    #                 mask[i,j]  = 1
    #             if (i%2 == 1 and j%2 != 0):
    #                 mask[i,j]  = fr
    #             if i%2 != 1 and j%2 == 0:
    #                 mask[i,j]  = fb
    # if pattern == "RGGB":
    #     # print(pattern)
    #     # print("2")
    #     for i in range(w):
    #         for j in range(h):
    #             if i%2 == 0 and j%2 == 0:
    #                 mask[i,j]  = fr
    #             if (i%2 == 1 and j%2 != 0) or (i%2 != 1 and j%2 == 0):
    #                 mask[i,j]  = 1
    #             if i%2 != 0 and j%2 != 0:
    #                 mask[i,j]  = fb   

    # if pattern == "GBRG":
    #     # print(pattern)
    #     # print("3")
    #     for i in range(w):
    #         for j in range(h):
    #             if (i%2 == 0 and j%2 == 0) or (i%2 != 0 and j%2 != 0):
    #                 mask[i,j]  = 1
    #             if (i%2 == 1 and j%2 != 0):
    #                 mask[i,j]  = fb
    #             if i%2 != 1 and j%2 == 0:
    #                 mask[i,j]  = fr   
                  
    # if pattern == "BGGR":
    #     for i in range(w):
    #         for j in range(h):
    #             if i%2 == 0 and j%2 == 0:
    #                 mask[i,j]  = fb
    #             if (i%2 == 1 and j%2 != 0) or (i%2 != 1 and j%2 == 0):
    #                 mask[i,j]  = 1
    #             if i%2 != 0 and j%2 != 0:
    #                 mask[i,j]  = fr


    if pattern == "GRBG":
        # print(pattern)
        # print("1")
        for i in range(h):
            for j in range(w):
                if (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
                    mask[i,j]  = 1
                if (i%2 == 0 and j%2 == 1):
                    mask[i,j]  = fr
                if i%2 == 1 and j%2 == 0:
                    mask[i,j]  = fb                    

    if pattern == "RGGB":
        # print(pattern)
        # print("2")
        for i in range(h):
            for j in range(w):
                if i%2 == 0 and j%2 == 0:
                    mask[i,j]  = fr
                if (i%2 == 1 and j%2 == 0) or (i%2 == 0 and j%2 == 1):
                    mask[i,j]  = 1
                if i%2 == 1 and j%2 == 1:
                    mask[i,j]  = fb 

    if pattern == "GBRG":
        # print(pattern)
        # print("3")
        for i in range(h):
            for j in range(w):
                if (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
                    mask[i,j]  = 1
                if (i%2 == 0 and j%2 == 1):
                    mask[i,j]  = fb
                if i%2 == 1 and j%2 == 0:
                    mask[i,j]  = fr
                  
    if pattern == "BGGR":
        for i in range(h):
            for j in range(w):
                if i%2 == 0 and j%2 == 0:
                    mask[i,j]  = fb
                if (i%2 == 1 and j%2 == 0) or (i%2 == 0 and j%2 == 1):
                    mask[i,j]  = 1
                if i%2 == 1 and j%2 == 1:
                    mask[i,j]  = fr          


    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
        
    return mask