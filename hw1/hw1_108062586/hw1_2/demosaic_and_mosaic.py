
import numpy as np

from demosaic_2004 import demosaicing_CFA_Bayer_Malvar2004

def mosaic(img, pattern):
    '''
    Input:
        img: H*W*3 numpy array, input image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W numpy array, output image after mosaic.
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create the H*W output numpy array.                              #   
    #   2. Discard other two channels from input 3-channel image according #
    #      to given Bayer pattern.                                         #
    #                                                                      #
    #   e.g. If Bayer pattern now is BGGR, for the upper left pixel from   #
    #        each four-pixel square, we should discard R and G channel     #
    #        and keep B channel of input image.                            #     
    #        (since upper left pixel is B in BGGR bayer pattern)           #
    ########################################################################

    # print(img.shape)
    # print(pattern)
    # print("========================")
    h, w, depth = img.shape
    # print(w)
    # output = np.array((w,h))
    # print(r)

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    output = np.zeros((h,w), dtype=np.float)

    if pattern == "GRBG":
        # print(pattern)
        # print("1")
        for i in range(h):
            for j in range(w):
                if (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
                    output[i,j]  = g[i,j]
                if (i%2 == 0 and j%2 == 1):
                    output[i,j]  = r[i,j]
                if i%2 == 1 and j%2 == 0:
                    output[i,j]  = b[i,j]                    

    if pattern == "RGGB":
        # print(pattern)
        # print("2")
        for i in range(h):
            for j in range(w):
                if i%2 == 0 and j%2 == 0:
                    output[i,j]  = r[i,j]
                if (i%2 == 1 and j%2 == 0) or (i%2 == 0 and j%2 == 1):
                    output[i,j]  = g[i,j]
                if i%2 == 1 and j%2 == 1:
                    output[i,j]  = b[i,j] 

    if pattern == "GBRG":
        # print(pattern)
        # print("3")
        for i in range(h):
            for j in range(w):
                if (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
                    output[i,j]  = g[i,j]
                if (i%2 == 0 and j%2 == 1):
                    output[i,j]  = b[i,j]
                if i%2 == 1 and j%2 == 0:
                    output[i,j]  = r[i,j]
                  
    if pattern == "BGGR":
        for i in range(h):
            for j in range(w):
                if i%2 == 0 and j%2 == 0:
                    output[i,j]  = b[i,j]
                if (i%2 == 1 and j%2 == 0) or (i%2 == 0 and j%2 == 1):
                    output[i,j]  = g[i,j]
                if i%2 == 1 and j%2 == 1:
                    output[i,j]  = r[i,j] 

    # x, y = output.shape
    # print(x)
    # r, g, b = cv2.split(img)

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################

    return output

def demosaic(img, pattern):
    '''
    Input:
        img: H*W numpy array, input RAW image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W*3 numpy array, output de-mosaic image.
    '''
    #### Using Python colour_demosaicing library
    #### You can write your own version, too
    output = demosaicing_CFA_Bayer_Malvar2004(img, pattern)
    output = np.clip(output, 0, 1)

    return output

