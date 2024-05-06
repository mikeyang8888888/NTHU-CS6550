####################################
##	lab 1 Camera Calibration
###################################

Run python3 hw2_1.py


This python file will generate a file named output.txt which contained all detail processing information.

#---------------------

image path:
	img = skimage.io.imread('./data/chessboard_1.jpg')
	img2 = skimage.io.imread('./data/chessboard_2.jpg')
    

point path:
	get2Dlabel = np.load('./data/Point2D.npy')
	get2Dlabel2 = np.load('./data/Point2D_2.npy')

*****
	** bouns file包含自行拍攝的圖片及座標點
*****

####################################
##	lab 2 Homography transformation
###################################

Run python3 hw2_2.py