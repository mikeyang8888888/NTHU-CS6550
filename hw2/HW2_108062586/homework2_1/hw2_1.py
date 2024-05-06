import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy
import visualize as vi

f = open("output.txt", "w")
rawString = ""

def get_P(Ori_position, Twod_Position): #3d, 2d
    
    # x2, y2 = Twod_Position.shape
    # print(x2,",",y2)
    # print("------------------------------------------")
    # x3, y3 = Ori_position.shape
    # print(x3,",",y3)
    # print(Twod_Position)
    # print("---------")

    # #############################################
    #               get p matrix
    # #############################################

    # X_i = (Ori_position[0:4, :]).T #3d    
    # x_prime_i = (Twod_Position[0:4, :]).T #2d
    X_i = Ori_position.T
    x_prime_i = Twod_Position.T
    row, col = X_i.shape
    # final_p = np.zeros((col*2,12))

    # print(X_i)
    # print("------------------------------------------")
    # print(x_prime_i)    

    for i in range(col):
        X = X_i[0, i]
        Y = X_i[1, i]
        Z = X_i[2, i]        
        u = x_prime_i[0, i]
        v = x_prime_i[1, i]        
        # print(X,",",Y,",",Z,",",u,",",v)

        xX = np.dot(u, X_i[0, i])
        xY = np.dot(u, X_i[1, i])
        xZ = np.dot(u, X_i[2, i])
        yX = np.dot(v, X_i[0, i])
        yY = np.dot(v, X_i[1, i])
        yZ = np.dot(v, X_i[2, i])
        # P = np.array([X, Y, Z, 1, 0, 0, 0, 0, -xX, -xY, -xZ, -X])
        p_flat = np.array([X, Y, Z, 1, 0, 0, 0, 0, -xX, -xY, -xZ, -u, 0, 0, 0, 0, X, Y, Z, 1, -yX, -yY, -yZ, -v])
        # print('-------p_flat--------')
        # print(p_flat)
        p = np.reshape(p_flat, (2,-1))
        # print('-------p--------')
        # print(p)

        # p = np.array([X, Y, Z, 1, 0, 0, 0, 0, -xX, -xY, -xZ, -u],
        #             [0, 0, 0, 0, X, Y, Z, 1, -yX, -yY, -yZ, -v])
        if i == 0:
            final_p = np.reshape(p_flat, (2,-1))
        else:
            final_p = np.append(final_p, p, axis = 0)
   
    # #############################################
    #               caculate p singular value
    # #############################################
    # # print(final_p.shape)
    # # eigenvalue = np.linalg.eigvals(np.dot(final_p.T, final_p))
    # eigenvalue, eigenvectors = np.linalg.eig(np.dot(final_p.T, final_p))
    # # print(eigenvalue,",",eigenvectors)
    # # print(eigenvalue)
    # # print(eigenvectors)
    # # print("--------------------------------")
    # search_min_index = np.where( eigenvalue == eigenvalue.min())
    # # print(search_min[0])
    # # print(eigenvectors[search_min_index[0], :])
    # P = eigenvectors[search_min_index[0], :]
    # # print(eigenvectors.shape)
    # print(final_p[30:35, :])
    # print(final_p.shape)

    eigenvalue, eigenvectors = np.linalg.eig(np.dot(final_p.T, final_p))

    # print('----------------eigenvalue----------------')
    # print(eigenvalue)
    # print('----------------eigenvectors----------------')
    # print(eigenvectors)
    # print("--------------------------")

    search_min_index = np.where( eigenvalue == eigenvalue.min())
    # print(search_min_index[0])
    # P = eigenvectors[search_min_index[0], :]
    P = eigenvectors[:, search_min_index[0]]
    P = np.reshape(P, (3,4))

    print(' ------------------ projection matrix ------------------ ')
    print(P)

    f.writelines([" ------------------ projection matrix ------------------ ", "\n"])
    f.writelines([str(P), "\n"])
    # print("--------------------------")
    return P

def decompose_p(matrix_p):
    # a = matrix_p[:, ::2]    
    # print(matrix_p.shape)
    # # print("------------")
    # sub_matrix = np.zeros((3,3), dtype="float32")
    # for i in range(2):
    #     for j in range(2):
    #         sub_matrix[i][j] = matrix_p[i][j]

    # print(matrix_p[:, 0:3])
    # print(matrix_p)
    # print("---------------")
    # print(matrix_p[:,3])

    r, q = scipy.linalg.rq(matrix_p[:, 0:3])
    D = np.diag(np.sign(np.diag(r))) #sign 正負
    K = np.dot(r,D)
    R = np.dot(D,q)
    # compute matrix t
    t = np.dot(np.linalg.inv(K), matrix_p[:, 3])

    # # normalize
    K = np.divide(K,K[2][2])
    # r = r/r[-1, -1]

    print(" ------------------ K ------------------ ")
    print(K)
    print(" ------------------ R ------------------ ")
    print(R)
    print(" ------------------ t ------------------ ")
    print(t)

    f.writelines([" ------------------ K ------------------ ", "\n"])
    f.writelines([str(K), "\n"])
    f.writelines([" ------------------ R ------------------ ", "\n"])
    f.writelines([str(R), "\n"])
    f.writelines([" ------------------ t ------------------ ", "\n"])
    f.writelines([str(t), "\n"])
    # print("---------------------------------------")

    return K, R, t

def reproject(Ori_position, k, q):
    # print(Ori_position)
    P = np.dot(k, q)
    # print(P)    
    X_i = Ori_position.T
    # print(X_i)

    twoD_matrix = np.dot(P, X_i).T
    # twoD_matrix = np.dot(P, X_i)
    # print(twoD_matrix)
    # print(twoD_matrix)
    row, col = twoD_matrix.shape
    # print(row,",",col)
    # print("---------------------")

    # normalize
    for i in range(row):
        twoD_matrix[i, :] = twoD_matrix[i, :] / twoD_matrix[i, 2]
    
    # pos_x = twoD_matrix[0] / twoD_matrix[2]
    # pos_y = twoD_matrix[1] / twoD_matrix[2]
    # print("----------- x,y ----------")
    # print(pos_x)
    # print(pos_y)

    # # print(twoD_matrix[0:2, :])
    # # print(twoD_matrix[:, 0:2])
    uv_position = np.around(twoD_matrix[:, 0:2])    
    # # print("---------------------")
    # # print(uv_position[0:2, :])

    # # return pos_x, pos_y
    return uv_position

def RMSE(predictions, targets):
    error = np.sqrt(np.mean((predictions - targets)**2))
    # print(error)
    return error

def plot(trans_img, handcraft_label, reproject_label):

    fig, ax = plt.subplots()
    # plt.title('corner_after_nms_color #corner:'+str(corner_pos_num),loc ='center')
    c = handcraft_label[:, 0]
    r = handcraft_label[:, 1]

    c_reproject = reproject_label[:, 0]
    r_reproject = reproject_label[:, 1]

    ax.plot(c, r, "ro", markersize=3)
    ax.plot(c_reproject, r_reproject, "yo", markersize=6, markerfacecolor="none")
    ax.imshow(trans_img)
    plt.show()

    # return pic

def run(img, position_2d, position_3d):

    trans_img = np.array(img, dtype=np.uint8)
    # np.array(img, dtype='uint8')
    #### Normalize the value from [0, 255] to [0, 1]
    trans_img = trans_img.astype('float32') / 255.0

    # xi
    get2Dlabel = position_2d
    # print(get2Dlabel)
    label_position = np.insert(get2Dlabel, 2, values=1, axis=1)   

    # Xi
    get3Dlabel = position_3d
    Ori_position = np.insert(get3Dlabel, 3, values=1, axis=1)
    # x3, y3 = get3Dlabel.shape
    # print(Ori_position)    
    # print(get3Dlabel)
    print("# --------- begin compute the matrix P --------- #")
    f.writelines(["# --------- begin compute the matrix P --------- #", "\n"])
    # compute the matrix P
    P_matrix = get_P(Ori_position, label_position)
    print("# --------- begin decompose P --------- #")
    f.writelines(["# --------- begin decompose P --------- #", "\n"])
    # decompose P
    K, R, t = decompose_p(P_matrix)
    t = np.reshape(t, (3,1))
    Q = np.append(R, t, axis = 1)
    # print(Q)
    print("# --------- begin Reprojection --------- #")
    f.writelines(["# --------- begin Reprojection --------- #", "\n"])
    # Reprojection
    # reproject(Ori_position ,K ,Q)
    reproject_position = reproject(Ori_position ,K ,Q)
    # print(" --------- reproject_position --------- ")
    # print(reproject_position)

    # rmse
    error = RMSE(get2Dlabel, reproject_position)

    # plot
    plot(trans_img, get2Dlabel, reproject_position)

    # pic = plot(trans_img, get2Dlabel, reproject_position)

    # img = skimage.io.imread('E:/NTHU COURSE/CV_hw/hw2/data/chessboard_1.jpg')
    # p = np.zeros()
    # get2Dlabel = np.dot(p, Ori_position)

    # fig, ax = plt.subplots()
    # ax.imshow(reproject_position)
    # plt.show()

    return position_3d, R, t, error

if __name__ == '__main__':
    
    # xi
    # picture1
    img = skimage.io.imread('./data/chessboard_1.jpg')
    get2Dlabel = np.load('./data/Point2D.npy')

    # picture2
    img2 = skimage.io.imread('./data/chessboard_2.jpg')
    get2Dlabel2 = np.load('./data/Point2D_2.npy')

    # Xi
    get3Dlabel = np.loadtxt('./data/Point3D.txt')

    print("************** pic 1 *****************")
    f.writelines(["************** pic 1 *****************", "\n"])

    # run(img, get2Dlabel, get3Dlabel)
    position_3d1, R1, t1, error1 = run(img, get2Dlabel, get3Dlabel)
    print("chessboard_1 error : "+str(error1))
    f.writelines(["chessboard_1 error : "+str(error1), "\n"])

    print("************** pic 2 *****************")
    f.writelines(["************** pic 2 *****************", "\n"])

    # run(get2Dlabel2, get3Dlabel)
    position_3d2, R2, t2, error2 = run(img2, get2Dlabel2, get3Dlabel)

    print("chessboard_2 error : "+str(error2))
    f.writelines(["chessboard_2 error : "+str(error2), "\n"])

    vi.visualize(position_3d1, R1, t1, R2, t2)
    print("************** finish *****************")
    f.writelines(["************** finish *****************", "\n"])
pass