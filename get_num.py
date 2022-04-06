import cv2
import torch
import numpy as np


def cut_zong(img):
    zong = [0, img.shape[0] - 1]
    ind = 0
    for i in img:
        if 0 in i:
            zong[0] = ind
            break
        ind += 1

    ind2 = img.shape[0] - 1
    for i in img[::-1]:
        if 0 in i:
            zong[1] = ind2
            break
        ind2 -= 1

    if len(zong) == 1:
        zong.append(img.shape[0] - 1)
    zong_img_cut = img[zong[0]:zong[1], :]

    if zong[1] - zong[0] <= 20:
        zong_img_cut = img[:, :]
    return zong_img_cut


def find_num_sym(img):
    img1 = img
    img = img.T
    index = [0]

    def num_begin(ind1=0):
        ind_now1 = ind1
        index_begin = -1
        for i in img[ind1:]:
            if 0 in i:
                index_begin = ind_now1
                break
            ind_now1 = ind_now1 + 1
        return index_begin

    def num_end(ind2):
        ind_now2 = ind2
        index_end = 0
        for i in img[ind2:]:
            if not 0 in i:
                index_end = ind_now2
                break
            ind_now2 = ind_now2 + 1
            if ind_now2 >= img.shape[0] - 1:
                return img.shape[0] - 1
        return index_end

    while not index[-1] == -1:
        begin = num_begin(index[-1])
        index.append(begin)
        end = num_end(begin)
        index.append(end)
        if index[-1] == img.shape[0] - 1:
            break

    if len(index) == 3 and index[0] == 0 and index[-1] == 44:
        return [img1]
    elif len(index) == 3 and index[0] == 0 and index[1] == 0:
        return [img1]

    index = list(set(index))
    if 0 in index:
        index.remove(0)
    if -1 in index:
        index.remove(-1)
    if len(index) == 1:
        return [img1]

    index.sort()
    imgs = []
    for _ in range(int(len(index) / 2)):
        img_now = img1[:, index[2 * _]:index[2 * _ + 1]]
        imgs.append(img_now)
    return imgs


def get_num_sym(dir, img_size=45):
    img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

    ret1, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # ret = 127, thresh = img  black:0 white:255

    vertical_imgs = find_num_sym(thresh_img)

    horizontal_imgs = []
    for thresh_img2 in vertical_imgs:
        thresh_img3 = cut_zong(thresh_img2)
        horizontal_imgs.append(thresh_img3)

    imgs = []
    for img_now in horizontal_imgs:

        img_now = cv2.resize(img_now, (img_size, img_size),
                             interpolation=cv2.INTER_AREA)

        for i in range(img_now.shape[0]):
            for j in range(img_now.shape[1]):
                if not img_now[i][j] == 255:
                    img_now[i][j] = 0

        img_now = np.float32(img_now)  # [0,255] --> [0.,255.]
        img_now = img_now / 255.  # [0.,255.] --> [0.,1.]

        img_now = torch.from_numpy(img_now)  # ndarray --> tensor
        imgs.append(img_now)

    imgs = torch.stack(imgs, dim=0)  # list(N * h * w) ---> tensor(N * h * w)
    #
    imgs = torch.unsqueeze(imgs, dim=1)  # tensor(N * h * w) ---> tensor(N * 1 * h * w)

    return imgs


if __name__ == '__main__':

    a = get_num_sym(
        dir='./test01.jpg')

    print(type(a))
    a = torch.squeeze(a, dim=1)
    a = np.array(a)
    for i in a:
        cv2.imshow('a', i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pass
