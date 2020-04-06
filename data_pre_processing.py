import sys
import cv2
import os
import numpy as np
from utils import  *

def generate_ss_rect():
    # Pre-crop image patches generate by Selecttive-Search
    base_path = '/home/wqq/Data/Retrieval/Images'
    proposal_txt_save_path = '/home/wqq/Data/Retrieval/Selective_search_200'
    imgs = os.listdir(base_path)
    newHeight = 200
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    for i in range(len(imgs)):
        print('[%d]/[%d]' %(i, len(imgs)))
        im = cv2.imread(os.path.join(base_path, imgs[i]))
        newWidth = int(im.shape[1] * 200 / im.shape[0])
        im = cv2.resize(im, (newWidth, newHeight))
        ss.setBaseImage(im)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        np.save(os.path.join(proposal_txt_save_path, imgs[i][0:-4]+'.npy'), rects)

def crop_ss_rect():
    keep_proposal = 200
    base_path = '/home/wqq/Data/Retrieval'
    output_size = 224
    rect_npy_files = os.listdir(os.path.join(base_path, 'Selective_search_200'))
    count = 1
    for rect_npy in rect_npy_files:
        # print(count)
        # if count<3920:
        #     count+=1
        #     continue
        # if count == 3920:
        #     print('a')
        search_img = rect_npy[0:-4]
        rects = np.load(os.path.join(base_path, 'Selective_search_200', rect_npy))
        im = cv2.imread(os.path.join(base_path, 'Images', search_img+'.jpg'))
        H, W = im.shape[0], im.shape[1]
        if rects.shape[0]> keep_proposal:
            rects = rects[0:keep_proposal]
        for i in range(rects.shape[0]):
            rect = rects[i]
            rect = H/200.0*rect
            im_crop = get_crop(im, rect, output_size)
            savename_crop = os.path.join(base_path, 'Images_crop', 'Search-{}-{}.jpg'.format(search_img, str(i)))
            cv2.imwrite(savename_crop, im_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        count += 1

if __name__ == '__main__':
    print('Proposal Generation:')
    generate_ss_rect()
    print('Proposal Crop:')
    crop_ss_rect()

