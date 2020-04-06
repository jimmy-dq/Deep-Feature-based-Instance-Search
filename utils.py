import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import *

def get_crop(im, rect, output_size):
    im_sz = im.shape #H*W*C
    context_xmin = max(1, rect[0]) # start from one, be more tight
    context_ymin = max(1, rect[1]) # start from one, be more tight
    context_xmax = max(1, min(im_sz[1], rect[0]+rect[2]))
    context_ymax = max(1, min(im_sz[0], rect[1]+rect[3]))

    im_patch_original = im[int(context_ymin) - 1:int(context_ymax), int(context_xmin) - 1:int(context_xmax), :]
    im_patch = cv2.resize(im_patch_original, (int(output_size), int(output_size)), interpolation=cv2.INTER_CUBIC)
    return im_patch

def get_ranklist(result_dict):
    query_key_list = list(result_dict.keys())
    ranklist_dict = {}
    for query in query_key_list:
        r_list = result_dict[query]
        rank_list = []
        for j in range(len(r_list)):
            rank_list.append(r_list[j][0])
        ranklist_dict[query] = rank_list
    return ranklist_dict

def show_qualitative_result_wo_rectange(rank_dict, top_k_search = 10, query_num = 10):
    # This is used to validate the rank_dict
    query_img_path = '/home/wqq/Data/Retrieval/Queries'
    search_img_path = '/home/wqq/Data/Retrieval/Images'
    query_imgs = sorted(list(rank_dict.keys()))[0:query_num]

    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    index = 1
    for i in range(query_num):
        query_img = query_imgs[i]
        q_im = cv2.imread(os.path.join(query_img_path, query_img.split('-')[1] + '.jpg'))
        q_im = cv2.cvtColor(q_im, cv2.COLOR_BGR2RGB)
        plt.subplot(query_num, top_k_search+1, index)
        # q_anno = np.loadtxt(os.path.join(query_img_path, query_img.split('-')[1]+'.txt'))
        # if len(q_anno.shape) != 1:
        #     for l in range(q_anno.shape[0]):
        #         # rect_converted = q_im.shape[0]/200.0*q_anno[l]
        #         rect_converted = q_anno[l]
        # else:
        #     # rect_converted = q_im.shape[0]/200.0*q_anno
        #     rect_converted = q_anno
        # cv2.rectangle(q_im, (int(rect_converted[0]), int(rect_converted[1])),
        #               (int(rect_converted[0] + rect_converted[2]),
        #                int(rect_converted[1] + rect_converted[3])), (255, 0, 0), 10)
        plt.axis('off')
        plt.imshow(q_im)  # cmap='gray'
        index += 1
        for j in range(top_k_search):
            s_im = cv2.imread(os.path.join(search_img_path, rank_dict[query_img][j] + '.jpg'))
            s_im = cv2.cvtColor(s_im, cv2.COLOR_BGR2RGB)
            plt.subplot(query_num, top_k_search+1, index)
            plt.axis('off')
            plt.imshow(s_im)  # cmap='gray'
            index += 1
    plt.show()





def show_qualitative_result_w_rectange(result_dict, top_k_search = 10, query_num = 5, is_save=False, save_folder=None):
    # This is used to validate the rank_dict
    query_img_path = '/home/wqq/Data/Retrieval/Queries'
    search_img_path = '/home/wqq/Data/Retrieval/Images'
    npy_path = '/home/wqq/Data/Retrieval/Selective_search_200'
    # query_imgs = sorted(list(result_dict.keys()))[0:query_num]
    query_imgs = sorted(list(result_dict.keys()))

    fig = plt.figure(figsize=(16, 12))
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)
    index = 1
    for i in range(query_num):
        query_img = query_imgs[i]
        q_im = cv2.imread(os.path.join(query_img_path, query_img.split('-')[1] + '.jpg'))
        q_im = cv2.cvtColor(q_im, cv2.COLOR_BGR2RGB)
        plt.subplot(query_num, top_k_search+1, index)
        q_anno = np.loadtxt(os.path.join(query_img_path, query_img.split('-')[1]+'.txt'))
        if len(q_anno.shape) != 1:
            for l in range(q_anno.shape[0]):
                # rect_converted = q_im.shape[0]/200.0*q_anno[l]
                rect_converted = q_anno[l]
                cv2.rectangle(q_im, (int(rect_converted[0]), int(rect_converted[1])),
                              (int(rect_converted[0] + rect_converted[2]),
                               int(rect_converted[1] + rect_converted[3])), (255, 0, 0), 10)
        else:
            # rect_converted = q_im.shape[0]/200.0*q_anno
            rect_converted = q_anno
            cv2.rectangle(q_im, (int(rect_converted[0]), int(rect_converted[1])),
                          (int(rect_converted[0] + rect_converted[2]),
                           int(rect_converted[1] + rect_converted[3])), (255, 0, 0), 10)
        plt.axis('off')
        plt.imshow(q_im)  # cmap='gray'
        index += 1
        for j in range(top_k_search):
            ranked_search_image = result_dict[query_img][j]
            search_id = ranked_search_image[0]
            instances = ranked_search_image[1]
            s_im = cv2.imread(os.path.join(search_img_path, search_id + '.jpg'))
            s_im = cv2.cvtColor(s_im, cv2.COLOR_BGR2RGB)
            rects = np.load(os.path.join(npy_path, search_id + '.npy'))
            if len(instances) == 2: #single instance
                instance_name = instances[0]
                rect = rects[int(instance_name.split('-')[-1])]
                rect = s_im.shape[0] / 200.0 * rect
                cv2.rectangle(s_im, (int(rect[0]), int(rect[1])),
                              (int(rect[0] + rect[2]),
                               int(rect[1] + rect[3])), (255, 0, 0), 10)
            else:
              for r in range(0, len(instances)-1):
                    instance_name = instances[r][0]
                    rect = rects[int(instance_name.split('-')[-1])]
                    rect = s_im.shape[0] / 200.0 * rect
                    cv2.rectangle(s_im, (int(rect[0]), int(rect[1])),
                                  (int(rect[0] + rect[2]),
                                   int(rect[1] + rect[3])), (255, 0, 0), 10)


            plt.subplot(query_num, top_k_search+1, index)
            plt.axis('off')
            plt.imshow(s_im)  # cmap='gray'
            index += 1
    if is_save:
        fig.savefig(
            './results/' + save_folder + '/results_query_'+str(query_num) + '_search_' + str(top_k_search) + '.pdf', dpi=100)

    plt.show()


def save_ranklist(file_name, ranklist):
    queries = sorted(list(ranklist.keys()))
    with open(file_name, 'w', encoding='utf-8') as f:
        for query in queries:
            query_result = 'Q' + str(int(query.split('-')[1])) + ': '
            r_list = ranklist[query]
            for i in range(len(r_list)):
                query_result = query_result + str(int(r_list[i])) + ' '
            query_result = query_result + '\n'
            f.write(query_result)
    f.close()






