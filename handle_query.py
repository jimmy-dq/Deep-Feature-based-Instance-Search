# Crop instances in query images
# Extract instance feats


from utils import  *
import torchvision.models as models
import torch.nn as nn
from Dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import json
import torch.nn.functional as F

def save_query_crop():
    output_size = 224
    base_path = '/home/wqq/Data/Retrieval'
    gts_list = os.listdir(os.path.join(base_path, 'Queries'))
    count = 0
    # save query crop imgs
    for gt in gts_list:
        # Read anno file
        if '.txt' in gt:
            print(gt)
            count += 1
            anno = np.loadtxt(os.path.join(base_path, 'Queries', gt))
            im = cv2.imread(os.path.join(base_path, 'Queries', gt[0:-4]+'.jpg'))
            # if gt == '44.txt':
            #     print('debug')
            if len(anno.shape) != 1:
                for i in range(anno.shape[0]):
                    # for instance: Query-gt[0:-4]-Instance-i
                    rect = anno[i]
                    # crop and save
                    im_crop = get_crop(im, rect, output_size)
                    savename_crop = os.path.join(base_path, 'Queries_crop', 'Query-{}-{}.jpg'.format(gt[0:-4], str(i)))
                    cv2.imwrite(savename_crop, im_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            else:
                    # for instance: Query-gt[0:-4]-Instance-0
                    rect = anno
                    im_crop = get_crop(im, rect, output_size)
                    savename_crop = os.path.join(base_path, 'Queries_crop', 'Query-{}-{}.jpg'.format(gt[0:-4], str(0)))
                    cv2.imwrite(savename_crop, im_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(count)

def resnet_18_extraction():
    # Feature extraction for quiries
    batch_size = 4
    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    # Extract features for query instances
    base_path = '/home/wqq/Data/Retrieval'
    imgs = os.listdir(os.path.join(base_path, 'Queries_crop'))
    # ResNet
    resnet18 = models.resnet18(pretrained=True)
    # we remove the last fc
    modules = list(resnet18.children())[:-1]
    resnet18 = nn.Sequential(*modules)
    resnet18 = resnet18.cuda()
    resnet18.eval()

    dataset = Dataset(os.path.join(base_path, 'Queries_crop'), imgs, resnet_transform)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, drop_last=False)
    feat_dict = {}
    for j, data in enumerate(tqdm(train_loader)):
        img_names, imgs = data
        imgs = imgs.cuda()
        feats = resnet18(imgs)
        for i in range(len(img_names)):
            feat_dict[img_names[i]] = feats[i].squeeze().data.cpu().numpy().tolist()
    print(feat_dict)
    print(len(feat_dict.keys()))
    json.dump(feat_dict, open('./resnet_18_query_feat.json', 'w'), indent=2)


def vgg_11_extraction():
    batch_size = 4
    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    # Extract features for query instances
    base_path = '/home/wqq/Data/Retrieval'
    imgs = os.listdir(os.path.join(base_path, 'Queries_crop'))
    # ResNet
    vgg11 = models.vgg11(pretrained=True)
    # we remove the last fc
    vgg11 = vgg11.features
    vgg11 = vgg11.cuda()
    vgg11.eval()

    dataset = Dataset(os.path.join(base_path, 'Queries_crop'), imgs, resnet_transform)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, drop_last=False)
    feat_dict = {}
    for j, data in enumerate(tqdm(train_loader)):
        img_names, imgs = data
        imgs = imgs.cuda()
        feats = vgg11(imgs)
        feats = F.avg_pool2d(feats, kernel_size=7)
        for i in range(len(img_names)):
            feat_dict[img_names[i]] = feats[i].squeeze().data.cpu().numpy().tolist()
    print(feat_dict)
    print(len(feat_dict.keys()))
    json.dump(feat_dict, open('./vgg_11_query_feat.json', 'w'), indent=2)


if __name__ == '__main__':
    # crop queries
    save_query_crop()

    # renet-18 features
    resnet_18_extraction()

    # vgg-11 features
    # vgg_11_extraction()



