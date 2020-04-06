from utils import  *
import torchvision.models as models
import torch
import torch.nn as nn
from Dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import json
import torch.nn.functional as F

def get_search_dict(base_path):
    imgs = os.listdir(os.path.join(base_path, 'Images_crop'))
    search_dict = {}
    for i in range(len(imgs)):
        img = imgs[i]
        full_name = img.split('-')
        search_id = full_name[1]
        if search_dict.get(search_id, -1) == -1:
            search_dict[search_id] = []
            search_dict[search_id].append(img)
        else:
            search_dict[search_id].append(img)
    json.dump(search_dict, open('./search_dict_200.json', 'w'), indent=2)

# resnet-18 for retrieval and save the final similarity_dict
def resnet_18_retrieval():
    query_feat = json.load(open('./resnet_18_query_feat.json', 'r'))

    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
    batch_size = 50
    base_path = '/home/wqq/Data/Retrieval'

    search_dict = json.load(open('./search_dict_200.json', 'r'))

    # ResNet
    resnet18 = models.resnet18(pretrained=True)
    # we remove the last fc
    modules = list(resnet18.children())[:-1]
    resnet18 = nn.Sequential(*modules)
    resnet18 = resnet18.cuda()
    resnet18.eval()

    similarity_dict = {}

    search_ids = list(search_dict.keys())
    query_ids = list(query_feat.keys())

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    for i in range(len(query_ids)):
        similarity_dict[query_ids[i]] = {}
        query_feat[query_ids[i]] = torch.from_numpy(np.array(query_feat[query_ids[i]])).type(torch.cuda.FloatTensor)

    for loop in tqdm(range(len(search_ids))):
        # t1 = time.time()
        search_id = search_ids[loop]
        if loop!=0 and loop%500==0:
            print(loop)
        dataset = Dataset(os.path.join(base_path, 'Images_crop'), search_dict[search_id], resnet_transform)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=1, drop_last=False)
        temp_dict = {}
        for i in range(len(query_ids)):
            temp_dict[query_ids[i]] = {}
        for j, data in enumerate((train_loader)):
            img_names, imgs = data #ims are proposals
            imgs = imgs.cuda()
            feats = resnet18(imgs)
            for p in range(len(query_ids)):
                q_f = query_feat[query_ids[p]].cuda()
                for q in range(len(img_names)):
                    f = feats[q]
                    c_s = cos(f.squeeze(), q_f)
                    temp_dict[query_ids[p]][img_names[q]] = c_s.data.cpu().numpy().tolist()
        for p in range(len(query_ids)):
            t_dict = temp_dict[query_ids[p]]
            best_instance = sorted(t_dict.items(), key=lambda item:item[1])[-1]
            similarity_dict[query_ids[p]][search_id] = best_instance
    json.dump(similarity_dict, open('./resnet_18_similarity_dict.json', 'w'), indent=2)



def vgg_11_retrieval():
    query_feat = json.load(open('./vgg_11_query_feat.json', 'r'))
    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    batch_size = 30
    base_path = '/home/wqq/Data/Retrieval'

    search_dict = json.load(open('./search_dict_200.json', 'r'))
    vgg11 = models.vgg11(pretrained=True)
    vgg11 = vgg11.features
    vgg11 = vgg11.cuda()
    vgg11.eval()

    similarity_dict = {}

    search_ids = list(search_dict.keys())
    query_ids = list(query_feat.keys())

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    for i in range(len(query_ids)):
        similarity_dict[query_ids[i]] = {}
        query_feat[query_ids[i]] = torch.from_numpy(np.array(query_feat[query_ids[i]])).type(torch.cuda.FloatTensor)

    for loop in tqdm(range(len(search_ids))):
        # t1 = time.time()
        search_id = search_ids[loop]
        if loop != 0 and loop % 500 == 0:
            print(loop)
        if loop != 0 and loop % 5000 == 0:
            json.dump(similarity_dict, open('./similarity_dict.json', 'w'), indent=2)
        dataset = Dataset(os.path.join(base_path, 'Images_crop'), search_dict[search_id], resnet_transform)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=1, drop_last=False)
        temp_dict = {}
        for i in range(len(query_ids)):
            temp_dict[query_ids[i]] = {}
        for j, data in enumerate((train_loader)):
            img_names, imgs = data  # ims are proposals
            imgs = imgs.cuda()
            feats = vgg11(imgs)
            feats = F.avg_pool2d(feats, kernel_size=7)
            for p in range(len(query_ids)):
                q_f = query_feat[query_ids[p]].cuda()
                for q in range(len(img_names)):
                    f = feats[q]
                    c_s = cos(f.squeeze(), q_f)
                    temp_dict[query_ids[p]][img_names[q]] = c_s.data.cpu().numpy().tolist()
        for p in range(len(query_ids)):
            t_dict = temp_dict[query_ids[p]]
            best_instance = sorted(t_dict.items(), key=lambda item: item[1])[-1]
            similarity_dict[query_ids[p]][search_id] = best_instance
    json.dump(similarity_dict, open('./vgg_11_similarity_dict.json', 'w'), indent=2)


if __name__ == '__main__':
    base_path = '/home/wqq/Data/Retrieval'
    get_search_dict(base_path)

    # use resnet-18
    resnet_18_retrieval()

    # use vgg-11
    # vgg_11_retrieval()