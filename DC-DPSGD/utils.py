
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets 
import torchvision.io as io

from functools import partial

import os


import numpy as np
from rdp_accountant import compute_rdp, get_privacy_spent
#from lanczos import Lanczos

from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100


def process_grad_batch(params, clipping=1):
    n = params[-1].grad_batch.shape[0]
    #print("n:", n)
    #print("len(params):", len(params))
    grad_norm_list = torch.zeros(n).cuda()
    for p in params:  #every layer
        if p.requires_grad == True:
            flat_g = p.grad_batch.reshape(n, -1)
            #print("flat_g:", flat_g.shape)
            current_norm_list = torch.norm(flat_g, dim=1)
            #print("current_norm_list:", current_norm_list[0:10])
            grad_norm_list += torch.square(current_norm_list)
            #print("grad_norm_list",grad_norm_list.shape)
    grad_norm_list = torch.sqrt(grad_norm_list)

    # clipping dp-sgd
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1  

    # auto clip - bu
    # gamma=0.01
    # scaling = clipping/(grad_norm_list + gamma)

    # auto clip - xia
    # gamma=0.01
    # scaling = clipping/(grad_norm_list + gamma / (grad_norm_list + gamma) )

    for p in params:
        if p.requires_grad == True:
            p_dim = len(p.shape)
            #print("scaling:",scaling.shape)
            scaling = scaling.view([n] + [1]*p_dim)
            #print("scaling-a:",scaling.shape)
            p.grad_batch *= scaling
            p.grad = torch.mean(p.grad_batch, dim=0)
            p.grad_batch.mul_(0.)




def process_grad(params, clipping=1):
    for p in params:
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)


def process_grad(params, clipping=1):
    for p in params:
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)
        

def process_layer_grad_batch(params, batch_idx, Vk, clipping=1):
    n = params[0].grad_batch.shape[0]
    #print("n:", n)
    #print("len(params):", len(params))
    grad_norm_list = torch.zeros(len(params), n).cuda()
    idx_layer = 0
    for p in params:  #every layer
        flat_g = p.grad_batch.reshape(n, -1)
        mean_batch = torch.mean(flat_g, dim=0)

        ### random proj
        # if batch_idx == 0:
        #     random_p = np.random.random(size=(flat_g[0].cpu().numpy().size, 1))
        #     Vk_layer, _ = np.linalg.qr(random_p)
        #     Vk.append(Vk_layer)
        # Vk_layer = torch.from_numpy(Vk[idx_layer]).float().cuda()
        # flat_g = torch.matmul(Vk_layer, torch.matmul(Vk_layer.T, flat_g.T)).T
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)

        ### random vec
        # Vk.append(torch.randn(flat_g.shape[1], 1,dtype=torch.float32).cuda())
        # Vk /= torch.norm(Vk)
        # flat_g = torch.matmul(Vk, torch.matmul(Vk.T, flat_g.T)).T
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        
        ### sparsify
        # Vk = sparsify(flat_g.shape[1], 0.5)
        # flat_g = torch.mul(flat_g.T, Vk).T
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        #print("Vk:", Vk)
        #print("flat_g:", flat_g)

        ### pca-lanczos
        # if batch_idx == 0:
        #     Vk_layer = eigen_by_lanczos((flat_g - mean_batch).cpu().numpy(), 1)
        #     Vk.append(Vk_layer)
        # Vk_layer = torch.from_numpy(Vk[idx_layer]).float().cuda()
        # flat_g = torch.matmul(Vk_layer, torch.matmul(Vk_layer.T, flat_g.T)).T + mean_batch
        # p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        #print("flat_g:", flat_g.shape)

        ### pca-torch
        if batch_idx == 0:
            print("flat_g - mean_batch:", (flat_g - mean_batch).shape)
            Vk_layer, _, _ = torch.linalg.svd( (flat_g - mean_batch).T, full_matrices=False)
            Vk.append(Vk_layer[:,0:1])
            print("Vk_layer:", Vk_layer[:,0:1].shape)
        Vk_layer = Vk[idx_layer]
        flat_g = torch.matmul(Vk_layer, torch.matmul(Vk_layer.T, flat_g.T)).T + mean_batch
        p.grad_batch = flat_g.reshape(p.grad_batch.shape)
        
        ### classic
        #flat_g = torch.matmul(Vk, torch.matmul(Vk.T, flat_g.T)).T
        current_norm_list = torch.norm(flat_g, dim=1)
        #print("current_norm_layer_list:", current_norm_list[0:10])
        grad_norm_list[idx_layer] += current_norm_list
        idx_layer += 1 
    #print("grad_norm_layer_list-10:",grad_norm_list[15,0:10])
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1
    #print("scaling.shape:", scaling.shape)
    
    idx_layer = 0
    for p in params:
        p_dim = len(p.shape)
        #print("scaling:",scaling.shape)
        scaling_layer = scaling[idx_layer].view([n] + [1]*p_dim)
        #print("scaling_layer.shape:", scaling_layer.shape)
        idx_layer += 1
        #print("p.grad_batch:", p.grad_batch)
        p.grad_batch *= scaling_layer
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)
    return grad_norm_list[15,0], Vk


def sparsify(d, ratio):
    vec_one = np.ones((d,1))
    vec_zero = np.zeros((d,1))
    vec = np.concatenate((vec_one[0:int(d*ratio),:], vec_zero[0:int(d*(1-ratio)),:]))
    idx = np.arange(d)
    np.random.shuffle(idx)
    vec = torch.from_numpy(vec[idx]).float().cuda()
    return vec

def eigen_by_lanczos(mat, proj_dims):
        T, V = Lanczos(mat, 128)
        T_evals, T_evecs = np.linalg.eig(T)
        idx = T_evals.argsort()[-1 : -(proj_dims+1) : -1]
        Vk = np.dot(V.T, T_evecs[:,idx])
        return Vk


def get_data_loader(dataset, batchsize):
    if(dataset == 'svhn'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN('./data/SVHN',split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=73257, shuffle=True, num_workers=0) #load full btach into memory, to concatenate with extra data

        extraset = torchvision.datasets.SVHN('./data/SVHN',split='extra', download=True, transform=transform)
        extraloader = torch.utils.data.DataLoader(extraset, batch_size=531131, shuffle=True, num_workers=0) #load full btach into memory

        testset = torchvision.datasets.SVHN('./data/SVHN',split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        return trainloader, extraloader, testloader, len(trainset)+len(extraset), len(testset)

    if(dataset == 'mnist'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.MNIST(root='./data',train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=2) #load full btach into memory, to concatenate with extra data

        testset = torchvision.datasets.MNIST(root='./data',train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, trainset, len(trainset), len(testset)

    elif(dataset == 'fmnist'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data',train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=2) #load full btach into memory, to concatenate with extra data

        testset = torchvision.datasets.FashionMNIST(root='./data',train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
        return trainloader, testloader, trainset, len(trainset), len(testset)

    elif(dataset == 'imagenette'):  
        #data_dir = "./data/tiny-imagenet-200/"
        data_dir = "./data/imagenette/imagenette2"
        num_workers = {"train": 2, "val": 0}
        image_size = 32
        image_read_func = partial(io.read_image, mode=io.image.ImageReadMode.RGB)
        data_transforms = {
            "train": transforms.Compose(
                [   
                    transforms.Resize((image_size, image_size)),
                    #transforms.RandomRotation(20),
                    #transforms.RandomHorizontalFlip(0.5),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            ),
            "val": transforms.Compose(
                [   
                    transforms.Resize((image_size, image_size)),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            ),
        }
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x], loader=image_read_func) for x in ["train", "val"]
        }
        dataloaders = {
            x: data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=True, num_workers=num_workers[x])
            for x in ["train", "val"]
        }

        return dataloaders["train"], dataloaders["val"], len(image_datasets["train"]), len(image_datasets["val"])

        
    elif(dataset == 'tiny_imagenet'):  
        data_dir = "tiny-imagenet-200/"
        num_workers = {"train": 2, "val": 0, "test": 0}
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            ),
        }
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
        }
        dataloaders = {
            x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
            for x in ["train", "val", "test"]
        }

        return dataloaders["train"], dataloaders["test"], len(image_datasets["train"]), len(image_datasets["test"])


    elif(dataset == 'cifar10'):
        transform_train = transforms.Compose([
        #transforms.RandomResizedCrop(224), 
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = IMBALANCECIFAR10(root='./data/CIFAR10', imb_type="exp", imb_factor=0.01, rand_number=0, train=True, download=False, transform=transform_train)
        #trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=False, transform=transform_train) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=False, transform=transform_test) 
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

        return trainloader, testloader, len(trainset), len(testset)



def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma
    
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps


def restore_param(cur_state, state_dict):
    own_state = cur_state
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)

def sum_list_tensor(tensor_list, dim=0):
    return torch.sum(torch.cat(tensor_list, dim=dim), dim=dim)

def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def checkpoint(net, acc, epoch, sess):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess  + '.ckpt')

def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    decay = 1.0
    if(epoch<all_epoch*0.5):
        decay = 1.
    elif(epoch<all_epoch*0.75):
        decay = 10.
    else:
        decay = 100.

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / decay
    return init_lr / decay
