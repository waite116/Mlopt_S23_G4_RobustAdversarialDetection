import cvxpy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import advertorch
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import L2PGDAttack
from advertorch.attacks import L1PGDAttack
from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.attacks import SpatialTransformAttack
from advertorch.attacks import MomentumIterativeAttack
from advertorch.attacks import DDNL2Attack


""" 
NETWORK AND DICTIONARY HELPER FUNCTIONS

These function define the network architecture, 
forward pass, and load the saved network and dictionaries 
from the appropriate directories.
"""

class MNIST_CNN(nn.Module):
    
    def __init__(self, num_conv_layers = 4, \
                 kernels = [3,3,3,3], \
                 conv_in_channels= [1, 32, 32, 64], \
                 conv_out_channels = [32, 32, 64,64], \
                 pool_size = 2, \
                 pool_stride = 1, \
                 dropout_prop = .1, \
                 num_lin_layers = 3, \
                 lin_in_layer_sizes = [200, 200, 10], \
                 lin_out_layer_sizes = [200, 200,10]):
        super().__init__()

        # setup conv, relu, pool layers
        self.num_conv_layers = num_conv_layers
        self.layer_list = nn.ParameterList()
        for i in range(num_conv_layers):
          self.layer_list.append(nn.Conv2d(in_channels= conv_in_channels[i], out_channels = conv_out_channels[i], kernel_size=kernels[i]))
          self.layer_list.append(nn.ReLU())
          if (i==1 or i==3): 
            self.layer_list.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)) 
            # find size of linear layer after conv layers
        
        pool_out_size = 28
        for i in range(len(kernels)):
            conv_out_size = (pool_out_size - kernels[i]) + 1
            
            pool_out_size = (conv_out_size - pool_size)//pool_stride + 1

          
        
        linear_size = conv_out_channels[-1]*pool_out_size*pool_out_size

        # record position of linear index so we know when to flatten in forward pass
        self.linear_index = len(self.layer_list)
        # setup linear layers
        for i in range(num_lin_layers):
          if i==0:
              self.layer_list.append(nn.Linear(in_features = 20736,out_features=lin_out_layer_sizes[i]))
              self.layer_list.append(nn.ReLU())
          else:
              self.layer_list.append(nn.Linear(in_features=lin_out_layer_sizes[i-1],out_features=lin_out_layer_sizes[i]))
              self.layer_list.append(nn.ReLU())

        
    def forward(self, x):
      
        # send input x through all layers
        for i in range(len(self.layer_list)):
          
          #flatten in at linear layer
          if i == self.linear_index : 
            x = torch.flatten(x,1)

          x = self.layer_list[i](x)

        # return
        return x



##### Test function  #####
def test(net, loader, device):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:

            data, target = data.to(device), target.to(device)
            
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            
            total = total + 1

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))))
    
    return 100.0 * correct / len(loader.dataset)

##### Load network to a given device  #####   
def load_network(device):
    cwd = os.getcwd()
    os.chdir(cwd)
    test_batch_size = 200
    transforms = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),])
    test_dataset = torchvision.datasets.MNIST('./datasets/', train=False, download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    model_path = 'Project Models/'
    PATH = model_path+'MNIST_CNN2.pth'
    network = MNIST_CNN( )
    checkpoint = torch.load(PATH)
    network.load_state_dict(checkpoint)
    network = network.to(device)
    print("Running Test to Confirm load successful: ")
    test(network, test_loader, device)
    return network

##### Load zipped dictionaries  #####
def load_dicts():

    cwd = os.getcwd()
    os.chdir(cwd)
    dict_path = 'Project Dicts/'
    FILENAME = dict_path+'Dicts.npz'

    D= np.load(FILENAME)
    D_a = D['D_a']
    D_s = D['D_s']
    return (D_s, D_a)


""" 
SOLVER HELPER FUNCTIONS

These funcitons define the objective function we optimize 
when reconstructing, implement that Active Set Homotopy Algorithm from Thaker et al 2022, 
and reconstruct and image from its sparsity representation.
"""

##### Optimization objective for reconstruction  #####  
def objective(x_prime, D_s, D_a, cs_s, ca_s, lamb_s, lamb_a, T_s, T_a):
  
  t1s = []
  t2s = []

  # difference term
  for i in T_s:
        t1s.append(D_s[i,:,:] @ cs_s[i])
        for j in T_a: 
            t2s.append(D_a[i,j,:,:] @ ca_s[i][j])
  delta = x_prime - sum(t1s) - sum(t2s)
  # regularizer terms
  reg_s = 0
  reg_a = 0
  for i in range(10): 
      reg_s += cp.norm2(cs_s[i])
      for j in range(3): 
          reg_a += cp.norm2(ca_s[i][j])
  
  return .5*(cp.norm2(delta))**2 + lamb_s*reg_s + lamb_a*reg_a


##### Active Set Homotopy Algorithm #####  
def recover_sparse(x_prime,D_a, D_s, gamma=0.9): 
    c_s = np.zeros((10,200,1))
    c_a = np.zeros((10,3,200,1))
    T_s = []
    T_a = []
    done = False
    k=1

    # make a dictionary 10x784x200 (class, image, image number)
    # dictionary for adversarial perterbations, 10 classes, 3 types of attack, perterbations the same size as inputs
    while not done:
        if k==1: 
            o_k = x_prime
        else:
            t1 = np.zeros(x_prime.shape)
            t2 = np.zeros(x_prime.shape)

            # compute the orgin guess using the candidate indices
            for i in T_s:
                t1 +=  D_s[i,:,:]@c_s[i,:,:]
                for j in T_a: 
                    t2 += D_a[i,j,:,:] @ c_a[i,j,:,:]
            o_k = x_prime - t1 - t2

        norms_s = np.zeros(10)
        norms_a = np.zeros((10,3))
        # compute norms across all classes and adversaries
        for i in range(10): 
            norms_s[i] = np.linalg.norm(D_s[i,:,:].T @ o_k, ord=2)
            for j in range(3):
                norms_a[i,j] = np.linalg.norm(D_a[i,j,:,:].T @ o_k, ord=2)

        lamb_k_s = gamma*np.max(norms_s)
        lamb_k_a = gamma*np.max(norms_a)
        i_hat = np.argmax(norms_s)
        j_hat = np.argmax(norms_a[i_hat,:])
        
        if (i_hat in T_s) and (j_hat in T_a):
          done = True
        elif i_hat not in T_s:
          T_s.append(i_hat)
        elif j_hat not in T_a:
          T_a.append(j_hat)
        
        Ca_s = []
        Cs_s = []
        for i in range(10):
          Cs_s.append(cp.Variable(shape=c_s[i,:,:].shape))
          Ca_s.append([])
          for j in range(3):
            Ca_s[i].append(cp.Variable(shape=c_a[i,j,:,:].shape))

        #define problem and solve
        obj = cp.Minimize(objective(x_prime, D_s, D_a, Cs_s, Ca_s, lamb_k_s, lamb_k_a, T_s, T_a))
        prob = cp.Problem(obj)
        prob.solve(solver='SCS', max_iters=50)
        for i in range(10):
          c_s[i,:,:] = Cs_s[i].value

          for j in range(3):
            c_a[i,j,:,:] = Cs_s[i][j].value

        k+=1

    return (c_s, c_a)

#### Reconstruct Image from sparsity representations #####  
def predict_class_attack(x_prime, c_s, c_a, D_s, D_a):
    # Compute Flattened D_a @ c_a vectors
    da = np.zeros((784,1))
    ds = np.zeros((784,1))
    for i in range(10):
        ds += D_s[i,:,:] @ c_s[i,:,:]
        for j in range(3):
            da += D_a[i,j,:,:] @ c_a[i,j,:,:] 
      
    norms_s = np.zeros(10)
    norms_a = np.zeros(3)
    for i in range(10): 
        norms_s[i] = np.linalg.norm(x_prime - D_s[i,:,:] @ c_s[i,:,:]-da, ord=2)
    i_hat = np.argmin(norms_s)
    for j in range(3):
        norms_a[j] = np.linalg.norm(x_prime-ds - D_a[i_hat,j,:,:] @ c_a[i_hat,j,:,:], ord=2)

    j_hat = np.argmin(norms_a)
    x_hat = D_s[i_hat,:,:] @ c_s[i_hat, :,:]
    return (i_hat, j_hat, x_hat)

"""
EXPERIMENTAL FUNCTIONS

These function execute the experiments used to generate 
Tables 1 and 2 from the project report.
"""

##### Helper to deal with tuples #####  
def repack_data(attack_info):
    ims = np.zeros((len(attack_info), 2, 28*28))
    info = np.zeros((len(attack_info), 6))
    for j in range(len(attack_info)):
        (i, input_vec, x_hat, pred, pred_x_hat, i_hat, attack_type, j_hat) = attack_info[j]
        ims[j,0,:] = input_vec.reshape(28*28)
        ims[j,1,:] = x_hat.reshape(28*28)
        info[j,0] = i
        info[j,1] = pred
        info[j,2] = pred_x_hat
        info[j,3] = i_hat
        info[j,4] = attack_type
        info[j,5] = j_hat
    return (ims, info)

##### Implements detect and identify algorithm on a mix of l1, l2, and linf attacks #####  
def detect_and_identify_mixed_PGD(network, device, D_s, D_a, subset_size=1000, num_attacks=300,eps_l1 = 10., eps_l2 =2., eps_linf = .3):
    if subset_size%10 != 0: 
        print("Subset of test data need to be a multiple of 10.")
        return
    elif num_attacks%30 !=0: 
        print("Number of Attacks needs to be a multiple of 30.")
    
    # load the testing data
    ## form the dictionaries for signals and attacks
    _,(X_test, y_test) = tf.keras.datasets.mnist.load_data()
    #extract a subset of the testing data experiment with (10 per class, so 100 total)
    samp_size = subset_size//10
    small_inds= np.zeros(10*samp_size)
    small_inds = []
    for i in range(10): 
        small_inds += list(np.random.choice(np.where(y_test == i)[0], size=samp_size, replace=False))
    # assign subset
    y_small = y_test[small_inds]
    X_small = X_test[small_inds, :, :]
    
    #extract a subset of the testing data to be perturbed (select indices with class info)
    # 30 is number of attacks per class, 10 per attack type 
    samp_size = num_attacks//10
    perturbed_inds= []
    for i in range(10): 
        perturbed_inds += list(np.random.choice(np.where(y_small == i)[0], size=samp_size, replace=False))

    # choose the attack types from each class (10 of each type per class, total of 30 attacked images of the 100)
    L1_inds = []
    L2_inds = [] 
    Linf_inds = []
    for i in range(10):
        L1_inds += perturbed_inds[i*samp_size:i*samp_size+samp_size//3]
        L2_inds += perturbed_inds[i*samp_size+samp_size//3:i*samp_size+2*samp_size//3]
        Linf_inds += perturbed_inds[i*samp_size+2*samp_size//3: i*samp_size+samp_size]
        
    # make a batch for each set of perterbations (10 L1, 10 L2, 10 LInf)
    temp_tensor = torch.Tensor(X_small)
    imgs = temp_tensor/255.
    imgs = imgs[:,None,:,:]
    imgs=imgs.to(device)

    labels = torch.tensor(y_small, dtype=torch.long)
    labels = labels.to(device)
    
    ## Create the adversarial dictionary by perturbing the images in each class by all 3 types of attack
    adversary_L1 = L1PGDAttack(network, eps=eps_l1, eps_iter=0.8, nb_iter=200, rand_init=True, targeted=False)
    adversary_L2 = L2PGDAttack(network, eps=eps_l2, eps_iter=0.1, nb_iter=100, rand_init=True, targeted=False)
    adversary_Linf = LinfPGDAttack(network, eps=eps_linf, eps_iter=0.01, nb_iter=100, rand_init=True, targeted=False)
    
    advimgs_L1 = adversary_L1.perturb(imgs[L1_inds], labels[L1_inds])
    advimgs_L2 = adversary_L2.perturb(imgs[L2_inds], labels[L2_inds])
    advimgs_Linf = adversary_Linf.perturb(imgs[Linf_inds], labels[Linf_inds])
    X_perturbed = imgs
    X_perturbed[L1_inds] = advimgs_L1
    X_perturbed[L2_inds] = advimgs_L2
    X_perturbed[Linf_inds] = advimgs_Linf
    print("Adversaries Generated, starting Detection and Identification... ")
    
    Attacks = []
    False_positives = []
    False_negatives = []
    raw_correct = 0
    attacks_seen = 0
    attacks_detected = 0
    recon_correct = 0
    for i in range(len(X_perturbed)):
        # compute prediction on image
        input = X_perturbed[i]
        input = input[None,:,:,:]
        output = network(input)
        pred = output.data.max(1, keepdim=True)[1][0].item()
        true_label = y_small[i]
        raw_correct += true_label==pred

        # get reconstruction
        input_vec = input.cpu().numpy().reshape(28*28,1)
        (c_s, c_a) = recover_sparse(input_vec, D_a, D_s, gamma=0.9)
        (i_hat, j_hat, x_hat) = predict_class_attack(input_vec, c_s, c_a, D_s, D_a)

        # get network results for reconstruction
        x_hat_tensor = torch.Tensor(x_hat.reshape(28,28))
        x_hat_tensor = x_hat_tensor[None,None, :, :]
        x_hat_tensor = x_hat_tensor.to(device)
        out_x_hat = network(x_hat_tensor)
        pred_x_hat = out_x_hat.data.max(1, keepdim=True)[1][0].item()
        recon_correct += true_label == pred_x_hat

        # determine real attack type
        if i in L1_inds: 
            attack_type = 0
        elif i in L2_inds:
            attack_type = 1
        elif i in Linf_inds:
            attack_type = 2
        else: 
            attack_type = -1

        info_out = (i, input_vec, x_hat, pred, pred_x_hat, i_hat, attack_type, j_hat)
        # attack is detected according to our algo
        if pred_x_hat != pred: 
            # and it actually was an attack
            if attack_type != -1:
                attacks_seen +=1
                attacks_detected += 1
                Attacks.append(info_out)
            # otherwise it wasn't actually an attack
            else: 
                False_positives.append(info_out)
        # we did not detect an attack
        else:
            # and it actually was an attack
            if attack_type != -1:
                attacks_seen +=1
                False_negatives.append(info_out)
        if i !=0 and (i%10 == 0 or i == len(X_perturbed)-1): 
            print("Test Image #", str(i), " /", str(subset_size),":\
            \n{} Attacks seen, {} Attacks Correctly Identified, {} False Positives"\
                  .format(attacks_seen, attacks_detected, len(False_positives)))
    A_ims, A_info = repack_data(Attacks) 
    FP_ims, FP_info = repack_data(False_positives)
    FN_ims, FN_info = repack_data(False_negatives)
    print("PRINTING RESULTS!!! \n\n\n")
    
    # determine the number of correctly identified attacks
    total_attacks = len(L1_inds) + len(L2_inds) + len(Linf_inds)
    print("There were {} total attacks. \n".format(total_attacks))
    percent_detected = 100*(len(A_info)/total_attacks)
    num_false_pos = len(FP_info)
    num_false_neg = len(FN_info)
    percent_L1_detected = 100*len(np.where(A_info[:,4]==0)[0])/len(L1_inds)
    percent_L2_detected = 100*len(np.where(A_info[:,4]==1)[0])/len(L2_inds)
    percent_Linf_detected = 100*len(np.where(A_info[:,4]==2)[0])/len(Linf_inds)
    print("Overall {0:.2f} Percent of Attacks were detected".format(percent_detected))
    print("We detected {0:.2f} Percent of L1 Attacks".format(percent_L1_detected))
    print("We detected {0:.2f} Percent of L2 Attacks".format(percent_L2_detected))
    print("We detected {0:.2f} Percent of Linf Attacks".format(percent_Linf_detected))
    print()
    print("There were {} True Positives, {} False positives, and {} False Negatives.".format(len(A_info), num_false_pos, num_false_neg))
    print()
    L1_detected = len(np.where(A_info[:,4]==0))

    # determine percent of correctly identified attack types: 
    percent_identified = 100*len(np.where(A_info[:,4] == A_info[:,5])[0])/len(A_info)
    print("Overall, of the correctly detected attacks we Classified {0:.2f} Percent of attacks correctly".format(percent_identified))
    print()

    # get breakdown on the type of attacks identified correctly
    L1_correct = 0
    L2_correct = 0
    Linf_correct = 0
    for j in range(len(A_info)): 
        if A_info[j,4] == 0 and A_info[j,4]==A_info[j,5]:
            L1_correct+=1
        elif A_info[j,4] ==1 and A_info[j,4]==A_info[j,5]: 
            L2_correct +=1
        elif A_info[j,4] ==2 and A_info[j,4]==A_info[j,5]: 
            Linf_correct +=1

    percent_L1_correct = 100*L1_correct/len(np.where(A_info[:,4]==0)[0])
    percent_L2_correct = 100*L2_correct/len(np.where(A_info[:,4]==1)[0])
    percent_Linf_correct = 100*Linf_correct/len(np.where(A_info[:,4]==2)[0])
    print("Percent L1 Correctly Identified = {0:.2f}, \nPercent L2 Correctly Identified = {0:.2f},\nPercent Linf Correctly Identified = {0:.2f}".format(percent_L1_correct, percent_L2_correct, percent_Linf_correct))
    return (Attacks, False_positives, False_negatives)

##### Implements robust detection algorithm on a mix of l1, l2, and linf attacks #####  
def detect_robust_mixed_PGD(network, device, D_s, D_a, subset_size=1000, num_attacks=300,eps_l1 = 10., eps_l2 =2., eps_linf = .3):
    if subset_size%10 != 0: 
        print("Subset of test data need to be a multiple of 10.")
        return
    elif num_attacks%30 !=0: 
        print("Number of Attacks needs to be a multiple of 30.")
    
    # load the testing data
    ## form the dictionaries for signals and attacks
    _,(X_test, y_test) = tf.keras.datasets.mnist.load_data()
    #extract a subset of the testing data experiment with (10 per class, so 100 total)
    samp_size = subset_size//10
    small_inds= np.zeros(10*samp_size)
    small_inds = []
    for i in range(10): 
        small_inds += list(np.random.choice(np.where(y_test == i)[0], size=samp_size, replace=False))
    # assign subset
    y_small = y_test[small_inds]
    X_small = X_test[small_inds, :, :]
    
    #extract a subset of the testing data to be perturbed (select indices with class info)
    # 30 is number of attacks per class, 10 per attack type 
    samp_size = num_attacks//10
    perturbed_inds= []
    for i in range(10): 
        perturbed_inds += list(np.random.choice(np.where(y_small == i)[0], size=samp_size, replace=False))

    # choose the attack types from each class (10 of each type per class, total of 30 attacked images of the 100)
    L1_inds = []
    L2_inds = [] 
    Linf_inds = []
    for i in range(10):
        L1_inds += perturbed_inds[i*samp_size:i*samp_size+samp_size//3]
        L2_inds += perturbed_inds[i*samp_size+samp_size//3:i*samp_size+2*samp_size//3]
        Linf_inds += perturbed_inds[i*samp_size+2*samp_size//3: i*samp_size+samp_size]
        
    # make a batch for each set of perterbations (10 L1, 10 L2, 10 LInf)
    temp_tensor = torch.Tensor(X_small)
    imgs = temp_tensor/255.
    imgs = imgs[:,None,:,:]
    imgs=imgs.to(device)

    labels = torch.tensor(y_small, dtype=torch.long)
    labels = labels.to(device)
    
    ## Create the adversarial dictionary by perturbing the images in each class by all 3 types of attack
    adversary_L1 = L1PGDAttack(network, eps=eps_l1, eps_iter=0.8, nb_iter=200, rand_init=True, targeted=False)
    adversary_L2 = L2PGDAttack(network, eps=eps_l2, eps_iter=0.1, nb_iter=100, rand_init=True, targeted=False)
    adversary_Linf = LinfPGDAttack(network, eps=eps_linf, eps_iter=0.01, nb_iter=100, rand_init=True, targeted=False)
    
    advimgs_L1 = adversary_L1.perturb(imgs[L1_inds], labels[L1_inds])
    advimgs_L2 = adversary_L2.perturb(imgs[L2_inds], labels[L2_inds])
    advimgs_Linf = adversary_Linf.perturb(imgs[Linf_inds], labels[Linf_inds])
    X_perturbed = imgs
    X_perturbed[L1_inds] = advimgs_L1
    X_perturbed[L2_inds] = advimgs_L2
    X_perturbed[Linf_inds] = advimgs_Linf
    print("Adversaries Generated, starting Robust Detection... ")
    Attacks = []
    False_positives = []
    False_negatives = []
    raw_correct = 0
    rest_correct = 0
    TP_correct = 0
    FN_correct = 0

    recon_correct = 0
    recon_null_correct = 0

    attacks_seen = 0
    attacks_detected =0
    for i in range(len(X_perturbed)):
        # compute prediction on image
        input = X_perturbed[i]
        input = input[None,:,:,:]
        output = network(input)
        pred = output.data.max(1, keepdim=True)[1][0].item()
        true_label = y_small[i]
        raw_correct += true_label==pred

        # get reconstruction with attack
        input_vec = input.cpu().numpy().reshape(28*28,1)
        (c_s, c_a) = recover_sparse(input_vec, D_a, D_s, gamma=0.9)
        (i_hat, j_hat, x_hat) = predict_class_attack(input_vec, c_s, c_a, D_s, D_a)

        # get network results for reconstruction with attack
        x_hat_tensor = torch.Tensor(x_hat.reshape(28,28))
        x_hat_tensor = x_hat_tensor[None,None, :, :]
        x_hat_tensor = x_hat_tensor.to(device)
        out_x_hat = network(x_hat_tensor)
        pred_x_hat = out_x_hat.data.max(1, keepdim=True)[1][0].item()
        recon_correct += true_label == pred_x_hat

        # get reconstruction without attack
        D_a_dummy = np.zeros(D_a.shape)
        (c_s, c_a) = recover_sparse(input_vec, D_a_dummy, D_s, gamma=0.9)
        (i_hat_null, j_hat_null, x_hat_null) = predict_class_attack(input_vec, c_s, c_a, D_s, D_a_dummy)

        # get network results for reconstruction with attack
        x_hat_null_tensor = torch.Tensor(x_hat_null.reshape(28,28))
        x_hat_null_tensor = x_hat_null_tensor[None,None, :, :]
        x_hat_null_tensor = x_hat_null_tensor.to(device)
        out_x_hat_null = network(x_hat_null_tensor)
        pred_x_hat_null = out_x_hat_null.data.max(1, keepdim=True)[1][0].item()
        recon_null_correct += true_label == pred_x_hat_null

        # determine real attack type
        if i in L1_inds: 
            attack_type = 0
        elif i in L2_inds:
            attack_type = 1
        elif i in Linf_inds:
            attack_type = 2
        else: 
            attack_type = -1

        info_out = (i, input_vec, x_hat, pred, pred_x_hat, i_hat, attack_type, j_hat, x_hat_null, pred_x_hat_null, i_hat_null)


        # actual attack
        if attack_type != -1:
            attacks_seen += 1
            if (pred_x_hat_null != pred):
                # and we detect it
                attacks_detected += 1
                TP_correct += pred == true_label
                Attacks.append(info_out)
            else:
                FN_correct += pred == true_label
                rest_correct += pred == true_label
                False_negatives.append(info_out)
        # not an attack
        else:
            # yet it looks like one
            if (pred_x_hat_null != pred):
                # this is probably an attack. the net denoised image without attack matches the raw image
                False_positives.append(info_out)
            else:
                rest_correct += pred == true_label


        if i !=0 and (i%10 == 0 or i == len(X_perturbed)-1): 
            print("Sample", str(i), ":  {} Attacks seen, {} Attacks Correctly Identified, {} False Positives".format(attacks_seen, attacks_detected, len(False_positives)))


    print("Accuracy on True Positives: {0:.2f}".format(100*TP_correct/attacks_detected))
    print("Accuracy on False Negatives: {0:.2f}".format(100*FN_correct/len(False_negatives)))
    print("Accuracy on Clean images X_C: {0:.2f}".format(100*rest_correct/(len(X_perturbed) - len(Attacks)-len(False_positives))))
    print("Accuracy raw: {0:.2f}".format(100*raw_correct/(len(X_perturbed))))
    return (Attacks, False_positives, False_negatives)
    

##### Implements robust detection algorithm on single type of attack specified by attack_name #####  
def detect_robust_single(network, device, D_s, D_a, attack_name="CW", subset_size=1000, num_attacks=300,eps_l1 = 10., eps_l2 =2., eps_linf = .3):
    if subset_size%10 != 0: 
        print("Subset of test data need to be a multiple of 10.")
        return
    elif num_attacks%10 !=0: 
        print("Number of Attacks needs to be a multiple of 10.")
    
    # load the testing data
    ## form the dictionaries for signals and attacks
    _,(X_test, y_test) = tf.keras.datasets.mnist.load_data()
    #extract a subset of the testing data experiment with (10 per class, so 100 total)
    samp_size = subset_size//10
    small_inds= np.zeros(10*samp_size)
    small_inds = []
    for i in range(10): 
        small_inds += list(np.random.choice(np.where(y_test == i)[0], size=samp_size, replace=False))
    # assign subset
    y_small = y_test[small_inds]
    X_small = X_test[small_inds, :, :]
    
    #extract a subset of the testing data to be perturbed (select indices with class info)
    # 30 is number of attacks per class, 10 per attack type 
    samp_size = num_attacks//10
    perturbed_inds= []
    for i in range(10): 
        perturbed_inds += list(np.random.choice(np.where(y_small == i)[0], size=samp_size, replace=False))
    Attack_inds = perturbed_inds
    
    
    # make a batch for each set of perterbations (10 L1, 10 L2, 10 LInf)
    temp_tensor = torch.Tensor(X_small)
    imgs = temp_tensor/255.
    imgs = imgs[:,None,:,:]
    imgs=imgs.to(device)

    labels = torch.tensor(y_small, dtype=torch.long)
    labels = labels.to(device)
    
    if attack_name == "CW":
        adversary = CarliniWagnerL2Attack(network, 10, max_iterations=500)
    elif attack_name == "ST":
        adversary = SpatialTransformAttack(network, 10)
    elif attack_name == "MIM":
        adversary = MomentumIterativeAttack(network)
    elif attack_name == "DDN": 
        adversary = DDNL2Attack(network)
    elif attack_name == "L1": 
        adversary = L1PGDAttack(network, eps=10., eps_iter=0.8, nb_iter=200, rand_init=True, targeted=False)
    elif attack_name == "L2":
        adversary = L2PGDAttack(network, eps=2., eps_iter=0.1, nb_iter=100, rand_init=True, targeted=False)
    elif attack_name =="Linf":
        adversary = LinfPGDAttack(network, eps=0.3, eps_iter=0.01, nb_iter=100, rand_init=True, targeted=False)
    else: 
        print("Attack name is not valid, please choose from the documented options.")
        return
    # make aversaries and inject into data
    advimgs = adversary.perturb(imgs[Attack_inds], labels[Attack_inds])
    X_perturbed = imgs
    X_perturbed[Attack_inds] = advimgs
    Attacks = []
    False_positives = []
    False_negatives = []
    raw_correct = 0
    rest_correct = 0
    TP_correct = 0
    FN_correct = 0

    recon_correct = 0
    recon_null_correct = 0

    attacks_seen = 0
    attacks_detected =0
    for i in range(len(X_perturbed)):
        # compute prediction on image
        input = X_perturbed[i]
        input = input[None,:,:,:]
        output = network(input)
        pred = output.data.max(1, keepdim=True)[1][0].item()
        true_label = y_small[i]
        raw_correct += true_label==pred

        # get reconstruction with attack
        input_vec = input.cpu().numpy().reshape(28*28,1)
        (c_s, c_a) = recover_sparse(input_vec, D_a, D_s, gamma=0.9)
        (i_hat, j_hat, x_hat) = predict_class_attack(input_vec, c_s, c_a, D_s, D_a)

        # get network results for reconstruction with attack
        x_hat_tensor = torch.Tensor(x_hat.reshape(28,28))
        x_hat_tensor = x_hat_tensor[None,None, :, :]
        x_hat_tensor = x_hat_tensor.cuda()
        out_x_hat = network(x_hat_tensor)
        pred_x_hat = out_x_hat.data.max(1, keepdim=True)[1][0].item()
        recon_correct += true_label == pred_x_hat

        # get reconstruction without attack
        D_a_dummy = np.zeros(D_a.shape)
        (c_s, c_a) = recover_sparse(input_vec, D_a_dummy, D_s, gamma=0.9)
        (i_hat_null, j_hat_null, x_hat_null) = predict_class_attack(input_vec, c_s, c_a, D_s, D_a_dummy)

        # get network results for reconstruction with attack
        x_hat_null_tensor = torch.Tensor(x_hat_null.reshape(28,28))
        x_hat_null_tensor = x_hat_null_tensor[None,None, :, :]
        x_hat_null_tensor = x_hat_null_tensor.cuda()
        out_x_hat_null = network(x_hat_null_tensor)
        pred_x_hat_null = out_x_hat_null.data.max(1, keepdim=True)[1][0].item()
        recon_null_correct += true_label == pred_x_hat_null

        # determine real attack type
        if i in Attack_inds: 
            attack_type = 0
        else: 
            attack_type = -1

        info_out = (i, input_vec, x_hat, pred, pred_x_hat, i_hat, attack_type, j_hat, x_hat_null, pred_x_hat_null, i_hat_null)


        # actual attack
        if attack_type != -1:
            attacks_seen += 1
            if (pred_x_hat_null != pred):
                # and we detect it
                attacks_detected += 1
                TP_correct += pred == true_label
                Attacks.append(info_out)
            else:
                FN_correct += pred == true_label
                rest_correct += pred == true_label
                False_negatives.append(info_out)
        # not an attack
        else:
            # yet it looks like one
            if (pred_x_hat_null != pred):
                # this is probably an attack. the net denoised image without attack matches the raw image
                False_positives.append(info_out)
            else:
                rest_correct += pred == true_label


        if i !=0 and (i%10 == 0 or i == len(X_perturbed)-1): 
            print("Sample", str(i), ":  {} Attacks seen, {} Attacks Correctly Identified, {} False Positives".format(attacks_seen, attacks_detected, len(False_positives)))
        
    print("Accuracy on True Positives: {0:.2f}".format(100*TP_correct/attacks_detected))
    print("Accuracy on False Negatives: {0:.2f}".format(100*FN_correct/len(False_negatives)))
    print("Accuracy on X_A -A: {0:.2f}".format(100*rest_correct/(len(X_perturbed) - len(Attacks)-len(False_positives))))
    print("Accuracy raw: {0:.2f}".format(100*raw_correct/(len(X_perturbed))))
    return (Attacks, False_positives, False_negatives)

       
        
        