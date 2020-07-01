import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os
import csv
from torch.autograd import Variable
import sys
import numpy as np
from sklearn import metrics

import network
import resnet3d 
import dataset





# Configure parameter
datatype    = 't1'
skullstrip  = False
vrscomp     = 1                 # 1, 2
groupcomp   = 'CNManifestHD'    #'CNHD'
batch_size  = 1
epoch       = 20
fold        = 2

if __name__ == '__main__':

    if  len(sys.argv)!=4:
        groupcomp    = 'CNManifestHD' 
        datatype    = 't1'
        epoch       = 0
    else:
        groupcomp    = sys.argv[1]
        datatype    = sys.argv[2]
        epoch       = int(sys.argv[3])


    path_mdl = '/data/h_oguz_lab/hettk/MICCAI_2020'

    mean_auc    = np.zeros((4,1))
    mean_prec   = np.zeros((4,1))
    mean_acc    = np.zeros((4,1))
    mean_spe    = np.zeros((4,1))
    mean_sen    = np.zeros((4,1))

    for fold in range(1,5):
    
         # Load dataset
         path = '/data/h_oguz_lab/hettk/data/source/MNI-PREDICT-HD/'
         utils = dataset.Utils(path, 'Subset/demographics_3T_testing-fold-{}_{}_{}.csv'.format(1,groupcomp, vrscomp), 
                                     'Subset/data_3T_testing-fold-{}_{}_{}.csv'.format(1,groupcomp, vrscomp))
         testDataset = utils.create_dataset(datatype, skullstrip)
         testloader  = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=2)
         classes = ('Control','Disease')
         
         # Setting up the GPU
         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         # Assuming that we are on a CUDA machine, this should print a CUDA device:
         print(device)
        

         print('Comparison type = {}, data type = {}, epoch = {}, fold = {}'.format(groupcomp, datatype, epoch, fold))
        

         # Network architecture selection
         PATH = '{}/models/predict-hd_resenet_tmp_#{}_fold-{}_{}_{}_{}.pth'.format(path_mdl, epoch, fold, groupcomp, vrscomp, datatype)
         net = resnet3d.ResNet()
         net.load_state_dict(torch.load(PATH, map_location=device))
         net.to(device);
         
         correct     = 0
         total       = 0        
         sen         = 0
         sen_total   = 0
         spe         = 0
         spe_total   = 0
         
         y = np.array([])
         y_score = np.array([])
         with torch.no_grad():
             for data in testloader:
         
                 inputs, labels = data[0].to(device), data[1].to(device)
                 labels = Variable(labels).long()
                 outputs = net(inputs)
                 
                 # Accuracy on the training set
                 train_prob_predict = F.softmax(outputs, dim=1)
                 _, predicted = train_prob_predict.topk(1)
                 predicted = predicted.squeeze() 
                 
                 #y_score.append(train_prob_predict.cpu().numpy())
                 #y.append(labels.cpu().numpy())

                 if len(y)==0:
                    y_score = train_prob_predict.cpu().numpy()
                    y = labels.cpu().numpy()
                    y_ = [predicted.cpu().numpy()];
                 else:
                    y_score = np.concatenate((y_score, train_prob_predict.cpu().numpy()),axis=0)
                    y = np.concatenate((y, labels.cpu().numpy()), axis=0)
                    y_ = np.concatenate((y_, [predicted.cpu().numpy()]))
                
                 labels = labels.cpu()
                 predicted = predicted.cpu()

                 total   += labels.size(0)
                 correct += (predicted == labels).sum()
                 if labels.size(0)==1:
                     # Specificity 
                     spe         += ((predicted==0) * (labels[0]==0))
                     spe_total   += (labels[0]==0)
                     # Sensititivity
                     sen         += ((predicted==1) * (labels[0]==1))
                     sen_total   += (labels[0]==1)
                 else:
                     # Specificity 
                     spe         += ((predicted==0) * (labels==0)).sum()
                     spe_total   += (labels==0).sum()
                     # Sensititivity
                     sen         += ((predicted==1) * (labels==1)).sum()
                     print (((predicted==0) * (labels==0)).sum())
                     sen_total   += (labels==1).sum()
         
         fpr, tpr, thresholds = metrics.roc_curve(y, y_score[:,1], pos_label=1)
         auc = metrics.auc(fpr, tpr)
         precision = metrics.precision_score(y, y_)
         
         if fold==1:
           Y_score = train_prob_predict.cpu().numpy()
           Y = labels.cpu().numpy()
         else:
           Y_score = np.concatenate((Y_score, y_score),axis=0)
           Y = np.concatenate((Y, y), axis=0)

         mean_auc[fold-1] = auc
         mean_prec[fold-1] = precision
         mean_acc[fold-1] = float(correct)/total
         mean_spe[fold-1] = float(spe)/float(spe_total)
         mean_sen[fold-1] = float(sen)/float(sen_total)
         

         #print('Accuracy of the network on the test set')
         #print('         -- auc    : %.3f'% (auc))
         #print('         -- precision    : %.3f'% (precision))
         #print('         -- accuracy    : %.3f'% (float(correct)/total))
         #print('         -- specificity : %.3f'% (float(spe)))
         #print('         -- sensitivity : %.3f'% (float(sen)))

#FPR, TPR, thresholds = metrics.roc_curve(Y, Y_score[:,1], pos_label=1)




print('Accuracy of the network on the test set')
print('         -- auc    : %.3f - %.3f'% (mean_auc.mean(), mean_auc.std()))
print('         -- precision    : %.3f - %.3f'% (mean_prec.mean(),mean_prec.std()))
print('         -- accuracy    : %.3f - %.3f'% (mean_acc.mean(), mean_acc.std()))
print('         -- specificity : %.3f - %.3f'% (mean_spe.mean(), mean_spe.std()))
print('         -- sensitivity : %.3f - %.3f'% (mean_sen.mean(), mean_sen.std()))
    
