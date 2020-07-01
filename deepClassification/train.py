import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import sys
from sklearn import metrics

import network
import resnet3d 
import dataset
import config


datatype    = 't1'
skullstrip  = False
vrscomp     = 1                 # 1, 2
groupcomp   = 'CNManifestHD'    #'CNHD' , 'CNManifestHD', 'CNPreManifestHD'
batch_size  = 2
n_epoch     = 20
s_epoch     = 0
fold        = 2


path_mdl    = '/data/h_oguz_lab/hettk/MICCAI_2020'

if __name__ == '__main__':

    if len(sys.argv)==2:
        groupcomp    = 'CNManifestHD' 
        datatype    = 't1'
    else:
        groupcomp    = sys.argv[1]
        datatype    = sys.argv[2]
    
    if len(sys.argv)==5:
        fold = int(sys.argv[4])
    print('Comparison type = {}, data type = {}, fold = {}'.format(groupcomp, datatype, fold))

    # Load dataset
    path = '/data/h_oguz_lab/hettk/data/source/MNI-PREDICT-HD/'
    utils = dataset.Utils(path, 'Subset/demographics_3T_training-fold-{}_{}_{}.csv'.format(fold, groupcomp, vrscomp), 
                                'Subset/data_3T_training-fold-{}_{}_{}.csv'.format(fold, groupcomp, vrscomp))
    trainDataset = utils.create_dataset(datatype, skullstrip)
    trainloader  = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    # Setting up the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    

    # Network architecture selection
    #net = network.SimpleNet()
    #net = network.AdvancedNet()
    net = resnet3d.ResNet()
    if len(sys.argv)==4:
        resepoch = int(sys.argv[3])
        if resepoch>0:
            PATH = '{}/models/predict-hd_resenet_tmp_#{}_{}_{}_{}.pth'.format(path_mdl,resepoch, groupcomp, vrscomp, datatype)
            net.load_state_dict(torch.load(PATH))
            print('loading from epoch #{}'.format(resepoch))
            n_epoch += resepoch
            s_epoch += resepoch
    net.to(device)
    
    # Loss function 
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    #optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
    
    
    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-5)
    
    # Training
    for epoch in range(s_epoch,n_epoch):  # loop over the dataset multiple times 
        correct = 0
        total = 0
        sen = 0
        sen_total = 0
        spe = 0
        spe_total = 0

        niter = 0
        running_loss = 0.0

        net.train()
        y = []
        y_score = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            #inputs, labels = data[0], data[1].to(device)
            labels = Variable(labels).long()
           
            '''Multichannel approach'''
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            niter        += 1 
            running_loss += loss
            
            # Accuracy on the training set
            train_prob_predict = F.softmax(outputs, dim=1)
            _, maj_predicted = train_prob_predict.topk(1)
            maj_predicted = maj_predicted.squeeze()
            
            y_score.append(train_prob_predict.cpu())
            y.append(labels.cpu().numpy())

            total   += labels.size(0)
            correct += (maj_predicted == labels).sum()

            if labels.size(0)==1:
                # Specificity 
                spe         = spe + (maj_predicted==0 * labels==0)
                spe_total   = spe_total + (labels==0)
                # Sensititivity
                sen         = sen + (maj_predicted==1 * labels==1)
                sen_total   = sen_total + (labels==1)
            else:
                # Specificity 
                spe         += (maj_predicted[labels==0]==0).sum()
                spe_total   += (labels==0).sum()
                # Sensititivity
                sen         += (maj_predicted[labels==1]==1).sum()
                sen_total   += (labels==1).sum()

            if i % 10 == 9: 
                print('[%d, %5d] loss: %.3f - acc : %3.f'% (epoch+1, i+1, running_loss/niter,(float(correct)/total)))
       
        #fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
        print('-----------------------------------------------------------------')
        print('Epoch #%d -- average loss: %.3f'% (epoch + 1, running_loss/niter))
        #print('         -- auc         : %.3f'% (float(auc))
        print('         -- accuracy    : %.3f'% (float(correct)/total))
        print('         -- specificity : %.3f'% (float(spe)/float(spe_total)))
        print('         -- sensitivity : %.3f'% (float(sen)/float(sen_total)))
        print('-----------------------------------------------------------------')
        
        PATH = '{}/models/predict-hd_resenet_tmp_#{}_fold-{}_{}_{}_{}.pth'.format(path_mdl,epoch, fold, groupcomp, vrscomp, datatype)
        torch.save(net.state_dict(), PATH)

    print('Finished Training')
    
    
    # Save trained model
    PATH = '{}/models/predict-hd_resenet_fold-{}_{}_{}_{}.pth'.format(path_mdl,fold, groupcomp, vrscomp, datatype)
    torch.save(net.state_dict(), PATH)
    
    
   





''' Single channel approach '''
## Store predicted for each patch 
#predicted = np.zeros((labels.size(0), 2, inputs.size(1)))
#for k in range(inputs.size(1)):
#    for j in range(labels.size(0)):
#        patch_ = inputs[j,k,:,:,:]
#        patch_ = torch.unsqueeze(patch_,0)
#        patch_ = patch_.view(-1,1,64,64,64)
#        if j==0:
#            patch = patch_
#        else:
#            patch = torch.cat((patch,patch_))
#    patch = patch.to(device)
#    
#    # zero the parameter gradients
#    optimizer.zero_grad()
#    
#    # forward + backward + optimize
#    outputs = net(patch)
#    loss    = criterion(outputs, labels)
#    loss.backward()
#    optimizer.step()
#        
#    # print statistics
#    niter        += 1 
#    running_loss += loss
#    
#    # Accuracy on the training set
#    train_prob_predict = F.softmax(outputs, dim=1)
#    _, pred = train_prob_predict.topk(1)
#    pred = pred.squeeze()
#    if labels.size(0)>1:
#        for j in range(labels.size(0)):
#            predicted[j,pred[j],k] += 1
#    else:
#        predicted[0, pred, k] += 1

#predicted = predicted.sum(axis=2)
#maj_predicted = np.zeros(labels.size(0))
#for j in range(labels.size(0)):
#    maj_predicted[j] = np.argmax(predicted[j,:])

#maj_predicted = torch.from_numpy(maj_predicted)
#maj_predicted = (Variable(maj_predicted).long()).cuda()
