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

import resnet3d
import dataset

if __name__ == '__main__':

    demo_list = sys.argv[1]
    data_list = sys.argv[2]
    path_img  = sys.argv[3]
    path_mdl  = sys.argv[4]
    cvp_fold  = sys.argv[5]

    # Load dataset
    utils = dataset.Utils(path_img, "{}/{}".format(cvp_fold, demo_list), "{}/{}".format(cvp_fold, data_list))
    trainDataset = utils.create_dataset()
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=2, shuffle=True, num_workers=1)
    
    # Setting up the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Network architecture selection
    net = resnet3d.ResNet()
    net.to(device)
    
    # Loss function 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-5)
    
    # Training
    for epoch in range(0,20):  # loop over the dataset multiple times
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
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = Variable(labels).long()
           
            optimizer.zero_grad()
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
       
        print('-----------------------------------------------------------------')
        print('Epoch #%d -- average loss: %.3f'% (epoch + 1, running_loss/niter))
        print('          -- accuracy    : %.3f'% (float(correct)/total))
        print('          -- specificity : %.3f'% (float(spe)/float(spe_total)))
        print('          -- sensitivity : %.3f'% (float(sen)/float(sen_total)))
        print('-----------------------------------------------------------------')
        
        PATH = '{}/models/predict-hd_resenet_tmp_#{}_cvpfold-{}_epoch{}.pth'.format(path_mdl, cvp_fold, epoch)
        torch.save(net.state_dict(), PATH)
    print('Finished Training')
    
    
    # Save trained model
    PATH = '{}/models/predict-hd_resenet_cvpfold-{}.pth'.format(path_mdl, cvp_fold)
    torch.save(net.state_dict(), PATH)
    
    



