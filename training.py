import imageio
import numpy as np
import torch
import os
from skimage import io
import cv2
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 batch_size, dicom_Height, dicom_Width,
                 weight_rotation_loss_d, weight_rotation_loss_g, gp_weight=10,
                 critic_iterations=5, print_every=50,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.batch_size = batch_size
        self.dicom_Height = dicom_Height
        self.dicom_Width = dicom_Width

        self.num_steps = 0
        self.d_iter = 5
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.weight_rotation_loss_d = weight_rotation_loss_d
        self.weight_rotation_loss_g = weight_rotation_loss_g
        self.rotate = 4
        global CountLossTrain 


    def _critic_train_iteration(self, data, generated_data, batch_size):
        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        _, d_real_pro_logits, d_real_rot_logits, d_real_rot_prob = self.D(data)
        _, g_fake_pro_logits, g_fake_rot_logits, g_fake_rot_prob = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = torch.sum(g_fake_pro_logits) * 0.001  + (- torch.sum(d_real_pro_logits) * 0.001)  + gradient_penalty

        # Add auxiiary rotation loss
        rot_labels = torch.zeros(self.rotate*batch_size).cuda()
        for i in range(self.rotate*batch_size):
            if i < batch_size:
                rot_labels[i] = 0
            elif i < 2*batch_size:
                rot_labels[i] = 1
            elif i < 3*batch_size:
                rot_labels[i] = 2
            else:
                rot_labels[i] = 3
        
        rot_labels = F.one_hot(rot_labels.to(torch.int64), self.rotate).float()
        d_real_class_loss = torch.sum(F.binary_cross_entropy_with_logits(
                                    input = d_real_rot_logits,
                                    target = rot_labels))
        self.losses['D_d_real_class_loss'].append(d_real_class_loss.data.sum().item())


        d_loss += self.weight_rotation_loss_d * d_real_class_loss
        #d_loss.backward()
        d_loss.backward(retain_graph=True)

        self.D_opt.step()


    def _generator_train_iteration(self, generated_data, batch_size):
        self.G_opt.zero_grad()
        self.D_opt.zero_grad()
        # Calculate loss and optimize
        _, g_fake_pro_logits, g_fake_rot_logits, g_fake_rot_prob = self.D(generated_data)
        g_loss = - torch.sum(g_fake_pro_logits) * 0.001

        # add auxiliary rotation loss
        rot_labels = torch.zeros(self.rotate*batch_size,).cuda()
        for i in range(self.rotate*batch_size):
            if i < batch_size:
                rot_labels[i] = 0
            elif i < 2*batch_size:
                rot_labels[i] = 1
            elif i < 3*batch_size:
                rot_labels[i] = 2
            else:
                rot_labels[i] = 3
        
        rot_labels = F.one_hot(rot_labels.to(torch.int64), self.rotate).float()
        g_fake_class_loss = torch.sum(F.binary_cross_entropy_with_logits(
            input = g_fake_rot_logits, 
            target = rot_labels))

        g_loss += self.weight_rotation_loss_g * g_fake_class_loss
        #g_loss.backward()
        g_loss.backward(retain_graph=True)
        self.G_opt.step()

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        _, prob_interpolated, _, _ = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, logger_train, epoch, D_init_loss, G_init_loss, CountLossTrain):

        with tqdm(total=len(data_loader.dataset)) as progress_bar:
            for i, (Ori_img, Seg_img, Name_img) in enumerate(data_loader):
              
                Ori_img = Variable(torch.unsqueeze(Ori_img, dim=1).float(), requires_grad=False)
                x = Ori_img.cuda()
                generated_data = self.sample_generator(x)
            
                x = generated_data
                x_90 = x.transpose(2,3)
                x_180 = x.flip(2,3)
                x_270 = x.transpose(2,3).flip(2,3)
                generated_data = torch.cat((x, x_90, x_180, x_270),0)

                x = Variable(torch.unsqueeze(Seg_img, dim=1).float(), requires_grad=False)
                x_90 = x.transpose(2,3)
                x_180 = x.flip(2,3)
                x_270 = x.transpose(2,3).flip(2,3)
                data = torch.cat((x,x_90, x_180, x_270),0)

                for _ in range(self.d_iter):
                    self._critic_train_iteration(data, generated_data, self.batch_size)
                     

                self._generator_train_iteration(generated_data, self.batch_size)
                
                self.num_steps = self.num_steps + 1
                CountLossTrain = CountLossTrain + 1

                progress_bar.update(self.batch_size)

        return CountLossTrain, G_init_loss, D_init_loss

    def train(self, data_loader, epochs, save_training_gif=True):

        D_init_loss = 10000
        G_init_loss = 10000
        CountLossTrain = 0 
        for epoch in range(epochs):
            self.G.train()
            self.D.train()
            print("\nEpoch {}".format(epoch + 1))
            CountLossTrain, G_init_loss, D_init_loss = self._train_epoch(data_loader, logger_train, epoch, D_init_loss, G_init_loss, CountLossTrain)

    def sample_generator(self, Ori_img):

        generated_data = self.G(Ori_img)
        return generated_data
