import torch
import os
import torch.nn as nn
import torch.optim as optim
from training import Trainer
from GDModels.Gmodel import SODModel
from GDModels.Dmodel import Discriminator
from GDModels.Weight_Init import *
from torch.utils.data import Dataset, DataLoader
from HDF5_Read import *

class SSALwithDARL:
    def __init__(self, args):
        # parameters
        self.lr = args.learning_rate
        self.epochs = args.epochs
        self.d_iter = args.d_iter
        self.batch_size = args.batch_size
        self.dicom_Height = args.dicom_Height
        self.dicom_Width = args.dicom_Width
        self.dataset = []
        self.dataloader = []

        self.lambda_gp = args.lambda_gp
        self.lambda_d_real = args.lambda_d_real
        self.lambda_d_fake = args.lambda_d_fake
        self.lambda_g_fake = args.lambda_g_fake
        self.lambda_mse = args.lambda_mse
        self.lambda_vgg = args.lambda_vgg
        self.lambda_giou = args.lambda_giou

    def train(self, args):  

        self.Traindataset = H5Dataset(args.input_dir_train)
        self.Traindataloader = torch.utils.data.DataLoader(
        self.Traindataset,
        batch_size=self.batch_size,
        num_workers=4, 
        shuffle=True,
        drop_last = True)

        self.generator = SODModel().to("cuda:0")
        self.discriminator = Discriminator(resnet = False, spectral_normed = True, num_rotation = 4,
                        channel = 1, ssup = True).to("cuda:0")

        # Initialize optimizers
        lr = 1e-4
        betas = (.9, .99)
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        ##CUDA:
        if torch.cuda.is_available():
            if ',' in args.gpu_ids:
                gpu_ids = [int(ids) for ids in args.gpu_ids.split(",")]
                print(gpu_ids)
            else:
                gpu_ids = int(args.gpu_ids)

            if type(gpu_ids) is not int:
                self.discriminator = nn.DataParallel(self.discriminator, device_ids = gpu_ids).cuda()
                self.generator = nn.DataParallel(self.generator, device_ids=gpu_ids).cuda()

            self.gpu = True
            self.use_cuda = True
    
        if not self.load_model():
            initialize_weights(self.generator)
            initialize_weights(self.discriminator)

        # Train model
        trainer = Trainer(self.generator, self.discriminator, self.G_optimizer, self.D_optimizer,
                        self.batch_size, self.dicom_Height, self.dicom_Width,
                        weight_rotation_loss_d = 1.0, weight_rotation_loss_g = 0.5,
                        use_cuda=torch.cuda.is_available())
        trainer.train(self.Traindataloader, self.epochs, save_training_gif=False)


    def load_model(self):
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_train', type=str, default="./TrainData.hdf5")
    parser.add_argument('--input_dir_test', type=str,default="./TestData.hdf5")
    parser.add_argument('--gpu_ids', type=str, default="0,1")

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--d_iter', type=int, default=5)
    parser.add_argument('--dicom_Height',type=int, default=256)
    parser.add_argument('--dicom_Width',type=int, default=256)

    args = parser.parse_args()

    mwgan = SSALwithDARL(args)
    # training
    mwgan.train(args)