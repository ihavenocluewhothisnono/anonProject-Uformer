import os
import sys




# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)




import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
print(opt)




import utils
from dataset.dataset_denoise import *
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True




import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx




from losses import CharbonnierLoss




from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler




# from utils.loader import  get_training_data,get_validation_data












######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, 'denoising', opt.dataset, opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt')
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)




# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)




######### Model ###########
model_restoration = utils.get_arch(opt)




with open(logname,'a') as f:
  f.write(str(opt)+'\n')
  f.write(str(model_restoration)+'\n')




######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
  optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
      optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
  raise Exception("Error optimizer...")








######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.cuda()
 




######### Scheduler ###########
if opt.warmup:
  print("Using warmup and cosine strategy!")
  warmup_epochs = opt.warmup_epochs
  scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
  scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
  scheduler.step()
else:
  step = 50
  print("Using StepLR,step={}!".format(step))
  scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
  scheduler.step()




######### Resume ###########
if opt.resume:
  path_chk_rest = opt.pretrain_weights
  print("Resume from "+path_chk_rest)
  utils.load_checkpoint(model_restoration,path_chk_rest)
  start_epoch = utils.load_start_epoch(path_chk_rest) + 1
  lr = utils.load_optim(optimizer, path_chk_rest)




  # for p in optimizer.param_groups: p['lr'] = lr
  # warmup = False
  # new_lr = lr
  # print('------------------------------------------------------------------------------')
  # print("==> Resuming Training with learning rate:",new_lr)
  # print('------------------------------------------------------------------------------')
  for i in range(1, start_epoch):
      scheduler.step()
  new_lr = scheduler.get_lr()[0]
  print('------------------------------------------------------------------------------')
  print("==> Resuming Training with learning rate:", new_lr)
  print('------------------------------------------------------------------------------')




  # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)




######### Loss ###########
criterion = CharbonnierLoss().cuda()




######### DataLoader ###########
# print('===> Loading datasets')
# img_options_train = {'patch_size':opt.train_ps}
# train_dataset = get_training_data(opt.train_dir, img_options_train)
# train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
#        num_workers=opt.train_workers, pin_memory=False, drop_last=False)
# val_dataset = get_validation_data(opt.val_dir)
# val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
#        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)


print('===> Loading datasets')


if opt.arch != 'UformerExtended':
   img_options_train = {'patch_size':opt.train_ps}
   train_dataset = get_training_data(opt.train_dir, img_options_train)
   train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
           num_workers=opt.train_workers, pin_memory=False, drop_last=False)
   val_dataset = get_validation_data(opt.val_dir)
   val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
           num_workers=opt.eval_workers, pin_memory=False, drop_last=False)


   len_trainset = train_dataset.__len__()
   len_valset = val_dataset.__len__()
   print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)


   with torch.no_grad():
       model_restoration.eval()
       psnr_dataset = []
       psnr_model_init = []
       for ii, data_val in enumerate(val_loader, 0):
           target = data_val[0].cuda()
           input_ = data_val[1].cuda()
           with torch.cuda.amp.autocast():
               restored = model_restoration(input_)
               restored = torch.clamp(restored, 0, 1)
           psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
           psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())


       psnr_dataset = sum(psnr_dataset) / len_valset
       psnr_model_init = sum(psnr_model_init) / len_valset
       print('Input & GT (PSNR) -->%.4f dB' % (psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB' % (psnr_model_init))


   print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
   best_psnr = 0
   best_epoch = 0
   best_iter = 0
   eval_now = len(train_loader)//4
   print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))


   loss_scaler = NativeScaler()
   torch.cuda.empty_cache()


   for epoch in range(start_epoch, opt.nepoch + 1):
       epoch_start_time = time.time()
       epoch_loss = 0
       model_restoration.train()


       for i, data in enumerate(tqdm(train_loader), 0):
           optimizer.zero_grad()
           target = data[0].cuda()
           input_ = data[1].cuda()
           if epoch > 5:
               target, input_ = utils.MixUp_AUG().aug(target, input_)
           with torch.cuda.amp.autocast():
               restored = model_restoration(input_)
               loss = criterion(restored, target)
           loss_scaler(loss, optimizer, parameters=model_restoration.parameters())
           epoch_loss += loss.item()


           if (i+1) % eval_now == 0 and i > 0:
               with torch.no_grad():
                   model_restoration.eval()
                   psnr_val_rgb = []
                   for ii, data_val in enumerate(val_loader, 0):
                       target = data_val[0].cuda()
                       input_ = data_val[1].cuda()
                       with torch.cuda.amp.autocast():
                           restored = model_restoration(input_)
                       restored = torch.clamp(restored, 0, 1)
                       psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
                   psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                   if psnr_val_rgb > best_psnr:
                       best_psnr = psnr_val_rgb
                       best_epoch = epoch
                       best_iter = i
                       torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir,"model_best.pth"))
                   print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                   with open(logname,'a') as f:
                       f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr)+'\n')
                   model_restoration.train()
                   torch.cuda.empty_cache()
       scheduler.step()
       print("------------------------------------------------------------------")
       print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
       print("------------------------------------------------------------------")
       with open(logname,'a') as f:
           f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0])+'\n')
       torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir,"model_latest.pth"))
       if epoch % opt.checkpoint == 0:
           torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch)))


print("Now time is : ", datetime.datetime.now().isoformat())


# Extension Begins
#itloads the colorization dataset, initializes the model, loss functions, and optimizer, and trains the model to map grayscale inputs to RGB outputs using both pixel-level and perceptual loss.
if opt.arch == 'UformerExtended':
   print("\n[INFO] Starting colorization extension...\n")
   from torchvision import transforms
   from dataset.dataset_colorization import ColorizationDataset
   from model import UformerExtended as ColorizationUformer
   from losses import perceptual_loss

   color_transform = transforms.Compose([
       transforms.Resize((opt.train_ps, opt.train_ps)),
       transforms.ToTensor()
   ])
   color_dataset = ColorizationDataset(root_dir=opt.train_dir, transform=color_transform)
   color_loader = DataLoader(color_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.train_workers, pin_memory=False, drop_last=False)
   color_model = ColorizationUformer(img_size=opt.train_ps, in_chans=1, out_chans=3).cuda()
   color_model = torch.nn.DataParallel(color_model)
   color_optimizer = optim.Adam(color_model.parameters(), lr=opt.lr_initial)
   color_criterion = CharbonnierLoss().cuda()


   print(f"Loaded {len(color_dataset)} grayscale-to-RGB training samples.")

    # begin the training loop of the model based on previous parameters
   for epoch in range(1, opt.nepoch + 1):
       color_model.train()
       total_loss = 0
       for i, (gray, rgb, _, _) in enumerate(tqdm(color_loader), 0):
           gray, rgb = gray.cuda(), rgb.cuda()
           with torch.cuda.amp.autocast():
               output = color_model(gray)
               loss = color_criterion(output, rgb) + 0.1 * perceptual_loss(output, rgb)
           color_optimizer.zero_grad()
           loss.backward()
           color_optimizer.step()
           total_loss += loss.item()

       avg_loss = total_loss / len(color_loader)
       print(f"[Colorization Epoch {epoch}/{opt.nepoch}] Loss: {avg_loss:.4f}")
       torch.save({
           'epoch': epoch,
           'state_dict': color_model.state_dict(),
           'optimizer': color_optimizer.state_dict()
       }, os.path.join(model_dir, f"colorization_epoch_{epoch}.pth"))


   print("[INFO] Colorization training complete.")
# end
