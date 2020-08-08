import torch
import sys
import argparse
import os
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
# import matplotlib.pyplot as plt
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import logging
import time
### My libs
from dataset import DAVIS_MO_Train, DAVIS_MO_Val, Youtube_MO_Train, Tianchi_MO_Train
from model import STM
from helpers import *
import random

# Constants
MODEL_DIR = '../user_data/model_data/'
os.makedirs(MODEL_DIR, exist_ok=True)
NUM_EPOCHS = 100000


def main():
	print(">>>>\nMy STM training starting...")
	print('Python version: ', sys.version)
	print('Pytorch version: ', torch.__version__)
	run_train()


def get_arguments():
	"""
    Parse input arguments
    """
	parser = argparse.ArgumentParser(description="SST")
	parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
	parser.add_argument("-s", type=str, help="set", required=True)
	parser.add_argument("-y", type=int, help="year", required=True)
	parser.add_argument("-viz", help="Save visualization", action="store_true")
	parser.add_argument("-D", type=str, help="path to data", default='../data/')
	# new ones:

	# resume trained model
	parser.add_argument('--loadepoch', dest='loadepoch', help='epoch to load model',
						default=-1, type=int)
	parser.add_argument('--output_dir', dest='output_dir',
						help='output directory',
						default='checkpoint', type=str)
	# config
	parser.add_argument('--epochs', dest='num_epochs',
						help='number of epochs to train',
						default=NUM_EPOCHS, type=int)
	parser.add_argument('--lr', dest='lr',
						help='starting learning rate',
						default=1e-5, type=float)
	parser.add_argument('--eval_epoch', dest='eval_epoch',
						help='interval of epochs to perform validation',
						default=10, type=int)

	# return
	return parser.parse_args()


def run_train():
    # get arguments
    args = get_arguments()
    GPU = args.g
    YEAR = args.y
    SET = args.s
    # VIZ = args.viz
    DATA_ROOT = args.D

    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Using Dataset DAVIS', YEAR)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    print('--- CUDA:')

    # Device infos
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_devices = torch.cuda.device_count()
        print('Cuda version: ', torch.version.cuda)
        print('Current GPU id: ', torch.cuda.current_device())
        print('Device name: ', torch.cuda.get_device_name(device=torch.cuda.current_device()))
        print('Number of available devices:', num_devices)
    else:
        print('GPU is not available. CPU will be used.')
        device = torch.device("cpu")
        num_devices = 1

    # batch sizes
    train_batch_size = num_devices
    val_batch_size = 12
    Trainset = Tianchi_MO_Train(DATA_ROOT, imset='{}.txt'.format(SET), single_object=(YEAR == 16))
    Trainloader = data.DataLoader(Trainset, batch_size=train_batch_size, shuffle=False, num_workers=4)

    # # DAVIS data loader
	# Trainset = DAVIS_MO_Train(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR, SET),
	# 						  single_object=(YEAR == 16))
	# # Trainloader = data.DataLoader(Trainset, batch_size=train_batch_size, shuffle=False, num_workers=2)
    #
	# # Youtube data loader
	# youtube_Trainset = Youtube_MO_Train(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR, SET),
	# 									single_object=(YEAR == 16))
	# Trainloader = data.DataLoader(youtube_Trainset, batch_size=train_batch_size, shuffle=False, num_workers=1)
    #
	# Valset = DAVIS_MO_Val(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR, SET),
	# 					  single_object=(YEAR == 16))
	# Valloader = data.DataLoader(Valset, batch_size=val_batch_size, shuffle=False, num_workers=4)
    #
	# Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
	# Testloader = data.DataLoader(Testset, batch_size=1, shuffle=True, num_workers=1)

	# Isntantiate the model
    model = STM()
    print("Model instantiated")
    if torch.cuda.is_available():
        model = model.to(device)
        print("Model sent to cuda")
        if num_devices > 1:
            model = nn.DataParallel(model)
            print("Model parallelised in {} GPUs".format(num_devices))

	# parameters, optmizer and loss
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value]}]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter()
    start_epoch = 0
    iters_per_epoch = len(Trainloader)

    # load saved model if specified
    if args.loadepoch >= 0:
        print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, args.loadepoch, font.END))
        # diretory name where models are saved
        load_name = "../user_data/model_data/STM_weights.pth" # saved_models\#.pth
		# get the state dict of current model
        state = model.state_dict()
        # load entire saved model from checkpoint
        checkpoint = torch.load(load_name)  # dict_keys(['epoch', 'model', 'optimizer'])
        # set next epoch to resume the training
        start_epoch = 1 # checkpoint['epoch'] + 1
        # filter out unnecessary keys from checkpoint
        checkpoint = {k: v for k, v in checkpoint.items() if k in state}
        # overwrite entries in the existing state dict
        state.update(checkpoint)
        # load the new state dict
        model.load_state_dict(state)
        # load optimizere state dict
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()
        print('  - complete!')


    # loop for training and validation
    for epoch in range(start_epoch, args.num_epochs):

        # training
        model.eval()  # set eval mode to disable batchnorm and dropout
        print('[Train] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))

        # increases maximum frame skip by 5 (range 5->25) after 20 epochs
        Trainset.Set_frame_skip(epoch)

        for seq, V in enumerate(Trainloader):

            ############# interrupção só para testar
            if seq > 4:
                break

            Fs, Ms, num_objects, info = V
            # batch_size = 4:
            # Fs:  torch.Size([4, 3, 3, 100, 100])
            # Ms:  torch.Size([4, 11, 3, 100, 100])
            # num_objects:  tensor([[1],[1],[1],[1]], device='cuda:0')

            # send input tensors to gpu
            if torch.cuda.is_available():
                Fs = Fs.to(device)
                Ms = Ms.to(device)
                num_objects = num_objects.to(device)

            loss = 0
            Es = torch.zeros_like(Ms)
            Es[:, :, 0] = Ms[:, :, 0]
            # Es:  torch.Size([4, 11, 3, 100, 100])

            # loop over the 3-1 frame+annotation samples (1st frame is used as reference)
            for t in range(1, 3):

                # memorize
                prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))
                # prev_key(k4):  torch.Size([4, 11, 128, 1, 30, 54])
                # prev_value(v4):  torch.Size([4, 11, 512, 1, 30, 54])

                if t - 1 == 0:
                    this_keys, this_values = prev_key, prev_value  # only prev memory
                else:
                    this_keys = torch.cat([keys, prev_key], dim=3)
                    this_values = torch.cat([values, prev_value], dim=3)
                # t = 1:
                # this_keys:  torch.Size([4, 11, 128, 1, 30, 54])
                # this_values:  torch.Size([4, 11, 512, 1, 30, 54])

                # segment
                logit = model(Fs[:, :, t], this_keys, this_values, torch.tensor([num_objects]))
                # logit: torch.Size([4, 11, 480, 910])

                # compute probability maps for output
                Es[:, :, t] = F.softmax(logit, dim=1)

                # update
                keys, values = this_keys, this_values

                # compute loss
                loss += criterion(Es[:, :, t].clone(), Ms[:, :, t].float()) / train_batch_size
                # Es:  torch.Size([4, 11, 3, 480, 854])
                # Ms:  torch.Size([4, 11, 3, 480, 854])

            # backprop
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # logging and display
            # if (seq+1) % args.disp_interval == 0:
            if (seq + 1) % 1 == 0:
                mean_iou = iou(Es[:, :, 1:3], Ms[:, :, 1:3])
                writer.add_scalar('Train/BCE', loss, seq + epoch * iters_per_epoch)
                writer.add_scalar('Train/IOU', mean_iou, seq + epoch * iters_per_epoch)
                print('[TRAIN] idx: {}, loss: {}, iou: {}'.format(seq, loss, mean_iou))

                # print("iteration: {}/{} ".format(seq,iters_per_epoch))

        # saving checkpoint
        if epoch % 1000 == 0 and epoch > 0:
            save_name = '{}/{}.pth'.format(MODEL_DIR, epoch)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), }, save_name)
            print('Model saved in: {}'.format(save_name))

        # # validation
        # if epoch % args.eval_epoch == 0:
        #     with torch.no_grad():
        #         print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
        #         model.eval()
        #         run_validate(model, criterion, Valloader, device, Mem_every=5, Mem_number=None)
        #         print('  - complete!')

        # print("The End")

    # def run_validate(model, criterion, Valloader, device, Mem_every=None, Mem_number=None):
# 	# model = STM()
# 	# Fs:  torch.Size([1, 3, 69, 480, 910])
# 	# Ms:  torch.Size([1, 11, 69, 480, 910])
# 	# num_frames: 69
# 	# num_objects:  2
#
# 	idx = 0
# 	next_change = 0
# 	to_second_frame = False
#
# 	for seq, V in enumerate(Valloader):
#
# 		############# interrupção só para testar
# 		if seq > 4:
# 			break
#
# 		Fss, Mss, nums_objects, infos = V
# 		nums_frames = infos['num_frames']
# 		# Fss:  torch.Size([4, 3, 1, 100, 100])
# 		# Mss:  torch.Size([4, 11, 1, 100, 100])
# 		# nums_objects:  tensor([[1],[1],[1],[1]])
# 		# infos:  {'name': ['bear', 'bear', 'bear', 'bear'],
# 		#       'num_frames': tensor([82, 82, 82, 82]),
# 		#       'size_480p': [tensor([100, 100, 100, 100]), tensor([100, 100, 100, 100])]}
#
# 		if torch.cuda.is_available():
# 			Fss = Fss.to(device)
# 			Mss = Mss.to(device)
# 			nums_objects = nums_objects.to(device)
#
# 		batch_size = int(Fss.size(0))
# 		for batch_idx in range(batch_size):
#
# 			Fs, Ms = Fss[batch_idx, :, 0], Mss[batch_idx, :, 0]
# 			Fs = torch.unsqueeze(Fs, dim=0)
# 			Ms = torch.unsqueeze(Ms, dim=0)
# 			num_objects = nums_objects[batch_idx]
# 			num_frames = nums_frames[batch_idx].item()
#
# 			# New sequence begins
# 			if idx == next_change:
# 				next_change += num_frames
# 				if Mem_every:
# 					to_memorize = [int(i) for i in np.arange(idx, next_change, step=Mem_every)]
# 				elif Mem_number:
# 					to_memorize = [int(round(i)) for i in np.linspace(idx, next_change, num=Mem_number + 2)[:-1]]
# 				else:
# 					raise NotImplementedError
# 				# Se mem_every=5, então to_memorize = [0, 5, 10, 15...]
# 				loss = 0
# 				first_frame = True
# 				second_frame = False
# 				to_second_frame = True
# 			elif to_second_frame:
# 				first_frame = False
# 				second_frame = True
# 				to_second_frame = False
# 			else:
# 				first_frame = False
# 				second_frame = False
# 				to_second_frame = False
#
# 			if not first_frame:
#
# 				# memorize
# 				with torch.no_grad():
# 					prev_key, prev_value = model(prev_Fs, prev_Ms, torch.tensor([num_objects]))
# 					# prev_key(k4):  torch.Size([1, 11, 128, 1, 30, 57])
# 					# prev_value(v4):  torch.Size([1, 11, 512, 1, 30, 57])
#
# 				if second_frame:  #
# 					this_keys, this_values = prev_key, prev_value  # only prev memory
# 				else:
# 					this_keys = torch.cat([keys, prev_key], dim=3)
# 					this_values = torch.cat([values, prev_value], dim=3)
#
# 					# segment
# 				with torch.no_grad():
# 					logit = model(Fs, this_keys, this_values, torch.tensor([num_objects]))
# 					# (t=39) logit: torch.Size([1, 11, 480, 910])
#
# 					pred = F.softmax(logit, dim=1)
# 					# (t=39) Es: torch.Size([1, 11, 69, 480, 910])
#
# 					# compute loss
# 					loss += criterion(pred, Ms.float())
#
# 				# update
# 				if second_frame or idx in to_memorize:
# 					keys, values = this_keys, this_values
#
# 			prev_Fs = Fs
# 			prev_Ms = Ms
# 			idx += 1
# 			print("[VAL] idx: {}, loss: {}".format(idx, loss))
#
# 		if idx == next_change:
# 			loss /= num_frames - 1
# 			print('val loss: {}'.format(loss))


if __name__ == "__main__":
	main()