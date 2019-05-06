import time
import argparse
import torch
import numpy as np
import numpy.linalg as la
from itertools import cycle
from torch.autograd import Variable

from models.CycleGAN import *
from utils.params import *
from utils.data_loader import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path", help="param file", type=str)
    args = parser.parse_args()
    print("Loading params...")
    param = Params(args.params_path)

    # load data
    print(f"Loading dataset {param.train_path_a.split('/')[1]}...")
    datasetx, datasety = get_datasets(param, train=True)

    print("Creating model...")
    device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])
    model = CycleGAN(param, device)
    model.train()

    dataloaderx = torch.utils.data.DataLoader(datasetx, batch_size=param.batch_size,
                                              shuffle=True, num_workers=param.num_workers)
    dataloadery = torch.utils.data.DataLoader(datasety, batch_size=param.batch_size,
                                              shuffle=True, num_workers=param.num_workers)

    for e in range(param.epochs):
        print(f'Starting epoch [{e+1} / {param.epochs}]')
        epoch_start_time = time.time()
        avgloss = 0.0
        if len(datasetx) < len(datasety):
            packed = zip(cycle(dataloaderx), dataloadery)
        else:
            packed = zip(cycle(dataloadery), dataloaderx)
        for i, (data_x, data_y) in enumerate(packed):
            realA, maskA = data_x
            realB, maskB = data_y
            realA = realA.view(-1,param.in_nc,param.image_size,param.image_size).to(device)
            realB = realB.view(-1,param.in_nc,param.image_size,param.image_size).to(device)
            realA = Variable(realA)
            realB = Variable(realB)

            model.optimize_parameters(e, realA, realB, param, maskA, maskB)

            avgloss += model.g_loss
            
            if i%20 == 0:
                print(f'Training loss at epoch {e+1} step {i}: {float(avgloss / (i + 1))}')

            del realA, realB

        epoch_end_time = time.time()
        print(f'Finishing epoch [{e+1} / {param.epochs}] in {epoch_end_time - epoch_start_time}s')
        model.save(e + 1, param)
