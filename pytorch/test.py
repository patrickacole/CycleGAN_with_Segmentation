import os
import time
import argparse
import numpy as np
import numpy.linalg as la
import torch
from torch.autograd import Variable
from PIL import Image

from models.CycleGAN import *
from utils.params import *
from utils.data_loader import *

def save_outputs(outputs, filenames, out_directory, fake='A'):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    for i in range(outputs.shape[0]):
        image = outputs[i]
        image = 128.0 * (image + 1.0) # [-1,1] -> [0, 255]
        image = np.clip(image, 0, 255)
        f, ext = os.path.splitext(os.path.basename(filenames[i]))
        fname = os.path.join(out_directory, f"fake{fake}_{f}{ext}")
        image = Image.fromarray(image.astype(np.uint8), 'RGB')
        image.save(fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path", help="param file", type=str)
    args = parser.parse_args()
    print("Loading params...")
    param = Params(args.params_path)

    # load data
    print("Loading data...")
    datasetx, datasety = get_datasets(param, train=False)
    dataloaderx = torch.utils.data.DataLoader(datasetx, batch_size=param.batch_size,
                                              shuffle=False, num_workers=param.num_workers)
    dataloadery = torch.utils.data.DataLoader(datasety, batch_size=param.batch_size,
                                              shuffle=False, num_workers=param.num_workers)

    print("Creating model...")
    device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])
    model = CycleGAN(param, device)
    model.load(param)
    model.eval() # needs to be implemented, will essentailly just set all networks inside to eval

    print("Beginning to test...")
    if len(datasetx) < len(datasety):
        packed = zip(cycle(dataloaderx), dataloadery)
    else:
        packed = zip(cycle(dataloadery), dataloaderx)
    for i, (fx, data_x, fy, data_y) in enumerate(packed):
        data_x = data_x.view(-1,param.in_nc,param.image_size,param.image_size).to(device)
        data_y = data_y.view(-1,param.in_nc,param.image_size,param.image_size).to(device)
        realA = Variable(data_x)
        realB = Variable(data_y)

        fakeB, fakeA = model(data_x, data_y)
        fakeB = fakeB.data.cpu().numpy().transpose((0,2,3,1))
        fakeA = fakeA.data.cpu().numpy().transpose((0,2,3,1))

        # Save fakeB as fakeB_fx.jpg
        save_outputs(fakeB, fx, param.out_directory, fake='B')
        # Save fakeA as fakeA_fy.jpg
        save_outputs(fakeA, fy, param.out_directory, fake='A')

        del realA, realB, fakeA, fakeB
