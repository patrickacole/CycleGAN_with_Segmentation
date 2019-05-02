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

def save_outputs(outputs, filenames_in, filenames_out, out_directory):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    for i in range(outputs.shape[0]):
        image = outputs[i]
        image = 128.0 * (image + 1.0) # [-1,1] -> [0, 255]
        image = np.clip(image, 0, 255)
        f_in, ext = os.path.splitext(os.path.basename(filenames_in[i]))
        f_out, ext = os.path.splitext(os.path.basename(filenames_out[i]))
        fname = os.path.join(out_directory, f"{f_in}_{f_out}{ext}")
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
    for i, (fx, data_x) in enumerate(dataloaderx):
        for j, (fy, data_y) in enumerate(dataloadery):
            data_x = data_x.view(-1,param.in_nc,param.image_size,param.image_size).to(device)
            data_y = data_y.view(-1,param.in_nc,param.image_size,param.image_size).to(device)
            realA = Variable(data_x)
            realB = Variable(data_y)

            fakeB, fakeA = model(data_x, data_y)
            fakeB = fakeB.data.cpu().numpy()
            fakeA = fakeA.data.cpu().numpy()

            # Save fakeB as fx->fy.jpg
            print(len(fx))
            print(len(fy))
            save_outputs(fakeB, fx, fy, param.out_directory)
            # Save fakeA as fy->fx.jpg
            save_outputs(fakeA, fy, fx, param.out_directory)

            del realA, realB, fakeA, fakeB
