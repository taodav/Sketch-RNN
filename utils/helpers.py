import torch
import shutil
from argparse import ArgumentParser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, is_best, filename='./checkpoints/model_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best.pth.tar')

def load_checkpoint(resume_path, model, optimizer):
    print("Loading checkpoint")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded checkpoint")

    return model, optimizer


def get_args():
    parser = ArgumentParser(description='Image Autoencoder Experiment')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sparse', dest='sparse', action='store_true')
    parser.add_argument('--vae', dest='vae', action='store_true')
    parser.add_argument('--d_hidden', type=int, default=8 * 3 * 3)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--beta', type=int, default=3)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume_path', type=str, default='./checkpoints/model_checkpoint.pth.tar')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--resume_snapshot', dest='resume_snapshot', action='store_true')
    parser.set_defaults(resume_snapshot=False)
    args = parser.parse_args()
    return args
