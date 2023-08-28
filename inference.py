import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

from models import get_model
from utils.training_utils import tensor2img

def parse_args():
    parser = argparse.ArgumentParser(description='Inference Blur2Vid')

    # Additional parameters
    parser.add_argument('--backbone', type=str, required=True, help='backbone')
    parser.add_argument('--blur_path', type=str, required=True, help='Path to blurry image')
    parser.add_argument('--output_path', type=str, help='Path to blurry image', default="output")
    parser.add_argument('--pretrained_path', type=str, help='pretrained path')
    parser.add_argument('--target_frames', nargs='+', type=int, default=None, help="Target frames")


    args = parser.parse_args()
    args.model_name = 'Blur2Vid'
    num_frames = 2

    args.model_kwargs = {
        'backbone_name': args.backbone,
        'backbone_kwargs': {
            'inference': True, 
            'num_frames': num_frames,
            'stage1_path': 'checkpoints/purohit.pth',
            'loss_kwargs': {
                'loss_type': 'order_inv',
                'mu': 0.02,
                'HyperCUT': {
                    'pretrained_path': "",
                    'f_func': 'ResnetIm2Vec',
                    'g_func': 'Concat',
                    'num_frames': num_frames,
                    'out_dim': 128,
                    'alpha': 0.2,
                }
            }
        }
    }

    return args

def read_img(blur_path):
    # Define preprocessing transforms for the image
    preprocess = transforms.Compose([ 
        transforms.ToTensor(),           # Convert PIL Image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize image
    ])

    # Load the image
    image = Image.open(blur_path).convert("RGB")

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Move the input tensor to the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)

    return {'blur_img': input_batch}


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda")

    model = get_model(args.model_name, args.model_kwargs).to(device)
    model.load_state_dict(torch.load(args.pretrained_path))

    blur_img = read_img(args.blur_path)

    # Perform inference
    with torch.no_grad():
        output = model(blur_img)['recon_frames'].squeeze(0)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for i in range(7):
        deblur_img = tensor2img(output[i])  
        deblur_img = Image.fromarray(deblur_img)

        image_filename = f"deblur_{i}.png"
        deblur_img.save(f"{args.output_path}/{image_filename}")
