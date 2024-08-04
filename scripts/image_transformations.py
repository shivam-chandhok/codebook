import argparse
from pathlib import Path
import random
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

import src.shape_texture_neuron.generate_stylized_voc.net as net
from src.shape_texture_neuron.generate_stylized_voc.function import adaptive_instance_normalization, coral
#from src.shape_texture_neuron.generate_stylized_voc import net
#from src.shape_texture_neuron.generate_stylized_voc.function import adaptive_instance_normalization, coral
#print(net.vgg)
path = "/raid/shivam_RA_Vineethsir/3dlm/final_notebooks/3DLM/research_modules/codebook/src/shape_texture_neuron/generate_stylized_voc/selected_textures"
vgg_path ='/raid/shivam_RA_Vineethsir/3dlm/final_notebooks/3DLM/research_modules/codebook/src/shape_texture_neuron/generate_stylized_voc/models/vgg_normalised.pth'
decoder_path = '/raid/shivam_RA_Vineethsir/3dlm/final_notebooks/3DLM/research_modules/codebook/src/shape_texture_neuron/generate_stylized_voc/models/decoder.pth'

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        print(size)
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)

    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def stylize_images(content_dir, style_dir, output_dir, vgg_path, decoder_path, content_size = 512, style_size=512, crop=False, preserve_color=False, alpha = 1, device="cuda"):
    ## source https://github.com/islamamirul/shape_texture_neuron
    ## alpha controls degree of stylization (0,1)
    style_dir = Path(style_dir)
    output_dir =Path(output_dir) 
    style_files = sorted(list(style_dir.glob("*.jpg")))
    output_dir.mkdir(exist_ok=True, parents=True)

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path))
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)
    
    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    # Storing the styles
    styles = []
    for style_path in style_files:
        style = style_tf(Image.open(str(style_path)))
        if preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        styles.append(style)
    
    for img_path in Path(content_dir).glob("*"):
        print(img_path)
        
        content = content_tf(Image.open(str(img_path))).squeeze()
        content = content.to(device).unsqueeze(0)
        for i, style in enumerate(styles):
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        alpha)
            output = output.squeeze().permute(1, 2, 0).cpu().numpy()

            output = (output * 255).astype(np.uint8)

            out_path = output_dir / (str(i) + "_" + str(img_path.name))
            plt.imsave(str(out_path), output)

stylize_images(path, path, "../results",vgg_path,decoder_path)


    
    



def background_changes(original_image, binary_mask, transform= 'sioulette_white_background'):
    # Convert PIL images to numpy arrays
    if transform == 'red_circle':
        image = original_image
        mask = (np.mean(binary_mask,-1)>=7).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Get the center and radius of the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (255, 0, 0), 2)  # Red color (BGR), thickness = 2
        output_array = image
    
    elif transform == 'red_circle_white':
        original_array = original_image
        mask_array = binary_mask>=7
        output_array = np.zeros_like(original_array)
        output_array.fill(255)
        object_pixels = original_array * mask_array
        output_array[mask_array > 0] = object_pixels[mask_array > 0]

        image = output_array
        mask = (np.mean(binary_mask,-1)>=7).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Get the center and radius of the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (255, 0, 0), 2)  # Red color (BGR), thickness = 2
        output_array = image
        
    elif transform == 'black_background':
        original_array = original_image
        mask_array = binary_mask>=7
        output_array = original_array*mask_array
    elif transform == 'sioulette':
        original_array = original_image
        mask_array = binary_mask>=7
        output_array = original_array*(~mask_array)
    elif transform == 'white_background':
        original_array = original_image
        mask_array = binary_mask>=7
        output_array = np.zeros_like(original_array)
        output_array.fill(255)
        object_pixels = original_array * mask_array
        output_array[mask_array > 0] = object_pixels[mask_array > 0]
    elif transform == 'sioulette_white_background':
        original_array = original_image
        mask_array = binary_mask>=7
        original_array = original_array*(~mask_array)
        output_array = np.zeros_like(original_array)
        output_array.fill(255)
        object_pixels = original_array * mask_array
        output_array[mask_array > 0] = object_pixels[mask_array > 0]
    elif transform in ['edge', 'superimposed_edge', 'superimposed_edge_white']:
        if transform == 'superimposed_edge_white':
            original_array = original_image
            mask_array = binary_mask>=7
            output_array = np.zeros_like(original_array)
            output_array.fill(255)
            object_pixels = original_array * mask_array
            output_array[mask_array > 0] = object_pixels[mask_array > 0]
            original_image = output_array


        model = RCF().cuda()
        checkpoint = torch.load('./src/rcf/bsds500_pascal_model.pth')
        model.load_state_dict(checkpoint)

        model.eval()
        scale = [0.5, 1, 1.5]

        mean = np.array([122.67891434, 116.66876762, 104.00698793], dtype=np.float32)
        image = np.array(original_image, dtype=np.float32)-mean


        H, W, _ = image.shape
        in_ = image
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))

            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res

        ms_fuse = ms_fuse / len(scale)

        output_array = ((1 - ms_fuse) * 255)
        output_array = np.repeat(output_array[:, :, np.newaxis], 3, axis=2)
        if transform in ['superimposed_edge','superimposed_edge_white'] :
            output_array = (output_array>=128)*255
            output_array = 0.5*output_array.astype(np.uint8)+0.5*np.array(original_image, dtype=np.float32)
    else:
        print("No transformation selected. Returning original image!!!")
        output_array = original_image
        
        
    output_image = Image.fromarray(output_array.astype(np.uint8))
    # output_image.save(f'{transform}.png')
    return output_image
