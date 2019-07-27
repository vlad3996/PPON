import torch
import numpy as np
from models.architecture import PPON
import matplotlib.pyplot as plt


def load_netG(checkpoint_path, device):
    netG = PPON(in_nc=3, nf=64, nb=24, out_nc=3, upscale=4)
    netG.load_state_dict(torch.load(checkpoint_path), strict=True)
    netG.to(device)
    #netG.eval()
    return netG


def prepare_image(input_image):
    if len(input_image.shape) < 3:
        input_image = input_image[..., np.newaxis]
        input_image = np.concatenate([input_image] * 3, 2)

    if input_image.shape[2] > 3:
        input_image = input_image[..., 0:3]

    out_image = input_image / 255.0
    out_image = np.transpose(out_image, (2, 0, 1))
    out_image = out_image[np.newaxis, ...]
    out_image = torch.from_numpy(out_image).float()
    return out_image


def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img


def get_images_from_net(netG_output):
    outputs = []
    for out in netG_output:
        out_tensor = out.detach().cpu().numpy().squeeze()
        img_out = convert_shape(out_tensor)
        outputs.append(img_out)
    return outputs


def infer_single_image(netG, input_image, device):

    prep_image = prepare_image(input_image)
    with torch.no_grad():
        netG_out = netG(prep_image.to(device))
        out_c, out_s, out_p = get_images_from_net(netG_out)
    return out_c, out_s, out_p


def show_plot(source_img, out_c, out_s, out_p, figsize=(18, 12)):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2, 5, 1)
    ax1.set_title('Original')
    ax1.imshow(source_img)
    ax2 = fig.add_subplot(2, 5, 2)
    ax2.set_title('SRc output')
    ax2.imshow(out_c)
    ax3 = fig.add_subplot(2, 5, 3)
    ax3.set_title('SRs output')
    ax3.imshow(out_s)
    ax4 = fig.add_subplot(2, 5, 4)
    ax4.set_title('SRp output')
    ax4.imshow(out_p)
    plt.show()
