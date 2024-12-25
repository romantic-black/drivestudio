import os
import sys
import glob
import argparse
import torch
import gc
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from third_party.OSEDiff.osediff import OSEDiff_test
from third_party.OSEDiff.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from third_party.OSEDiff.ram.models.ram_lora import ram
from third_party.OSEDiff.ram import inference_ram as inference

class Args:
    pretrained_model_name_or_path = "/root/autodl-tmp/models/models--stabilityai--stable-diffusion-2-1-base"
    seed = 42
    process_size = 512
    upscale = 1
    align_method = 'adain'
    osediff_path = '/root/autodl-tmp/models/model_105001.pkl'
    prompt = ''
    ram_path = "/root/autodl-tmp/models/ram_swin_large_14m.pth"
    ram_ft_path = "/root/autodl-tmp/models/DAPE.pth"
    save_prompts = True
    mixed_precision = 'fp16'
    merge_and_unload_lora = False
    vae_decoder_tiled_size = 224
    vae_encoder_tiled_size = 1024
    latent_tiled_size = 96
    latent_tiled_overlap = 32

neg_prompts = [
'man', 'crowded', 'person', 'walk','pick up','skate', 'skier', 'woman', 'boy', 'girl','umbrella','pedestrian','motorbike','scooter','motorcyclist',
                                  'drive', 'car','taxi', 'jeep', 'vehicle', 'suv', 'minivan',
                                  'van', 'snowstorm', 'snow', 'snowy', 'blue', 'blanket', 'toy car', 'motorcycle', 'park', 'parking garage',
                                  'white','black',  'fill', 'night', 'rainy', 'rain'
]

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=torch.float16)
    captions = inference(lq_ram, model)
    prompts = f"{captions[0]}, {args.prompt},"
    # validation_prompt ="road, street, building"
    validation_prompt = []
    for prompt in prompts.split(','):
        if prompt.strip() not in neg_prompts:
            validation_prompt.append(prompt)

    validation_prompt = ','.join(validation_prompt)
    return validation_prompt, lq

class OSEDiffInfer:
    def __init__(self, args=Args):
        DAPE = ram(pretrained=args.ram_path,
                   pretrained_condition=args.ram_ft_path,
                   image_size=384,
                   vit='swin_l')
        DAPE.eval()
        DAPE.to("cuda")

        self.DAPE = DAPE.to(dtype=torch.float16)
        self.model = OSEDiff_test(args)
        self.args = args


    def infer(self, input_dir, output_dir):
        args = self.args
        os.makedirs(output_dir, exist_ok=True)
        if os.path.isdir(input_dir):
            image_names = sorted(glob.glob(f'{input_dir}/*.png'))
        else:
            image_names = [input_dir]

        for image_name in image_names:
            # make sure that the input image is a multiple of 8
            input_image = Image.open(image_name).convert('RGB')
            ori_width, ori_height = input_image.size
            rscale = args.upscale
            resize_flag = False
            if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
                scale = (args.process_size // rscale) / min(ori_width, ori_height)
                input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
                resize_flag = True
            input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))

            new_width = input_image.width - input_image.width % 8
            new_height = input_image.height - input_image.height % 8
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            bname = os.path.basename(image_name)

            # get caption
            validation_prompt, lq = get_validation_prompt(args, input_image, self.DAPE)
            print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8'))

            # translate the image
            with torch.no_grad():
                lq = lq * 2 - 1
                output_image = self.model(lq, prompt=validation_prompt)
                output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
                if args.align_method == 'adain':
                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                elif args.align_method == 'wavelet':
                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                else:
                    pass
                if resize_flag:
                    output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))

            output_pil.save(os.path.join(output_dir, bname))

    def clear_model(self):
        # 将模型移动到 CPU
        self.model.to('cpu')
        self.DAPE.to('cpu')

        # 删除引用
        del self.model
        del self.DAPE

        torch.cuda.empty_cache()







