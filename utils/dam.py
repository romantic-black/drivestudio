from transformers import pipeline
from PIL import Image
import os
import glob

class Args:
    pretrained_model_path = "/root/autodl-tmp/models/Depth-Anything-V2-Large-hf"


class DamInfer:
    def __init__(self, args=Args):
        self.args = args
        self.dam = pipeline(task="depth-estimation", model=args.pretrained_model_path)

    def infer(self, input_dir, output_dir):
        args = self.args
        os.makedirs(output_dir, exist_ok=True)
        if os.path.isdir(input_dir):
            image_names = sorted(glob.glob(f'{input_dir}/*.png'))
        else:
            image_names = [input_dir]

        for image_name in image_names:
            input_image = Image.open(image_name).convert('RGB')
            depth = self.dam(input_image)["depth"]
            save_path = os.path.join(output_dir, os.path.basename(image_name))
            depth.save(save_path)

    def clear_model(self):
        self.dam = pipeline(task="depth-estimation")
