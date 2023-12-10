import argparse
from pathlib import Path
from openvino.runtime import Core
from diffusers import StableDiffusionPipeline


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="AUTO")
args = parser.parse_args()
device = args.device.upper()

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to("cpu")

# for reducing memory consumption get all components from pipeline independently
text_encoder = pipe.text_encoder
text_encoder.eval()
unet = pipe.unet
unet.eval()
vae = pipe.vae
vae.eval()

conf = pipe.scheduler.config

del pipe

# Define a dir to save text-to-image models
txt2img_model_dir = Path("/model/sd2.1")
txt2img_model_dir.mkdir(exist_ok=True)

from implementation.conversion_helper_utils import convert_encoder, convert_unet, convert_vae_decoder, convert_vae_encoder 

# Convert the Text-to-Image models from PyTorch -> Onnx -> OpenVINO
# 1. Convert the Text Encoder
txt_encoder_ov_path = txt2img_model_dir / "text_encoder.xml"
convert_encoder(text_encoder, txt_encoder_ov_path)
# 2. Convert the U-NET
unet_ov_path = txt2img_model_dir / "unet.xml"
convert_unet(unet, unet_ov_path, num_channels=4, width=96, height=96)
# 3. Convert the VAE encoder
vae_encoder_ov_path = txt2img_model_dir / "vae_encoder.xml"
convert_vae_encoder(vae, vae_encoder_ov_path, width=768, height=768)
# 4. Convert the VAE decoder
vae_decoder_ov_path = txt2img_model_dir / "vae_decoder.xml"
convert_vae_decoder(vae, vae_decoder_ov_path, width=96, height=96)


core = Core()

text_enc = core.compile_model(txt_encoder_ov_path, device)
unet_model = core.compile_model(unet_ov_path, device)
vae_encoder = core.compile_model(vae_encoder_ov_path, device)
vae_decoder = core.compile_model(vae_decoder_ov_path, device)

from diffusers.schedulers import LMSDiscreteScheduler
from transformers import CLIPTokenizer
from implementation.ov_stable_diffusion_pipeline import OVStableDiffusionPipeline

scheduler = LMSDiscreteScheduler.from_config(conf)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

ov_pipe = OVStableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_enc,
    unet=unet_model,
    vae_encoder=vae_encoder,
    vae_decoder=vae_decoder,
    scheduler=scheduler
)

import ipywidgets as widgets

text_prompt = "Shibuya city filled by many cats, epic vista, beautiful landscape, 4k, 8k"
negative_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"
num_steps = 25 # min=1, max=50, value=25
seed = 42 # min=0, max=10000000, value=42

# Run inference pipeline
import time
start = time.time()
result = ov_pipe(text_prompt, 
		negative_prompt=negative_prompt, 
		num_inference_steps=num_steps,
		seed=seed)
end = time.time()
print(end-start)

final_image = result['sample'][0]
final_image.save('result.png')
