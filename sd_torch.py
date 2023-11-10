
# 1. 安装依赖：
# pip install diffusers==0.18.2
# pip install transformers
# pip install omegaconf
# pip install --pre openvino
# pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.1.0.dev20230809%2Bcpu-cp310-cp310-win_amd64.whl #也可以安装更新的版本
# 2. windows 需要修改eval_frame.py by commenting out the windows errors lines 
# 修改venv/Lib/site-packages/torch/_dynamo/eval_frames.py，找到line 484,注释高亮的两行， 因为windows 不支持torch.compile
# def check_if_dynamo_supported():
#    #if sys.platform == "win32":
#    #    raise RuntimeError("Windows not yet supported for torch.compile")
#    if sys.version_info >= (3, 11):
#        raise RuntimeError("Python 3.11+ not yet supported for torch.compile")


import os
import time
import torch
import numpy as np
from PIL import Image
import openvino.frontend.pytorch.torchdynamo.backend
from diffusers import  StableDiffusionPipeline, DPMSolverMultistepScheduler

os.environ["PYTORCH_TRACING_MODE"] = "TORCHFX"
#Adjust the ‘GPU’ parameter to ‘GPU.0’, ‘GPU.1’ etc. if there are multiple GPUs on the system
os.environ["OPENVINO_TORCH_BACKEND_DEVICE"] = "GPU" 
# enable caching
os.environ["OPENVINO_TORCH_MODEL_CACHING"] = "1"

checkpoint_path = "/home/adc2/crystal/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors"
local_config_file = "/home/adc2/crystal/stable-diffusion-webui/configs/v1-inference.yaml"
sd_model = StableDiffusionPipeline.from_single_file(checkpoint_path, local_config_file=local_config_file, load_safety_checker=False)
sd_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_model.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True) #DPM++ 2M Karras

sd_model.unet = torch.compile(sd_model.unet, backend="openvino")
sd_model.vae.decode = torch.compile(sd_model.vae.decode, backend="openvino")

prompt = "a red car in snowy forest"
negative_prompt = ""
steps = 20
cfg_scale = 7
#generator = [torch.Generator(device="cpu").manual_seed(s) for s in p.seeds]
#generator = [torch.Generator(device="cpu").manual_seed(1)]
def predict(prompt,negative_prompt, steps, cfg_scale):
    output = sd_model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    output_type="np",
                    width=512,
                    height=512,
            ).images[0]



    img_out = Image.fromarray((255. * output).astype(np.uint8))
    img_out.save("output.jpg")
    return img_out

start = time.time()
predict(prompt, negative_prompt, steps, cfg_scale)
print("Duration: ", time.time() - start)

start = time.time()
predict(prompt, negative_prompt, steps, cfg_scale)
print("Duration: ", time.time() - start)