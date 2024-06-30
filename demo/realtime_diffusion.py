# %%
import torch
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline


repo = "IDKiro/sdxs-512-dreamshaper"
seed = 42
weight_type = torch.float16  # or float32
device = "mps"
# Load model.


pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=weight_type).to(device)
vae_tiny = AutoencoderTiny.from_pretrained("IDKiro/sdxs-512-dreamshaper", subfolder="vae").to(
    device, dtype=weight_type
)
pipe.vae = vae_tiny
# pipe.set_progress_bar_config(leave=False)
# pipe.set_progress_bar_config(disable=True)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# %%

NEGATIVE_PROMPT = "dancing, worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
prompt = "geometric white light show, disco show, techno, high quality, 4K, realistic"
output_video_path = "/Users/matbreotten/Downloads/output_video_with_audio.mp4"
generator = torch.Generator(device="mps").manual_seed(seed)
# %%

pipe(
    prompt=prompt,
    negative_prompt=NEGATIVE_PROMPT,
    width=512,
    height=512,
    num_inference_steps=1,
    num_images_per_prompt=1,
    guidance_scale=0.0,
    generator=generator,
).images[0]

# %%
