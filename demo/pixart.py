# %%
import torch
from diffusers import LCMScheduler, PixArtAlphaPipeline, Transformer2DModel
from peft import PeftModel


# Load LoRA
transformer = Transformer2DModel.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="transformer", torch_dtype=torch.float16
)
transformer = PeftModel.from_pretrained(transformer, "jasperai/flash-pixart")

# Pipeline
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS", transformer=transformer, torch_dtype=torch.float16
)

# Scheduler
pipe.scheduler = LCMScheduler.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    subfolder="scheduler",
    timestep_spacing="trailing",
)

pipe.to("mps")

prompt = "A raccoon reading a book in a lush forest."

image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
# %%
image
