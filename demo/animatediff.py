import torch
from diffusers import AnimateDiffPipeline, AutoencoderTiny, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# wangfuyun/AnimateLCM

TORCH_DEVICE = "mps"
TORCH_DTYPE = torch.float16
PROMPT = "light show, disco, techno"
STEPS = 4
# Load the motion adapter
adapter = MotionAdapter().to(TORCH_DEVICE, TORCH_DTYPE)
adapter.load_state_dict(
    load_file(
        hf_hub_download("ByteDance/AnimateDiff-Lightning", f"animatediff_lightning_{STEPS}step_diffusers.safetensors"),
    )
)
adapter = adapter.to(TORCH_DEVICE)

pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=TORCH_DTYPE)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

# pipe.scheduler = EulerDiscreteScheduler.from_config(
#     pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear"
# )

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora(
    fuse_unet=True,
    fuse_text_encoder=True,
    lora_scale=1.0,
    safe_fusing=False,
)
# enable memory savings
# pipe.enable_vae_slicing()

output = pipe(prompt=PROMPT, guidance_scale=1.0, num_inference_steps=STEPS)
frames = output.frames[0]
export_to_gif(frames, "/Users/matbreotten/Downloads/animation.gif")
