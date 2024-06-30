# %%
from diffusers.pipelines import OnnxRuntimeModel
from huggingface_hub import snapshot_download
from optimum.onnxruntime import ORTStableDiffusionPipeline


taesd_dir = snapshot_download(repo_id="deinferno/taesd-onnx")

pipeline = ORTStableDiffusionPipeline.from_pretrained(
    "lemonteaa/sdxs-onnx",
    vae_decoder_session=OnnxRuntimeModel.from_pretrained(f"{taesd_dir}/vae_decoder"),
    vae_encoder_session=OnnxRuntimeModel.from_pretrained(f"{taesd_dir}/vae_encoder"),
    revision="onnx",
)
# %%
prompt = "Sailing ship in storm by Leonardo da Vinci"

image = pipeline(prompt, num_inference_steps=1, guidance_scale=0).images[0]

image.save("hello.png", "PNG")
