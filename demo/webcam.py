# %%
import cv2
import numpy as np
import torch
from diffusers import AutoencoderTiny, ControlNetModel, StableDiffusionPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


PROMPT = "a close-up picture of an old man standing in the rain"  # GOOD
NEGATIVE_PROMPT = ""

WIDTH, HEIGHT = 512, 512
TORCH_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)
TORCH_DTYPE = torch.float16
CTLNET_MODEL_LOC = "IDKiro/sdxs-512-dreamshaper-sketch"
MODEL_LOC = "IDKiro/sdxs-512-dreamshaper"


def get_result_and_mask(frame, center_x, center_y, width, height):
    "just gets full frame and the mask for cutout"

    mask = np.zeros_like(frame)
    mask[center_y : center_y + height, center_x : center_x + width, :] = 255
    cutout = frame[center_y : center_y + height, center_x : center_x + width, :]

    return frame, cutout


controlnet = ControlNetModel.from_pretrained(
    CTLNET_MODEL_LOC,
    torch_dtype=TORCH_DTYPE,
).to(TORCH_DEVICE)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_LOC,
    torch_dtype=TORCH_DTYPE,
    controlnet=controlnet,
    safety_checker=None,
).to(device=TORCH_DEVICE)


stream = StreamDiffusion(
    pipe,
    t_index_list=[1],
    torch_dtype=TORCH_DTYPE,
    do_add_noise=False,
    use_denoising_batch=True,
    cfg_type="self",
    frame_buffer_size=1,
)

# If the loaded model is not LCM, merge LCM
# stream.load_lcm_lora()
# stream.fuse_lora()

# Use Tiny VAE for further acceleration
pipe.vae = AutoencoderTiny.from_pretrained("IDKiro/sdxs-512-dreamshaper", subfolder="vae").to(
    TORCH_DEVICE, dtype=TORCH_DTYPE
)


# Prepare the stream
stream.prepare(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=50,
    guidance_scale=0.0,
    strength=0.8,
    generator=torch.Generator(TORCH_DEVICE),
    seed=42,
)

# optional
stream.enable_similar_image_filter(
    # similar_image_filter_threshold,
    # similar_image_filter_max_skip_frame
)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

CAP_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 320
CAP_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 240

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH / 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT / 2)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read a frame from the webcam (for warmup)
ret, image = cap.read()
center_x = (image.shape[1] - WIDTH) // 2
center_y = (image.shape[0] - HEIGHT) // 2
result_image, image_cutout = get_result_and_mask(image, center_x, center_y, WIDTH, HEIGHT)

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(1):
    stream(image_cutout)

# Run the stream infinitely
while True:
    # Read frame (image) from the webcam
    ret, frame = cap.read()

    # Break the loop if reading the frame fails
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # get center
    center_x = (frame.shape[1] - WIDTH) // 2
    center_y = (frame.shape[0] - HEIGHT) // 2

    result_image, result_cutout = get_result_and_mask(frame, center_x, center_y, WIDTH, HEIGHT)
    result_cutout = Image.fromarray(cv2.cvtColor(result_cutout, cv2.COLOR_BGR2RGB))

    x_output = stream(result_cutout)
    rendered_image = postprocess_image(x_output, output_type="pil")[0]  # .show()

    result_image[center_y : center_y + HEIGHT, center_x : center_x + WIDTH] = cv2.cvtColor(
        np.array(rendered_image), cv2.COLOR_RGB2BGR
    )

    # Display output
    cv2.imshow("output", result_image)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
