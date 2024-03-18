import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler


class Pipeline:
    def __init__(self, device="cuda"):
        print("Initializing depth ControlNet...")
        
        depth_controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            use_safetensors=True,
            torch_dtype=torch.float16
        ).to(device)

        print("Initializing autoencoder...")

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        ).to(device)

        print("Initializing SDXL pipeline...")

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=[depth_controlnet],
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16
            # low_cpu_mem_usage=True
        ).to("cuda")

        self.pipe.enable_model_cpu_offload()
        # speed up diffusion process with faster scheduler and memory optimization
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        # remove following line if xformers is not installed
        self.pipe.enable_xformers_memory_efficient_attention()


    def run_pipeline(self, image, positive_prompt, negative_prompt, seed):
        if seed == -1:
            print("Using random seed")
            generator = None
        else:
            print("Using seed:", seed)
            generator = torch.manual_seed(seed)

        images = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            num_images_per_prompt=4,
            controlnet_conditioning_scale=0.65,
            guidance_scale=10.0,
            generator=generator,
            image=image
        ).images

        return images