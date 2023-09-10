from __future__ import annotations
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline

import torch

# BASE MODEL ID FOR PHOTO RELASTIC: dreamlike-art/dreamlike-photoreal-2.0
# BASE MODEL ID FOR MORE ARTLIKE: runwayml/stable-diffusion-v1-5
# BASE MODEL ID FOR MORE HIGH CONTRAST FANTASY: 22h/vintedois-diffusion-v0-1
# BASE MODEL ID FOR MIDJOURNEY REALISM: prompthero/openjourney

class Model:
    def __init__(self,
                 base_model_id: str = 'runwayml/stable-diffusion-v1-5',
                 task_name: str = 'img2img'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_id = ''
        self.task_name = ''
        self.pipe = self.load_pipe(base_model_id, task_name)
    
    def load_pipe(self, base_model_id, task_name):
        if base_model_id == self.base_model_id and task_name == self.task_name and hasattr(
                self, 'pipe'):
            return self.pipe
        
        if self.task_name == 'img2img':
            return self.load_img2img_pipe(self.base_model_id)
        else:
            return self.load_sd_pipe(self.base_model_id)

    def load_img2img_pipe(self, base_model_id: str) -> StableDiffusionImg2ImgPipeline:
        pipe = None
        if torch.cuda.is_available():
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(base_model_id, 
                                          safety_checker=None, 
                                          torch_dtype=torch.float16
                                        )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(base_model_id)
        
        pipe.to(self.device)
        return pipe
    
    def load_sd_pipe(self, base_model_id) -> DiffusionPipeline:
        pipe = None
        if torch.cuda.is_available():
            pipe = DiffusionPipeline.from_pretrained(base_model_id, 
                                          safety_checker=None, 
                                          torch_dtype=torch.float16
                                        )
        else:
            pipe = DiffusionPipeline.from_pretrained(base_model_id)
        
        pipe.to(self.device)
        return pipe
    
    def resize(value,img):
        img = Image.open(img)
        img = img.resize((value,value))
        return img
    
    def run_img2img_pipe(
            self,
            source_img,
            prompt: str, 
            negative_prompt: str, 
            strength: float,
            guide: int, 
            steps: int, 
            seed: int, 
            num_images=1
    ) -> StableDiffusionImg2ImgPipeline:
        generator = torch.Generator(self.device).manual_seed(seed)
        source_image = self.resize(768, source_img)
        source_image.save('source.png')
        image = self.pipe(prompt, negative_prompt=negative_prompt, image=source_image, strength=strength, guidance_scale=guide, num_inference_steps=steps).images[0]
        return image

    def run_sd_pipe(
            self,
            prompt,
            negative_prompt,
            guide,
            steps,
            seed,
            num_images=1
    ) -> DiffusionPipeline:
        generator = torch.Generator(device).manual_seed(seed)
        image = sd_pipe(prompt, negative_prompt=negative_prompt, guidance_scale=guide, num_inference_steps=steps, num_images_per_prompt=num_images, height=768).images[0]
        return image