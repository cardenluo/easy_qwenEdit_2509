
import node_helpers
import comfy.utils
import math
import torch
import numpy as np
from PIL import Image
import json
import os
import copy
import folder_paths
import hashlib




class Easy_QwenEdit2509:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "auto_resize": (["crop", "pad", "stretch"], {"default": "crop"}), 
                "vl_size": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 64}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),

                "system_prompt": ("STRING", {"multiline": False, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "zero_negative", "latent",)
    FUNCTION = "QWENencode"
    CATEGORY = "conditioning"
    DESCRIPTION = """
    vl_size:视觉尺寸，会影响细节 
    latent_image: 生成图尺寸。Generate the size of the figure. 
    latent_mask: 生成图遮罩 
    system_prompt:系统提示词，指导图像特征描述与修改逻辑（默认提供基础配置） 
    auto_resize:尺寸适配模式（crop-中心裁剪/pad-黑色填充/stretch-强制拉伸）"""

    def _process_image_channels(self, image):
        if image is None:
            return None
        if len(image.shape) == 4:
            b, h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                black_bg = torch.zeros_like(rgb)
                image = rgb * alpha + black_bg * (1 - alpha)
                image = image[..., :3]
            elif c != 3:
                image = image[..., :3]
        elif len(image.shape) == 3:
            h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                black_bg = torch.zeros_like(rgb)
                image = rgb * alpha + black_bg * (1 - alpha)
                image = image[..., :3]
            elif c != 3:
                image = image[..., :3]
        image = image.clamp(0.0, 1.0)
        return image

    def _auto_resize(self, image: torch.Tensor, target_h: int, target_w: int, auto_resize: str) -> torch.Tensor:
        batch, ch, orig_h, orig_w = image.shape
        
        # 强制最小尺寸≥32（适配VAE 3×3卷积核）
        target_h = max(target_h, 32)
        target_w = max(target_w, 32)
        orig_h = max(orig_h, 32)
        orig_w = max(orig_w, 32)
        
        if auto_resize == "crop":
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            # 强制新尺寸≥目标尺寸，避免裁剪后不足
            new_w = max(new_w, target_w)
            new_h = max(new_h, target_h)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            x_offset = (new_w - target_w) // 2
            y_offset = (new_h - target_h) // 2
            # 裁剪后强制宽高≥32，避免过小
            crop_h = min(target_h, new_h - y_offset)
            crop_w = min(target_w, new_w - x_offset)
            crop_h = max(crop_h, 32)
            crop_w = max(crop_w, 32)
            result = scaled[:, :, y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
            
        elif auto_resize == "pad":
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            black_bg = torch.zeros((batch, ch, target_h, target_w), dtype=image.dtype, device=image.device)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            black_bg[:, :, y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled
            result = black_bg
            
        elif auto_resize == "stretch":
            result = comfy.utils.common_upscale(image, target_w, target_h, "bicubic", "disabled")
            
        else:
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            x_offset = (new_w - target_w) // 2
            y_offset = (new_h - target_h) // 2
            result = scaled[:, :, y_offset:y_offset + target_h, x_offset:x_offset + target_w]
        
        # 最终尺寸确保是8的倍数且≥32
        final_w = max(32, (result.shape[3] // 8) * 8)
        final_h = max(32, (result.shape[2] // 8) * 8)
        
        if final_w != result.shape[3] or final_h != result.shape[2]:
            x_offset = (result.shape[3] - final_w) // 2
            y_offset = (result.shape[2] - final_h) // 2
            result = result[:, :, y_offset:y_offset + final_h, x_offset:x_offset + final_w]
        
        return result

    def QWENencode(self, prompt="", image1=None, image2=None, image3=None, vae=None, clip=None, vl_size=384, 
                   latent_image=None, latent_mask=None, system_prompt="", auto_resize="crop"):
        
        if latent_image is None:
            raise ValueError("latent_image Must be input to determine the size of the generated image；latent_image 必须输入以确定生成图像的尺寸")
        
        image1 = self._process_image_channels(image1)
        image2 = self._process_image_channels(image2)
        image3 = self._process_image_channels(image3)
        orig_images = [image1, image2, image3]
        images_vl = []
        llama_template = self.get_system_prompt(system_prompt)
        image_prompt = ""

        for i, image in enumerate(orig_images):
            if image is not None:
                samples = image.movedim(-1, 1)
                current_total = samples.shape[3] * samples.shape[2]
                scale_by = math.sqrt(vl_size * vl_size / current_total) if current_total > 0 else 1.0
                width = max(64, round(samples.shape[3] * scale_by))
                height = max(64, round(samples.shape[2] * scale_by))
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"

        if latent_image is not None:
            latent_image = self._process_image_channels(latent_image)
            getsamples = latent_image.movedim(-1, 1)
            target_h, target_w = getsamples.shape[2], getsamples.shape[3]
            
            for i in range(3):
                if orig_images[i] is not None:
                    img_bchw = orig_images[i].movedim(-1, 1)
                    resized_img_bchw = self._auto_resize(img_bchw, target_h, target_w, auto_resize)
                    orig_images[i] = resized_img_bchw.movedim(1, -1)

        ref_latents = []
        for i, image in enumerate(orig_images):
            if image is not None and vae is not None:
                samples = image.movedim(-1, 1)
                # 强制尺寸≥32，避免VAE卷积报错
                orig_sample_h = max(samples.shape[2], 32)
                orig_sample_w = max(samples.shape[3], 32)
                if samples.shape[2] != orig_sample_h or samples.shape[3] != orig_sample_w:
                    samples = comfy.utils.common_upscale(samples, orig_sample_w, orig_sample_h, "bicubic", "disabled")
                # 计算8的倍数尺寸，仍强制≥32
                width = (orig_sample_w // 8) * 8
                height = (orig_sample_h // 8) * 8
                width = max(width, 32)
                height = max(height, 32)
                scaled_img = comfy.utils.common_upscale(samples, width, height, "bicubic", "disabled")
                ref_latents.append(vae.encode(scaled_img.movedim(1, -1)[:, :, :, :3]))

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        positive = conditioning
        negative = self.zero_out(positive)

        latent = {"samples": torch.zeros(1, 4, 64, 64)}
        if latent_image is not None:
            positive, negative, latent = self.addConditioning(positive, negative, latent_image, vae, mask=latent_mask if latent_mask is not None else None)

        return (positive, negative, latent)

    def addConditioning(self, positive, negative, pixels, vae, mask=None):
        pixels = self._process_image_channels(pixels)
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        orig_pixels = pixels
        pixels = orig_pixels.clone()

        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] = pixels[:, :, :, i] * m + 0.5 * (1 - m)
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": vae.encode(orig_pixels), "noise_mask": mask}
        else:
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": concat_latent}

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent})
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        return (out[0], out[1], out_latent)

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return c

    def get_system_prompt(self, instruction):
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        if instruction == "":
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        else:
            if template_prefix in instruction:
                instruction = instruction.split(template_prefix)[1]
            if template_suffix in instruction:
                instruction = instruction.split(template_suffix)[0]
            if "{}" in instruction:
                instruction = instruction.replace("{}", "")
            instruction_content = instruction
        return template_prefix + instruction_content + template_suffix

NODE_CLASS_MAPPINGS = {
    "Easy_QwenEdit2509": Easy_QwenEdit2509,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Easy_QwenEdit2509": "Easy_QwenEdit2509",
}






