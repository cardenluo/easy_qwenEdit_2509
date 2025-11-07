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
                "vl_size": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 64}),                   
                "prompt": ("STRING", {"multiline": True, "default": ""}),    
                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),
            }
        }
        
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "zero_negative", "latent",)
    FUNCTION = "QWENencode"
    CATEGORY = "conditioning"
    DESCRIPTION = """注释：
    vl_size:视觉尺寸，会影响细节    
    latent_image: 生成图尺寸。Generate the size of the figure.
    latent_mask: 生成图遮罩
    """


    def QWENencode(self, prompt="", image1=None, image2=None, image3=None, vae=None, clip=None,
                vl_size=384, latent_image=None, latent_mask=None):
        auto_resize = True
        # 自动调整输入图像尺寸以匹配latent_image
        if auto_resize and latent_image is not None:
            getsamples = latent_image.movedim(-1, 1)
            for i in range(3):
                img = [image1, image2, image3][i]
                if img is not None:
                    img = img.movedim(-1, 1)
                    resized_img = self.auto_resize(img, getsamples)
                    if i == 0:
                        image1 = resized_img.movedim(1, -1)
                    elif i == 1:
                        image2 = resized_img.movedim(1, -1)
                    else:
                        image3 = resized_img.movedim(1, -1)

        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        # 使用通用函数获取系统提示
        llama_template = self.get_system_prompt("")
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                # 处理视觉尺寸
                total = int(vl_size * vl_size)
                current_total = samples.shape[3] * samples.shape[2]
                scale_by = math.sqrt(total / current_total) if current_total > 0 else 1.0
                width = max(64, round(samples.shape[3] * scale_by))
                height = max(64, round(samples.shape[2] * scale_by))
                # 使用comfy的通用缩放函数
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))

                if vae is not None:
                    # 处理参考 latent
                    if latent_image is not None:
                        getsamples = latent_image.movedim(-1, 1)
                        getwidth, getheight = getsamples.shape[3], getsamples.shape[2]
                        total = getwidth * getheight
                        scale_by = math.sqrt(total / current_total) if current_total > 0 else 1.0
                        width = max(64, round(samples.shape[3] * scale_by / 8.0) * 8)
                        height = max(64, round(samples.shape[2] * scale_by / 8.0) * 8)
                    else:
                        total = int(1024 * 1024)
                        scale_by = math.sqrt(total / current_total) if current_total > 0 else 1.0
                        width = max(64, round(samples.shape[3] * scale_by / 8.0) * 8)
                        height = max(64, round(samples.shape[2] * scale_by / 8.0) * 8)

                    # 缩放图像用于VAE编码
                    scaled_img = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(scaled_img.movedim(1, -1)[:, :, :, :3]))

                image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"

        # 处理文本编码
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        positive = conditioning
        negative = self.zero_out(positive)

        # 处理latent图像和遮罩
        latent = {"samples": torch.zeros(1, 4, 64, 64)}  # 默认latent
        if latent_image is not None:
            positive, negative, latent = self.addConditioning(
                positive, negative, latent_image, vae, 
                mask=latent_mask if latent_mask is not None else None)

        return (positive, negative, latent)


    def addConditioning(self, positive, negative, pixels, vae, mask=None):
        # 确保尺寸是8的倍数
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        
        orig_pixels = pixels
        pixels = orig_pixels.clone()
        
        # 处理遮罩
        if mask is not None:
            # 调整遮罩尺寸以匹配像素
            mask = torch.nn.functional.interpolate(
                mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                size=(pixels.shape[1], pixels.shape[2]),
                mode="bilinear"
            )
            
            # 裁剪到8的倍数
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

            # 应用遮罩
            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] = pixels[:, :, :, i] * m + 0.5 * (1 - m)
                
            concat_latent = vae.encode(pixels)
            out_latent = {
                "samples": vae.encode(orig_pixels),
                "noise_mask": mask
            }
        else:
            # 无遮罩情况
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": concat_latent}

        # 更新条件
        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent})
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        
        return (out[0], out[1], out_latent)


    def auto_resize(self, image, get_image_size):
        # 确保输入图像有批次维度
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 获取目标尺寸
        _, _, height_max, width_max = get_image_size.shape
            
        # 缩放图像
        outimage = comfy.utils.common_upscale(image, width_max, height_max, "bicubic", "center")
        
        # 确保尺寸是8的倍数
        width = max(64, (outimage.shape[3] // 8) * 8)
        height = max(64, (outimage.shape[2] // 8) * 8)
        
        # 最终缩放
        return comfy.utils.common_upscale(outimage, width, height, "bicubic", "center")


    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            # 清零所有可能的条件输出
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
        # 复用参考节点中的系统提示生成逻辑
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
