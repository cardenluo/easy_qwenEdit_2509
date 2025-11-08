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
                "system_prompt": ("STRING", {"multiline": False, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "zero_negative", "latent",)
    FUNCTION = "QWENencode"
    CATEGORY = "conditioning"
    DESCRIPTION = """vl_size:视觉尺寸，会影响细节 latent_image: 生成图尺寸。Generate the size of the figure. latent_mask: 生成图遮罩 system_prompt:系统提示词，指导图像特征描述与修改逻辑（默认提供基础配置）"""

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

    def QWENencode(self, prompt="", image1=None, image2=None, image3=None, vae=None, clip=None, vl_size=384, latent_image=None, latent_mask=None, system_prompt=""):
        auto_resize = True

        if latent_image is None:
            raise ValueError("latent_image Must be input to determine the size of the generated image；latent_image 必须输入以确定生成图像的尺寸")
        

        image1 = self._process_image_channels(image1)
        image2 = self._process_image_channels(image2)
        image3 = self._process_image_channels(image3)
        orig_images = [image1, image2, image3]
        images_vl = []
        llama_template = self.get_system_prompt(system_prompt)
        image_prompt = ""

        # 第一步：用原始图处理vl_size，生成CLIP编码用的images_vl（核心优化）
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

        # 第二步：对原始图做尺寸适配（仅用于VAE编码ref_latents，不影响vl_size特征）
        if auto_resize and latent_image is not None:
            latent_image = self._process_image_channels(latent_image)
            getsamples = latent_image.movedim(-1, 1)
            for i in range(3):
                if orig_images[i] is not None:
                    img = orig_images[i].movedim(-1, 1)
                    resized_img = self.auto_resize(img, getsamples)
                    orig_images[i] = resized_img.movedim(1, -1)

        # 第三步：生成ref_latents（用适配后的图，不影响CLIP特征提取）
        ref_latents = []
        for i, image in enumerate(orig_images):
            if image is not None and vae is not None:
                samples = image.movedim(-1, 1)
                current_total = samples.shape[3] * samples.shape[2]
                if latent_image is not None:
                    getsamples = latent_image.movedim(-1, 1)
                    getwidth, getheight = getsamples.shape[3], getsamples.shape[2]
                    total = getwidth * getheight
                    scale_by = math.sqrt(total / current_total) if current_total > 0 else 1.0
                else:
                    total = 1024 * 1024
                    scale_by = math.sqrt(total / current_total) if current_total > 0 else 1.0
                width = max(64, round(samples.shape[3] * scale_by / 8.0) * 8)
                height = max(64, round(samples.shape[2] * scale_by / 8.0) * 8)
                scaled_img = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                ref_latents.append(vae.encode(scaled_img.movedim(1, -1)[:, :, :, :3]))

        # 文本编码与条件生成
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        positive = conditioning
        negative = self.zero_out(positive)

        # 处理latent图像和遮罩
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

    def auto_resize(self, image, get_image_size):
        """中心裁切实现：将输入图像裁切为目标尺寸（get_image_size的尺寸），保持中心区域"""
        # 确保输入为4维张量（batch, channel, height, width）
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        _, _, img_h, img_w = image.shape
        _, _, target_h, target_w = get_image_size.shape

        # 计算裁切偏移量：确保从中心开始裁切
        x_offset = max(0, (img_w - target_w) // 2)
        y_offset = max(0, (img_h - target_h) // 2)

        # 执行中心裁切
        cropped_img = image[:, :, y_offset:y_offset+target_h, x_offset:x_offset+target_w]

        # 确保输出尺寸为8的整数倍（适配VAE编码要求）
        final_w = max(64, (cropped_img.shape[3] // 8) * 8)
        final_h = max(64, (cropped_img.shape[2] // 8) * 8)

        # 若裁切后尺寸仍不满足8的倍数，微调至最近的8倍整数（仅缩小，不拉伸）
        if final_w != cropped_img.shape[3] or final_h != cropped_img.shape[2]:
            x微调_offset = (cropped_img.shape[3] - final_w) // 2
            y微调_offset = (cropped_img.shape[2] - final_h) // 2
            cropped_img = cropped_img[:, :, y微调_offset:y微调_offset+final_h, x微调_offset:x微调_offset+final_w]

        return cropped_img

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















