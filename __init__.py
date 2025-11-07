import torch
import math
import comfy.utils
import node_helpers
import comfy
from comfy.utils import common_upscale
import types


# 修复CLIP.tokenize扩展参数依赖
def patched_clip_tokenize(original_tokenize, self, text, images=None, llama_template=None):
    """为CLIP.tokenize添加images和llama_template参数支持"""
    if llama_template:
        text = llama_template.format(text)
    # 处理图像（保持与原逻辑兼容的空实现，可根据实际需求补充）
    if images is not None:
        pass  # 此处可添加原插件中的图像编码逻辑
    return original_tokenize(self, text)

# 应用CLIP猴子补丁
if not hasattr(comfy.sd.CLIP, 'original_tokenize'):
    comfy.sd.CLIP.original_tokenize = comfy.sd.CLIP.tokenize
    def new_tokenize(self, text, images=None, llama_template=None):
        return patched_clip_tokenize(self.original_tokenize, self, text, images, llama_template)
    comfy.sd.CLIP.tokenize = types.MethodType(new_tokenize, comfy.sd.CLIP)


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
                "vl_size": ("INT", {"default":384, "min": 64, "max": 2048, "step": 64}),                   
                "prompt": ("STRING", {"multiline": True, "default": ""}),    
                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),
            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("positive","zero_negative","latent",)
    FUNCTION = "QWENencode"
    CATEGORY = "conditioning"
    DESCRIPTION = """注释：
    vl_size:视觉尺寸，会影响细节    
    latent_image: 生成图尺寸。Generate the size of the figure.
    latent_mask: 生成图遮罩
    """


    def QWENencode(self, prompt="", image1=None, image2=None, image3=None, vae=None, clip=None,
                vl_size=384, latent_image=None, latent_mask=None, ):
        # 模型有效性检查
        if not hasattr(clip, 'tokenize') or not callable(clip.tokenize):
            raise ValueError("CLIP模型未正确初始化，缺少tokenize方法")
        if not hasattr(vae, 'encode') or not callable(vae.encode):
            raise ValueError("VAE模型未正确初始化，缺少encode方法")

        auto_resize = True
        if auto_resize and latent_image is not None:
            getsamples = latent_image.movedim(-1, 1)
            # 处理图像1
            if image1 is not None: 
                image1 = image1.movedim(-1, 1)
                image1 = self.auto_resize(image1, getsamples)[0]
                image1 = image1.movedim(1, -1)  
            # 处理图像2
            if image2 is not None:
                image2 = image2.movedim(-1, 1)
                image2 = self.auto_resize(image2, getsamples)[0]
                image2 = image2.movedim(1, -1)
            # 处理图像3
            if image3 is not None:
                image3 = image3.movedim(-1, 1)
                image3 = self.auto_resize(image3, getsamples)[0]
                image3 = image3.movedim(1, -1)

        # 初始化变量
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        # 处理输入图像
        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(vl_size * vl_size)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))

                # 生成参考latent
                if latent_image is not None:
                    getsamples = latent_image.movedim(-1, 1)
                    getwidth = round(getsamples.shape[3])
                    getheight = round(getsamples.shape[2])
                    total = int(getwidth * getheight)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8
                else:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                K = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                ref_latents.append(vae.encode(K.movedim(1, -1)[:, :, :, :3]))
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # 生成条件
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        positive = conditioning
        negative = self.zero_out(positive)

        # 处理latent图像
        # 初始化默认latent防止未定义
        latent = {"samples": torch.zeros((1, 4, 64, 64), device=vae.device)}
        if latent_image is not None:
            positive, negative, latent = self.addConditioning(
                positive, negative, latent_image, vae, 
                mask=latent_mask if latent_mask is not None else None)

        return (positive, negative, latent, )


    def addConditioning(self, positive, negative, pixels, vae, mask=None):
        # 计算有效尺寸
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        
        orig_pixels = pixels
        pixels = orig_pixels.clone()
        
        # 处理遮罩
        if mask is not None:
            mask = torch.nn.functional.interpolate(
                mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
                size=(pixels.shape[1], pixels.shape[2]), 
                mode="bilinear"
            )
            
            # 尺寸调整
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

            # 应用遮罩
            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] -= 0.5
                pixels[:, :, :, i] *= m
                pixels[:, :, :, i] += 0.5
                
            concat_latent = vae.encode(pixels)
            out_latent = {
                "samples": vae.encode(orig_pixels),
                "noise_mask": mask
            }
        else:
            # 无遮罩情况
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": concat_latent}

        # 处理条件
        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent})
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        
        return (out[0], out[1], out_latent)


    def auto_resize(self, image, get_image_size):
        # 确保输入是4维张量（B, H, W, C）
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # 增加batch维度
        
        B, H, W, C = image.shape
        _, height_max, width_max, _ = get_image_size.shape
            
        # 缩放图像
        image = image.movedim(-1, 1)
        outimage = common_upscale(image, width_max, height_max, "bicubic", "center")
        image = outimage.movedim(1, -1)
        
        # 确保尺寸是8的倍数
        width = max(image.shape[2], 64)
        height = max(image.shape[1], 64)
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # 二次调整到目标尺寸
        if image.shape[2] != width or image.shape[1] != height:
            image = image.movedim(-1, 1)
            image = common_upscale(image, width, height, "bicubic", "center")
            image = image.movedim(1, -1)
        
        return (image,)


    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            # 清空pooled_output
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            # 清空conditioning_lyrics
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return c


NODE_CLASS_MAPPINGS = {
    "Easy_QwenEdit2509": Easy_QwenEdit2509,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Easy_QwenEdit2509": "Easy QwenEdit 2509",
}