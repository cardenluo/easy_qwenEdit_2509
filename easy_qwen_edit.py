import torch
import math
import comfy.utils
import node_helpers
import comfy
from comfy.utils import common_upscale




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


    def QWENencode(self,prompt="",image1=None, image2=None, image3=None,vae=None,clip=None,
                vl_size=384,latent_image=None, latent_mask=None, ):

        auto_resize=True
        if auto_resize and latent_image is not None:
            getsamples = latent_image.movedim(-1, 1)
            if image1 is not None: 
                image1 = image1.movedim(-1, 1)
                image1 = self.auto_resize(image1, getsamples)[0]
                image1 = image1.movedim(1, -1)  
            if image2 is not None:
                image2 = image2.movedim(-1, 1)
                image2 = self.auto_resize(image2, getsamples)[0]
                image2 = image2.movedim(1, -1)
            if image3 is not None:
                image3 = image3.movedim(-1, 1)
                image3 = self.auto_resize(image3, getsamples)[0]
                image3 = image3.movedim(1, -1)

#-----------------------------------------------------
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(vl_size * vl_size)   #视觉统一处理
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))

                if vae is not None:
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

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        positive = conditioning
        negative=self.zero_out(positive)
#------------------------------------------------------------------------
        if latent_image is not None:
            positive, negative, latent = self.addConditioning(
                positive, negative, latent_image, vae, 
                mask=latent_mask if latent_mask is not None else None)

        return (positive,negative, latent, )


    def addConditioning(self,positive, negative, pixels, vae, mask=None):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        
        orig_pixels = pixels
        pixels = orig_pixels.clone()
        
        # 如果提供了 mask，则进行相关处理
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")
            
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
                mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:,:,:,i] -= 0.5
                pixels[:,:,:,i] *= m
                pixels[:,:,:,i] += 0.5
                
            concat_latent = vae.encode(pixels)
            
            out_latent = {}
            out_latent["samples"] = vae.encode(orig_pixels)
            out_latent["noise_mask"] = mask
        else:
            # 如果没有提供 mask，直接编码原始像素
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": concat_latent}

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent})
            # 只有当 mask 存在时才添加 concat_mask
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        
        return (out[0], out[1], out_latent)


    def auto_resize(self, image, get_image_size):
        if len(image.shape) == 3:
            H, W, C = image.shape
        else:  
            B, H, W, C = image.shape

        _, height_max, width_max, _ = get_image_size.shape
            
        image = image.movedim(-1,1)
        outimage = common_upscale(image, width_max, height_max, "bicubic", "center")
        image = outimage.movedim(1,-1)
        
        width = max(image.shape[2], 64)
        height = max(image.shape[1], 64)
        
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        return(image,)


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































