

## 插件介绍

修改了官方的尺寸限制，同时做了尺寸一致性处理，可以最大程度减少偏移问题。如果尺寸前处理规范，可以很容易实现零偏移。

是https://github.com/cardenluo/ComfyUI-Apt_Preset 插件的"总控_QwenEditplus堆"同款，非管线版本，同样带了遮罩功能


<img width="2457" height="1296" alt="image" src="https://github.com/user-attachments/assets/dec6cd9e-6814-43db-bdbf-54175c81ee03" />

演示：附件有工作流
<img width="1394" height="720" alt="image" src="https://github.com/user-attachments/assets/2aa7fdb7-c596-46c0-8d3c-8747cd74cccc" />




**可选参数：**
- `image1`: 第一张参考图像
- `image2`: 第二张参考图像
- `image3`: 第三张参考图像
- `vl_size`: 视觉尺寸，会影响细节（默认：384，范围：64-2048，步长：64）
- `prompt`: 文本提示（多行支持）
- `latent_image`: 生成图尺寸基准图（必填）
- `latent_mask`: 生成图遮罩（可选）

### 输出参数

- `positive`: 正条件
- `zero_negative`: 零负条件
- `latent`: 潜变量


## 版本历史

- 2509: 初始版本
