# easy_Qwen_Image_Edit

## 插件介绍

这个节点修改了官方的尺寸限制，同时做了尺寸一致性处理，可以最大程度减少偏移问题。如果尺寸前处理规范，比如8的倍数，可以很容易实现零偏移。

是https://github.com/cardenluo/ComfyUI-Apt_Preset 插件的"总控_QwenEditplus堆"同款，非管线版本，同样带了遮罩功能

<img width="599" height="648" alt="image" src="https://github.com/user-attachments/assets/15bb70f9-c4d2-4f44-b8d5-ff224ae6b3e8" />


演示：附件有工作流
<img width="1394" height="720" alt="image" src="https://github.com/user-attachments/assets/2aa7fdb7-c596-46c0-8d3c-8747cd74cccc" />



## 安装方法

1. 下载本插件的ZIP文件
2. 解压到ComfyUI的`custom_nodes`目录下
3. 重启ComfyUI

## 节点说明

### easy_qwenEdit_2509

该节点位于ComfyUI的   CATEGORY = "conditioning"类别下，提供以下功能：

- 支持多参考图像输入
- 通过文本提示编辑图像
- 自动调整图像尺寸
- 支持遮罩编辑
- 生成正条件、零负条件和潜变量

### 输入参数

**必填参数：**
- `clip`: CLIP模型
- `vae`: VAE模型

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

## 使用方法

1. 在ComfyUI工作流中添加"Easy Qwen Image Edit (2509)"节点
2. 连接CLIP和VAE模型
3. 提供参考图像（最多3张）
4. 设置视觉尺寸参数
5. 输入文本提示
6. 提供尺寸基准图（latent_image）
7. 可选：添加遮罩
8. 连接输出到后续节点

## 注意事项

- `latent_image`是必填参数，用于确定生成图像的尺寸
- 参考图像会自动调整大小以匹配基准图的尺寸
- 遮罩可以用于指定图像中需要编辑的区域
- vl_size参数控制视觉处理的分辨率，影响细节表现

## 示例工作流

1. 加载参考图像
2. 设置文本提示描述所需的编辑
3. 提供尺寸基准图
4. 连接到采样器节点生成最终图像

## 依赖

- ComfyUI
- PyTorch
- comfy.utils
- node_helpers

## 版本历史

- 2509: 初始版本
