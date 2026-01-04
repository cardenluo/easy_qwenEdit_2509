
## 插件介绍

官方做了1024*1024百万像素的限制，导致输入的图片小了就被放大，大了就被缩小。

1、本节点取消了这个限制，无论你用什么尺寸都可以直接出图。当然由于采样器输出必须是8的倍数，如果你的尺寸不是8的倍数，输出也还是会被自动调整成8的倍数，所以建议先提前处理好。

2、多图参考时，各参考图像尺寸不一致，也会导致参考比例失衡，造成偏移。 所以，节点内置了算法，会自动处理成与latent_image相同的尺寸，可以最大程度减少偏移问题。如果尺寸前处理规范，可以很容易实现零偏移, 

https://github.com/cardenluo/ComfyUI-Apt_Preset 插件的"总控_QwenEditplus堆"在B站做很多无偏移的案例， 此为相同原理的非管线版本


<img width="2457" height="1296" alt="11" src="https://github.com/user-attachments/assets/1a3e9c2c-160f-476c-a3db-867841cb7927" />



演示：附件有工作流

<img width="1320" height="697" alt="image" src="https://github.com/user-attachments/assets/7c86a5ba-7470-4bc9-85c3-b826dfc46eb2" />


一、当参考图和生成图尺寸一致，不用拉伸，也不用填充，信息就能全部参考，效果是最佳的。

二、当参考图和生成图尺寸不一致时，尺寸统一有3种选择方式，按照生成需要进行选择：

auto resize 缩放模式（crop=中心裁剪，pad=中心黑色填充，stretch=强制拉伸）

<img width="1911" height="1145" alt="image" src="https://github.com/user-attachments/assets/ec5e4280-0244-493f-8229-f345f47ca03c" />

<img width="2914" height="1160" alt="image" src="https://github.com/user-attachments/assets/ca04ee3f-5b0f-4ea7-9518-2bc9425dbc71" />


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
