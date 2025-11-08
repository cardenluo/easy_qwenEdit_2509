<img width="1320" height="697" alt="image" src="https://github.com/user-attachments/assets/78ef309e-8a28-4859-86fb-2d226cceb941" />

## 插件介绍

修改了官方的尺寸限制，同时做了尺寸一致性处理，可以最大程度减少偏移问题。如果尺寸前处理规范，可以很容易实现零偏移, 

https://github.com/cardenluo/ComfyUI-Apt_Preset 插件的"总控_QwenEditplus堆"在B站做很多无偏移的案例， 此为相同原理的非管线版本


<img width="2457" height="1296" alt="image" src="https://github.com/user-attachments/assets/dec6cd9e-6814-43db-bdbf-54175c81ee03" />

演示：附件有工作流

<img width="1320" height="697" alt="image" src="https://github.com/user-attachments/assets/7c86a5ba-7470-4bc9-85c3-b826dfc46eb2" />


更新：3种可选自动统一尺寸的方式

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
