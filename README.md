# Plane Image Generate

本项目旨在生成 被机库遮挡的飞机图像，并基于 DOTA 数据集实现结合 [Segment Anything (SAM)] 自动剪切与自动部件标注的功能。

## 机库遮挡的飞机图像生成

项目通过手工裁切的飞机贴片、机库遮挡图、影子图层等素材，自动完成图层叠加，模拟真实场景中的目标遮挡。主要功能包括：

✂️ 飞机与机库图层自动融合

🏗️ 机场背景自动叠加与机库位置智能摆放

🔁 飞机朝向 & 遮挡程度随机生成

🧾 生成符合 MMDetection 标准的训练数据（图像 & 标注）

用于生成数据集的元素（机库、飞机贴片，以及机场背景）保存在[场景要素](https://github.com/lygitgit/Plane_image_geneate/tree/main/processed_data)中，依照background_1的格式样例可以添加更多的场景。

运行[main_generate_shelter_plane.py](https://github.com/lygitgit/Plane_image_geneate/blob/main/main_generate_shelter_plane.py)，可得到如下训练训练集图像🖼️ ：

<img src="https://github.com/user-attachments/assets/3fd377fe-a772-4bc7-b39f-826d028aa445" width="400">
<img src="https://github.com/user-attachments/assets/26aabcf1-9e14-48c5-b4a3-705891d4b42e" width="400">

## 飞机部件自动标注生成

采用SAM（segment anything模型）以及原始Dota数据集飞机的旋转框标注进行飞机掩码的提取，根据掩码的对称性（水平轴/垂直轴设置在不同位置求上下/左右翻折后重叠的面积占比），得到部件的位置，进行标注

🤖 SAM 自动剪切与部件标注示意如下：

<img src="https://github.com/user-attachments/assets/f311afa3-8e3b-4ddc-bc67-f00d142280c9" width="400">
<img src="https://github.com/user-attachments/assets/d0c9545c-02d9-4d85-8ce7-4948f5f4a3d5" width="400">

修改[main_generate_shelter_plane.py](https://github.com/lygitgit/Plane_image_geneate/blob/main/main_generate_shelter_plane.py)中本地的Dota数据集路径，运行后可得到如下自动化部件标注示例🔍：(https://github.com/lygitgit/Plane_image_geneate/blob/main/main_generate_shelter_plane.py)，可得到如下训练训练集图像🖼️ ：

<img src="https://github.com/user-attachments/assets/e35ab821-c544-4a8b-91cc-6b582e0e02d8" width="600">
