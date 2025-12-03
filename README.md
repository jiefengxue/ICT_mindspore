对昇思大模型平台上的ResNet模型进行改进，并提供给从checkpoint格式到OM格式的转换方法路径

参考为：https://xihe.mindspore.cn/training-projects/wujiacong/skin

1.准备

在昇思大模型平台上https://xihe.mindspore.cn/training-projects

选择1*ascend-snt9b|ARM: 19核 180GB  python3.9-ms2.6.0-cann8.1.RC1.beta1

上传三个文件，都放在同级目录下

另需上传数据集文件，在昇思大模型平台上可以下载 https://xihe.mindspore.cn/datasets/knoka/pifudata_Maker/tree


2.步骤

解压
unzip pifudata_Maker.zip

训练 得到ONNX
python train_med.py

创建虚拟环境
chmod +x setup_onnx_env.sh

./setup_onnx_env.sh

激活
source /home/mindspore/work/onnx_fix_env/bin/activate

执行修复
python fix_med.py 

从ONNX到OM
atc --model=./resnet50_fixed.onnx --framework=5 --soc_version=Ascend310B1 --output=res50_final


3.参考
https://xihe.mindspore.cn/training-projects/wujiacong/skin

