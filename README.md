对昇思大模型平台上的ResNet模型进行改进，并提供给从checkpoint格式到OM格式的转换方法路径
https://cloud-42a1a443-a84e-4296-9893-6a23322f6775.snt9b.xihe.mindspore.cn/lab/tree/%E5%BA%94%E7%94%A8%E5%AE%9E%E8%B7%B5/%E7%83%AD%E9%97%A8LLM%E5%8F%8A%E5%85%B6%E4%BB%96AI%E5%BA%94%E7%94%A8/%E5%9F%BA%E4%BA%8EMindSpore%E7%9A%84%E6%81%B6%E6%80%A7%E7%9A%AE%E8%82%A4%E8%82%BF%E7%98%A4%E8%AF%86%E5%88%AB.ipynb
1.准备
在昇思大模型平台上https://xihe.mindspore.cn/training-projects
选择1*ascend-snt9b|ARM: 19核 180GB  python3.9-ms2.6.0-cann8.1.RC1.beta1
上传三个文件，都放在同级目录下
另需上传数据集文件，在昇思大模型平台上可以下载 https://xihe.mindspore.cn/datasets/knoka/pifudata_Maker/tree

2.步骤
unzip pifudata_Maker.zip
python train_med.py
chmod +x setup_onnx_env.sh
./setup_onnx_env.sh
source /home/mindspore/work/onnx_fix_env/bin/activate
python fix_med.py 
atc --model=./resnet50_fixed.onnx --framework=5 --soc_version=Ascend310B1 --output=res50_final
