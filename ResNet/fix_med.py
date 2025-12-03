import onnx
from onnx import helper

# 加载要转换的模型（确保路径正确）
model_path = './resnet50.onnx'
model = onnx.load(model_path)
print(f'正在检查模型：{model_path}')
print(f'模型中所有算子数量：{len(model.graph.node)}')
print('查找AveragePool算子：')
avgpool_count = 0
fixed_count = 0

for node in model.graph.node:
    if node.op_type == 'AveragePool':
        avgpool_count += 1
        print(f'第{avgpool_count}个AveragePool算子：{node.name}')
        # 打印当前所有属性
        print('当前属性：')
        for attr in node.attribute:
            attr_val = list(attr.ints) if attr.name in ['kernel_shape', 'strides', 'pads'] else attr
            print(f'{attr.name}: {attr_val}')
        
        # 1. 修复kernel_shape（ResNet50最终平均池化需为7x7）
        has_kernel = any(attr.name == 'kernel_shape' for attr in node.attribute)
        if not has_kernel:
            node.attribute.append(helper.make_attribute('kernel_shape', [7, 7]))
            fixed_count += 1
            print(f'已补全kernel_shape=[7,7]')
        else:
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    current_kernel = list(attr.ints)
                    if current_kernel != [7, 7] or len(current_kernel) != 2:
                        attr.ints[:] = [7, 7]  # 强制设置为7x7
                        fixed_count += 1
                        print(f'kernel_shape错误（当前{current_kernel}），已修复为[7,7]')
                else:
                    print(f'已包含正确kernel_shape=[7,7]')
        
        # 2. 修复strides（步长需为1x1）
        has_strides = any(attr.name == 'strides' for attr in node.attribute)
        if not has_strides:
            node.attribute.append(helper.make_attribute('strides', [1, 1]))
            fixed_count += 1
            print(f'已补全strides=[1,1]')
        else:
            for attr in node.attribute:
                if attr.name == 'strides':
                    current_strides = list(attr.ints)
                    if current_strides != [1, 1] or len(current_strides) != 2:
                        attr.ints[:] = [1, 1]
                        fixed_count += 1
                        print(f'strides错误（当前{current_strides}），已修复为[1,1]')

        # 3. 修复pads（无填充，需为0x0x0x0）
        has_pads = any(attr.name == 'pads' for attr in node.attribute)
        if not has_pads:
            node.attribute.append(helper.make_attribute('pads', [0, 0, 0, 0]))
            fixed_count += 1
            print(f'已补全pads=[0,0,0,0]')
        else:
            for attr in node.attribute:
                if attr.name == 'pads':
                    current_pads = list(attr.ints)
                    if current_pads != [0, 0, 0, 0] or len(current_pads) != 4:
                        attr.ints[:] = [0, 0, 0, 0]
                        fixed_count += 1
                        print(f'pads错误（当前{current_pads}），已修复为[0,0,0,0]')

# 保存补全后的模型
output_path = './resnet50_fixed.onnx'
onnx.save(model, output_path)
print(f'处理完成！共检查{avgpool_count}个AveragePool算子，修复{fixed_count}处问题')
print(f'修复后模型路径：{output_path}')