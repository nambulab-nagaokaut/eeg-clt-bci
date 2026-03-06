import torch
import torch.nn as nn
import torch.nn.functional as F

# Hàm lấy parameters của từng layer riêng biệt để áp dụng weight decay
def get_parameters_by_layer_type(module, layer_type, parent_name=''):
    params = []
    for name, child in module.named_children():
        # Tạo tên đầy đủ cho layer hiện tại
        full_name = f'{parent_name}.{name}' if parent_name else name
        
        if isinstance(child, layer_type):
            # print(f'Kernel regularizer applied Layer: {full_name}')
            params.extend(child.parameters())
        else:
            # Đệ quy vào các con của module hiện tại
            params.extend(get_parameters_by_layer_type(child, layer_type, full_name))
    
    return params

# Định nghĩa hàm apply_max_norm
def apply_max_norm(model, max_norm, modules_to_apply, layers=(nn.Conv1d, nn.Conv2d)):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, layers):
                # Kiểm tra xem module có nằm trong modules_to_apply hoặc là con của chúng không
                if is_child_of_modules(module, modules_to_apply):
                    # In ra tên của layer Conv được truy cập
                    # print(f"Kernel Constraint applied layer: {name}")
                    # Xử lý trường hợp CausalConv1D
                    if isinstance(module, layers):
                        weight = module.weight
                    else:
                        weight = module.weight
                    # Áp dụng ràng buộc max-norm
                    weight_flat = weight.view(weight.size(0), -1)
                    norms = weight_flat.norm(dim=1, keepdim=True)
                    desired = torch.clamp(norms, min=0, max=max_norm)
                    scale = desired / (norms + 1e-7)
                    weight_flat.mul_(scale)


def is_child_of_modules(module, modules):
    for parent_module in modules:
        if module is parent_module:
            return True
        for sub_module in parent_module.modules():
            if module is sub_module:
                return True
    return False