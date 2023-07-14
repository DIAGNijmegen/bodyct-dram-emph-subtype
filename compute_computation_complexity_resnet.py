from monai.networks.nets.resnet import resnet50, resnet34, resnet18
from ptflops import get_model_complexity_info

res503d = resnet50(spatial_dims=3)
res343d = resnet34(spatial_dims=3)
res183d = resnet18(spatial_dims=3)


macs, params = get_model_complexity_info(res503d, (3, 224, 224, 224), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity res503d: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters res503d: ', params))


macs, params = get_model_complexity_info(res343d, (3, 224, 224, 224), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity res343d: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters res343d: ', params))

macs, params = get_model_complexity_info(res183d, (3, 224, 224, 224), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity res183d: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters res183d: ', params))


res502d = resnet50(spatial_dims=2)
res342d = resnet34(spatial_dims=2)
res182d = resnet18(spatial_dims=2)

macs, params = get_model_complexity_info(res502d, (3, 224, 224), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity res502d: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters res502d: ', params))

macs, params = get_model_complexity_info(res342d, (3, 224, 224), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity res342d: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters res342d: ', params))

macs, params = get_model_complexity_info(res182d, (3, 224, 224), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity res182d: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters res182d: ', params))