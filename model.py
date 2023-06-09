from torchvision import models
from data import norm
import torch
import os
# from Resnet import resnet152_denoise, resnet101_denoise, resnet152
# from resnet import resnet152, resnet101
import timm

# load models from torchvision.models, you also can load your own models
def load_models(source_model_names, device):
    source_models = []
    for model_name in source_model_names:
        print("Loading model: {}".format(model_name))
        if model_name == 'resnet50':
            # source_model = models.resnet50(pretrained=True).eval()
            source_model = models.resnet50()
            source_model.load_state_dict(torch.load("torch/hub/checkpoints/resnet50-0676ba61.pth"))
            source_model.eval()
        elif model_name == 'resnet101':
            # source_model = models.resnet101(pretrained=True).eval()
            source_model = models.resnet101()
            source_model.load_state_dict(torch.load("models/resnet101-5d3b4d8f.pth"))
            source_model.eval()
            # source_models.append(source_model)
        elif model_name == 'resnet152':
            source_model = models.resnet152(pretrained=True).eval()
        elif model_name == 'inception_v3':
            source_model = models.inception_v3()
            source_model.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
            source_model.eval()
        elif model_name == 'inception_v4':
            # # source_model = models.inception_v4()
            # source_model = torch.load(os.path.join('./weight', 'inception_v4.pth'))
            # source_model.eval()
            source_model = timm.create_model('inception_v4', pretrained=True)
            source_model.eval()
        # elif model_name == 'resnet101_adv':
        #     source_model = resnet101_denoise()
        #     loaded_state_dict = torch.load(os.path.join('./weight', 'Adv_Denoise_Resnext101.pytorch'))
        #     source_model.load_state_dict(loaded_state_dict, strict=True)
        #     # source_model = torch.load(os.path.join('./weight', 'Adv_Denoise_Resnext101.pytorch'))
        #     source_model.eval()
        #     # source_model.cuda()
        elif model_name == 'vgg16':
            # source_model = models.vgg16(pretrained=True).eval()
            source_model = models.vgg16()
            source_model.load_state_dict(torch.load("torch/hub/checkpoints/vgg16-397923af.pth"))
            source_model.eval()
        elif model_name == 'densenet161':
            source_model = models.densenet161(pretrained=True).eval()
        elif model_name == 'ens_adv_inception_resnet_v2':
            source_model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
            source_model.eval()
        elif model_name == 'adv_inception_v3':
            source_model = timm.create_model('adv_inception_v3', pretrained=True)
            source_model.eval()
        elif model_name == 'inception_resnet_v2':
            source_model = timm.create_model('inception_resnet_v2', pretrained=True)
            source_model.eval()
        elif model_name == 'vit_base_patch32_224':
            source_model = timm.create_model('vit_base_patch32_224', pretrained=True)
            source_model.eval()
        elif model_name == 'vit_large_patch32_224':
            source_model = timm.create_model('vit_large_patch32_224', pretrained=True)
            source_model.eval()
        elif model_name == 'vit_large_patch16_224':
            source_model = timm.create_model('vit_large_patch16_224', pretrained=True)
            source_model.eval()
        elif model_name == 'vit_small_patch32_224':
            source_model = timm.create_model('vit_small_patch32_224', pretrained=True)
            source_model.eval()
        elif model_name == 'deit3_base_patch16_224':
            source_model = timm.create_model('deit3_base_patch16_224', pretrained=True)
            source_model.eval()
        elif model_name == 'pit_b_224':
            source_model = timm.create_model('pit_b_224', pretrained=True)
            source_model.eval()
        elif model_name == 'pit_s_224':
            source_model = timm.create_model('pit_s_224', pretrained=True)
            source_model.eval()
        elif model_name == 'visformer_small':
            source_model = timm.create_model('visformer_small', pretrained=True)
            source_model.eval()
        elif model_name == 'levit_256':
            source_model = timm.create_model('levit_256', pretrained=True)
            source_model.eval()
        elif model_name == 'convit_base':
            source_model = timm.create_model('convit_base', pretrained=True)
            source_model.eval()            


        source_model.to(device)
        source_models.append(source_model)
    print(len(source_models))
    return source_models


# calculate the ensemble logits of models
def get_logits(X_adv, source_models):
    ensemble_logits = 0
    for source_model in source_models:
        ensemble_logits += source_model(norm(X_adv))  # ensemble

    ensemble_logits /= len(source_models)
    return ensemble_logits




