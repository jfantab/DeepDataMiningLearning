import torch

from DeepDataMiningLearning.detection.models import create_detectionmodel

if __name__ == "__main__":
    device='cuda:0'
    model, preprocess, classes = create_detectionmodel('yolov8', num_classes=80, trainable_layers=0, ckpt_file='/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt', fp16=False, device= device)
    scripted_model = torch.jit.script(model)
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    scripted_output = scripted_model(dummy_input)
    scripted_model.save('scripted_yolov8.pt')

    