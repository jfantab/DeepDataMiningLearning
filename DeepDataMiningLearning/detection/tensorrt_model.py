import os 

import torch.onnx
import tensorrt
import numpy as np

from DeepDataMiningLearning.detection.models import create_detectionmodel

if __name__ == "__main__":
    # Setting the device
    device='cuda:0'

    # Importing the YOLOv8 model
    model, preprocess, classes = create_detectionmodel('yolov8', num_classes=80, trainable_layers=0, ckpt_file='/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt', fp16=False, device= device)

    # Checking Tensor RT
    print(tensorrt.__version__)
    assert tensorrt.Builder(tensorrt.Logger())

    # Exporting model to ONNX
    BATCH_SIZE = 32
    dummy_input = torch.rand(BATCH_SIZE, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, "yolov8_model.onnx", verbose=False)

    # Optimizing model with Tensor RT
    USE_FP16 = True
    target_dtype = np.float16 if USE_FP16 else np.float32

    ## I attempted here to run the terminal commands for 
    ## trtexec, but I had issues trying to install trtexec

    # if USE_FP16:
    #     os.system("trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16")
    # else:
    #     os.system("trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch")

    