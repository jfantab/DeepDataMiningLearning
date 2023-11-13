# # Shell script to run evaluator files 

export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128 # To allow internet access via GPU

# python3 myinference.py
# python3 myevaluator.py

python3 torchscript_model.py
# python3 tensorrt_model.py