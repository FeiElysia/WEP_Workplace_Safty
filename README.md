# Task

<img src="assets/1.png" alt="" width="500"/>

<img src="assets/2.png" alt="" width="500"/>

# Install

```
conda create -n wep python=3.10 -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-utils[decord]
```

# run
For Pose Estimation, Activity Classification, and PPE Style  
```
python main.py
# or
sh run.sh
```

For PPE detection and classification:
```
python HAK_ft.py
```  

# Results
check outputs for results
