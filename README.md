# Task

<img src="assets/1.png" alt="" width="1000"/>

<img src="assets/2.png" alt="" width="1000"/>

Hi guys. Here is the summary of everything i found so far, as well as ideas for potential solutions.

**1. The literature review**

can be found here  
https://link.springer.com/article/10.1007/s10462-024-10978-x  


First, i installed the latest Pytorch as follows  
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Next, i installed the model and cv2 library as follows

```
pip install ultralytics opencv-python-headless
```

To run the code, run
```
python3 HAK.py -d
```

 So far, i used yolo11n.pt because it is the lighest model. There are different options which we can find on the website.

 <img src="assets/Models.JPG" alt="" width="300"/>

 The code and the video example is attached
