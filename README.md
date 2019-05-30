# Carton-box-detector-in-a-video-using-YOLOv3
This model will detect carton boxes in a video(like from CCTV camera etc),and count them .This is a application in used during supply chain management and logistics.

1.1 Download data.
We can goto ImageNet website and download the images urls.

OR

Search in images.google.com 
and paste foloowing url :

urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));

in the console(Shift+J).

1.2
Split the images folder into train and test folders with 80%-90% and 20%-10% of total images in each folder respectively.
(Check the Split.py for code)

1.3 Darknet
 Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
https://pjreddie.com/darknet/


1.4 YOLOv3 architechture

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*PHv_qaMpnFM21Vw9wajYuA.png">
</p>
