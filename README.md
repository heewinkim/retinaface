# [RetinaFace](https://github.com/heewinkim/retinaface)
 
 face detector with landmarks, RetinaFace PyPI implement
 
 reference : https://github.com/peteryuX/retinaface-tf2 
 
![](https://img.shields.io/badge/python-3.6.1-blue)
![](https://img.shields.io/badge/tensorflow-2.0.0-orange)

----

### INSTALL
```sh
pip3 install refinaface
```

### USEAGE

```python

import cv2 

from retinaface import Retinaface()

detector = Retinaface(quality="normal")

# same with cv2.imread,cv2.cvtColor 
rgb_image = detector.read("path/to/image")

faces = detector.predict(rgb_image)
# faces is list of face dictionary
# each face dictionary contains x1 y1 x2 y2 left_eye right_eye nose left_lip right_lip
# faces=[{"x1":20,"y1":32, ... }, ...]

result_img = detector.draw(rgb_image,faces)
    
# save
# cv2.imwrite("result_img.jpg",result_img)

# show using cv2
# cv2.imshow("result",result_img)
# cv2.waitKey()
```
