# [RetinaFace](https://github.com/heewinkim/retinaface)
 
 face detector with landmarks, RetinaFace PyPI implement
 
 reference : https://github.com/peteryuX/retinaface-tf2 
 
![](https://img.shields.io/badge/python-3.7-blue)
![](https://img.shields.io/badge/tensorflow-2.5.0-orange)

[![Run on Streamlit](https://img.shields.io/badge/Run-STREAMLIT-green)](https://heewinkim-retinaface-streamlit-app-3nx0av.streamlitapp.com/)

<a href="https://www.buymeacoffee.com/heewinkim" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

----

### INSTALL
```sh
pip3 install refinaface
```

### USEAGE

```python

#pip3 install opencv-python
import cv2 
from retinaface import RetinaFace

# init with normal accuracy option
detector = RetinaFace(quality="normal")

# same with cv2.imread,cv2.cvtColor 
rgb_image = detector.read("data/hian.jpg")

faces = detector.predict(rgb_image)
# faces is list of face dictionary
# each face dictionary contains x1 y1 x2 y2 left_eye right_eye nose left_lip right_lip
# faces=[{"x1":20,"y1":32, ... }, ...]

result_img = detector.draw(rgb_image,faces)

# save ([...,::-1] : rgb -> bgr )
cv2.imwrite("data/result_img.jpg",result_img[...,::-1])

# show using cv2
# cv2.imshow("result",result_img[...,::-1])
# cv2.waitKey()
```

### result with drawing
![](./data/result_img.jpg)

