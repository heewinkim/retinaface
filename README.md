# [RetinaFace](https://github.com/heewinkim/retinaface)
 
 tensorflow 2 - PyPI implement
 
 reference : https://github.com/peteryuX/retinaface-tf2 

----

### INSTALL
    $pip3 install refinaface

### USEAGE

    #pip3 install opencv-python
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

    
