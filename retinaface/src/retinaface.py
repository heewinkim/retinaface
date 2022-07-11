import tensorflow as tf
import numpy as np
import cv2
import os


class RetinaFace(object):

    def __init__(self,quality='normal'):
        """
        :param quality: one of [ 'high','normal','speed' ]
        """

        if quality == 'normal':
            self._resizeFunc = lambda v: self._resize_image(v[0], **{v[1]: 800})
        elif quality =='speed':
            self._resizeFunc = lambda v: self._resize_image(v[0], **{v[1]: 320})
        else:
            self._resizeFunc = lambda v: v[0]

        print("model[{} quality] init ..".format(quality))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with tf.io.gfile.GFile(current_dir+'/frozen_graph.pb', "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        self._model = wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, ['x:0']),
            tf.nest.map_structure(import_graph.as_graph_element, ['Identity:0'])
        )
        self.predict(np.zeros((320,320,3),dtype=np.float32))
        print("model success !")

    def _resize_image(self,img_cv, width=0, height=0):
        """
        resizing image with keeping ratio, you can offer width, height in pixel or percent

        :param width: pixel size or ratio
        :param hegith: pixel size or ratio
        :param img_cv: cv image
        :return: cv image
        """

        h, w, _ = img_cv.shape

        if width == 0 and height == 0:
            return cv2.resize(img_cv, None, fx=0.5, fy=0.5)

        if (width < 1 and width != 0) or (height < 1 and height != 0):

            if height == 0:
                re_h = width
                re_w = width
                return cv2.resize(img_cv, None, fx=re_w, fy=re_h)

            elif width == 0:
                re_w = height
                re_h = height
                return cv2.resize(img_cv, None, fx=re_w, fy=re_h)

        else:

            if height == 0:
                re_h = int(width / w * h)
                re_w = width
                return cv2.resize(img_cv, (re_w, re_h))
            elif width == 0:
                re_w = int(height / h * w)
                re_h = height
                return cv2.resize(img_cv, (re_w, re_h))

    def read(self,image_path):
        """
        read image from path

        :param image_path:
        :return: rgb image, float32
        """
        img_cv = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB).astype(np.float32)
        return rgb_image

    def _predict(self,rgb_image,threshold=0.95):
        """
        detect face in rgb image

        :param rgb_image: rgb image, ! width, height have to multiplier of 32 !, float32
        :param threshold: threshold of confidence
        :return: faces(list), eache face(dict) has a key = [ x1, y1, x2, y2,left_eye,right_eye,nose,left_lip,right_lip ]
        """
        img_h, img_w = rgb_image.shape[:2]

        # preprocessing (padding)
        x = tf.cast(rgb_image, dtype=tf.float32)

        # prediction
        outputs = tf.squeeze(self._model(x[tf.newaxis, ...]), axis=0)

        # postprocessing (remove-padding,ratio to pixcel, threshold)
        outputs = tf.concat([
            tf.reshape(tf.multiply(tf.reshape(tf.slice(outputs, [0, 0], [-1, 14]), [-1, 7, 2]),[img_w, img_h]),[-1, 14]),
            tf.slice(outputs, [0, 14], [-1, 2])
        ], axis=1)
        outputs = tf.gather_nd(outputs, tf.where(tf.squeeze(tf.slice(outputs, [0, 15], [-1, 1]), axis=-1) >= threshold))

        faces = []
        for bbox in outputs:
            x1, y1, x2, y2 = list(map(int, bbox[:4]))
            left_eye, right_eye, nose, left_lip, right_lip = list(map(tuple, np.reshape(bbox, [-1, 2]).astype(np.int)[2:-1]))
            faces.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'left_eye': left_eye, 'right_eye': right_eye, 'nose': nose, 'left_lip': left_lip, 'right_lip': right_lip
            })

        return faces

    def predict(self,rgb_image,threshold=0.95):
        """
        detect face in rgb image

        :param rgb_image: rgb image, any size, float32
        :param threshold: threshold of confidence
        :return: faces(list), eache face(dict) has a key = [ x1, y1, x2, y2,left_eye,right_eye,nose,left_lip,right_lip ]
        """
        img_h_, img_w_ = rgb_image.shape[:2]

        if img_h_>img_w_:
            rgb_image = self._resizeFunc([rgb_image,'height'])
        else:
            rgb_image = self._resizeFunc([rgb_image, 'width'])

        img_h, img_w = rgb_image.shape[:2]

        # preprocessing (padding)
        max_steps = 32
        img_h_pad = max_steps - img_h % max_steps if img_h and img_h % max_steps != 0 else 0
        img_w_pad = max_steps - img_w % max_steps if img_w and img_w % max_steps != 0 else 0
        padded_img = tf.pad(rgb_image, [[0, img_h_pad], [0, img_w_pad], [0, 0]])
        x = tf.cast(padded_img, dtype=tf.float32)

        # prediction
        outputs = tf.squeeze(self._model(x[tf.newaxis, ...]), axis=0)

        # postprocessing (remove-padding,ratio to pixcel, threshold)
        outputs = tf.concat([
            tf.reshape(tf.multiply(tf.reshape(tf.slice(outputs, [0, 0], [-1, 14]), [-1, 7, 2]),
                                   [tf.add(img_w_pad, img_w if img_w else 0),
                                    tf.add(img_h_pad, img_h if img_h else 0)]),
                       [-1, 14]),
            tf.slice(outputs, [0, 14], [-1, 2])
        ], axis=1)
        outputs = tf.gather_nd(outputs, tf.where(tf.squeeze(tf.slice(outputs, [0, 15], [-1, 1]), axis=-1) >= threshold))

        faces=[]
        for bbox in outputs:
            w_ex = img_w_ / img_w
            h_ex = img_h_ / img_h
            x1, y1, x2, y2 = list(map(int, np.multiply(bbox[:4],[w_ex,h_ex,w_ex,h_ex])))
            left_eye,right_eye,nose,left_lip,right_lip = list(map(tuple,np.multiply(np.reshape(bbox, [-1, 2]),[w_ex,h_ex]).astype(np.int)[2:-1]))
            faces.append({
                'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                'left_eye':left_eye,'right_eye':right_eye,'nose':nose,'left_lip':left_lip,'right_lip':right_lip
            })

        return faces

    def draw(self,rgb_image, faces,thickness=3,**kwargs):
        """

        :param rgb_image: rgb_image , same size of predict's input
        :param faces: result of predict method
        :param thickness: thickness of line's
        :keyword colors: list of color, each color element mean [ faceRect, left_eye, right_eye, nose, left_lip, right_lip ]
        :return: result image
        """
        darwing_img = rgb_image.copy()
        if 'colors' in kwargs:
            colors = kwargs['colors']
        else:
            colors = [(255, 0, 255),(255, 0, 0),(0, 255, 0),(0, 0, 255),(0, 0, 0),(255, 0, 255)]

        for face in faces:
            cv2.rectangle(darwing_img, (face['x1'], face['y1']), (face['x2'], face['y2']), colors[0], thickness)
            cv2.circle(darwing_img, face['left_eye'], 1, colors[1], thickness)
            cv2.circle(darwing_img, face['right_eye'], 1, colors[2], thickness)
            cv2.circle(darwing_img, face['nose'], 1, colors[3], thickness)
            cv2.circle(darwing_img, face['left_lip'], 1, colors[4], thickness)
            cv2.circle(darwing_img, face['right_lip'], 1, colors[5], thickness)
        return darwing_img