# -*- coding: utf-8 -*-
'''
https://github.com/Alireza-Akhavan
'''
import tensorflow as tf
import numpy as np
from facenet import face

class Verification:
    """
    تطابق یا عدم تطابق دو چهره
    """
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.images_placeholder = ''
        self.embeddings = ''
        self.phase_train_placeholder = ''
        self.embedding_size = ''
        self.session_closed = False

    def __del__(self):
        if not self.session_closed:
            self.session.close()

    def kill_session(self):
        self.session_closed = True
        self.session.close()

    def load_model(self, model):
        """
        بارگذاری مدل
        این تابع حتما باید قبل از توابع دیگر فراخوانی شود.
        ورودی این تابع مسیرمدل از قبل آموزش دیده برای استخراج ویژگی است.
        """
        face.load_model(model, self.session)

    def initial_input_output_tensors(self):
        """
        ایجاد تنسورهای ورودی و خروجی از روی مدل لود شده توسط تابع
        load_model
        """
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]


    def img_to_encoding(self, img, image_size):
        """
        محاسبه embedding یک تصویر
        در اینجا یک وکتور 128 تایی برای هر تصویر
        """
        image = face.make_image_tensor(img, False, False, image_size)
        
        feed_dict = {self.images_placeholder: image, self.phase_train_placeholder:False }
        emb_array = np.zeros((1, self.embedding_size))
        emb_array[0, :] = self.session.run(self.embeddings, feed_dict=feed_dict)

        return np.squeeze(emb_array)

