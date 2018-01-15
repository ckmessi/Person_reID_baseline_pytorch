import redis
import cv2
import numpy as np
import pickle
import time
from PIL import Image

REDIS_IMAGE_HSET_KEY="reid_image_list"
REDIS_RESULT_HSET_KEY="reid_result_list"

r = redis.Redis(host='localhost', port=6379, db=0)

img = cv2.imread('/home/chenkai/project/Person_reID_baseline_pytorch/train.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pickled_object = pickle.dumps(img)

key_value = '00001'
r.hset(REDIS_IMAGE_HSET_KEY, key_value, pickled_object)

while True:
    key_exist = r.hexists(REDIS_RESULT_HSET_KEY, key_value)
    if key_exist == False:
        time.sleep(1)
    else:
        value = r.hget(REDIS_RESULT_HSET_KEY, key_value)
        value = pickle.loads(value)
        print(value)
        break

