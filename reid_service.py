from feature_service import FeatureService
import argparse
import parser
from PIL import Image
import time
import redis 
import pickle

REDIS_IMAGE_HSET_KEY="reid_image_list"
REDIS_RESULT_HSET_KEY="reid_result_list"

def fetch_one_from_redis(r):
    image_keys = r.hkeys(REDIS_IMAGE_HSET_KEY)
    if len(image_keys) == 0:
        return None, None
    else:
        return image_keys[0], r.hget(REDIS_IMAGE_HSET_KEY, image_keys[0])

def write_result_to_redis(r, key, output_feature):
    feature_bundle = pickle.dumps(output_feature)
    res = r.hset(REDIS_RESULT_HSET_KEY, key, feature_bundle)
    return res

def remove_finish_from_redis(r, key):
    if r.hexists(REDIS_IMAGE_HSET_KEY, key):
        res = r.hdel(REDIS_IMAGE_HSET_KEY, key)
        return res
    else:
        return True

def rebuild_image(image_str):
    img = pickle.loads(image_str)
    return img

def process_redis_list(r, featureService):
    while True:
        key, img_value = fetch_one_from_redis(r)
        if key != None:
            img = rebuild_image(img_value)
            output_feature = featureService.feature(img)
            write_result_to_redis(r, key, output_feature)
            remove_finish_from_redis(r, key)
            print('process feature finish.')
        else:
            print('no image to process')
            time.sleep(1)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='2', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--img_dir', default='test.jpg', type=str, help='path of input image')
    opt = parser.parse_args()

    r = redis.Redis(host='localhost', port=6379, db=0)
    featureService = FeatureService(opt)
    
    process_redis_list(r, featureService)

    '''
    feature_service = FeatureService(opt)

    img_path = '/home/chenkai/project/Person_reID_baseline_pytorch/query_images/subfolder/train.jpg'
    img_pil = Image.open(img_path)
    print("----------------------------")
    output_feature = feature_service.feature(img_pil)
    print(output_feature)

    '''

    
