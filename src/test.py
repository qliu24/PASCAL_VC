from scipy.spatial.distance import cdist
from FeatureExtractor import *
from config_PASCAL_VC import *

extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer='pool4', which_snapshot=0)
# imgf = '/home/qing/Downloads/huiyu.jpg'
# img = cv2.imread(imgf)
# # img = myresize(img, 224, 'short')
# layer_feature1 = extractor.extract_feature_image_all(img)

# imgf = '/home/qing/Downloads/qing.jpg'
# img = cv2.imread(imgf)
# # img = myresize(img, 224, 'short')
# layer_feature2 = extractor.extract_feature_image_all(img)

# with open('/home/qing/Downloads/features.pickle','wb') as fh:
#     pickle.dump([layer_feature1,layer_feature2], fh)