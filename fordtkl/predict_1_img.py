from unicodedata import name
import numpy as np
import cv2
from PIL import Image
import torch
import albumentations
import segmentation_models_pytorch
from PIL import Image
import matplotlib.pyplot as plt
encoder = "densenet201"
def convert_to_tensor(x,**kwargs):
    return x.transpose(2,0,1).astype("float32")

encoder_wts = "imagenet"
preprocess_func = segmentation_models_pytorch.encoders.get_preprocessing_fn(encoder,encoder_wts)   

device = "cpu"
best_model = torch.load('best_model.h5', map_location={'cuda:0': 'cpu'})

IMAGE_FG = "staticFiles/uploads/pr_mask.png"
OUT_NAME = "staticFiles/uploads/blend.jpg"

BLEND_X = 0
BLEND_Y = 0
BLEND_OPAQUE = 0.5

def blend(bg, fg, x, y, opaque=1.0, gamma=0):
    """
        bg: background (color image)
        fg: foreground (color image)
        x, y: top-left point of foreground image (percentage)
    """
    h, w = bg.shape[:2]
    fg = cv2.resize(fg, (w, h))
    x_abs, y_abs = int(x*w), int(y*h)
    
    fg_h, fg_w = fg.shape[:2]
    # patch = bg[y_abs:y_abs+fg_h, x_abs:x_abs+fg_w, :]
    patch = bg[y_abs:y_abs+fg_h, x_abs:x_abs+fg_w, :]
    blended = cv2.addWeighted(src1=patch, alpha=1-opaque, src2=fg, beta=opaque, gamma=gamma)
    result = bg.copy()
    result[y_abs:y_abs+fg_h, x_abs:x_abs+fg_w, :] = blended
    return result

def main(bg_path, fg_path):
    img_bg = cv2.imread(bg_path)
    img_fg = cv2.imread(fg_path)
    result = blend(bg=img_bg, fg=img_fg, x=BLEND_X, y=BLEND_Y, opaque=BLEND_OPAQUE)
    cv2.imwrite(OUT_NAME, result)
    # print("\x1b[1;%dm" % (31) + "Saved image @ %s" % OUT_NAME + "\x1b[0m")
    pass

# if __name__ == "__main__":
#     main(bg_path=IMAGE_BG, fg_path=IMAGE_FG)

transform = albumentations.Compose([
            albumentations.Resize(height=224,width=224,interpolation=Image.BILINEAR),
            albumentations.Lambda(image=preprocess_func),
            albumentations.Lambda(image=convert_to_tensor)
        ])
def seg(imagespath):
    # imagespath = "path to input image"
    # imagevis = cv2.cvtColor(cv2.imread(imagespath),cv2.COLOR_BGR2RGB)
    # imagevis = cv2.resize(imagevis, (1280, 995))
    image_raw = Image.open(imagespath)
    image = np.array(image_raw)
    augmentations = transform(image=image)
    image = augmentations["image"]
    # print(image)
    x_tensor = torch.from_numpy(image)
    x_tensor = x_tensor.to(device).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = pr_mask.squeeze().cpu().numpy().round()
    # res = (pr_mask - pr_mask.min()) / (pr_mask.max() - pr_mask.min() + 1e-8)
    pr_mask = (pr_mask*255).astype(np.uint8)
    # pr_mask = cv2.resize(pr_mask, (1280, 995)) #960, 1280
    # out = cv2.addWeighted(src1=imagevis, alpha=0.8, src2=pr_mask, beta=0.2, gamma=0)
    # plt.imshow(np.array(imagevis), cmap = 'gray')
    # plt.imshow(np.array(pr_mask),alpha=0.4, cmap = 'Reds')
    # plt.title("Ground Truth")
    # plt.show()
    # plt.savefig('staticFiles/uploads/res.png')
    
    cv2.imwrite('staticFiles/uploads/pr_mask.png', pr_mask)
    main(imagespath, IMAGE_FG)
    return OUT_NAME


# seg('/home/mcn/DucHuy_K63/Det/SSD-base/fordtkl/staticFiles/uploads/abc.png')