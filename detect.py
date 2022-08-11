from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from datetime import datetime


backbones = ['mobilenetv3','resnet18','vgg16']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Load model checkpoint
models = {}
for bb in backbones:
    checkpoint = 'weight/checkpoint_ssd300'+bb+'.pth.tar'
    checkpoint = torch.load(checkpoint, map_location={'cuda:0': 'cpu'})
    # checkpoints.append(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    models[bb] = model
# start_epoch = checkpoint['epoch'] + 1
# print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)


# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(input_image, min_score, max_overlap, top_k, bb, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param input_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, 3 PIL Image
    """
    original_image = input_image
    model = models[bb]
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    start=datetime.now()
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                            max_overlap=max_overlap, top_k=top_k)
    runtime = datetime.now()-start
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_scores = det_scores[0].to('cpu')
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper()+ ': ' + str(det_scores[i].item()))
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper() + ': ' + str(det_scores[i].item()), fill='white',
                font=font)
    del draw
    return annotated_image, runtime


# if __name__ == '__main__':
#     img_path = './AllDatabase/LISCDatabase/Main Dataset/lymp/11.bmp'
#     original_image = Image.open(img_path, mode='r')
#     original_image = original_image.convert('RGB')
#     # display(detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)) .show()
#     annotated_images = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
#     for bb in annotated_images:
#         annotated_images[bb].save("output/annotated_image_"+bb+".jpg")
