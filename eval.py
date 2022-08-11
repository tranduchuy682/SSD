from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from datetime import datetime

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
backbones = ["mobilenetv3","resnet18","vgg16"]


# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model, epoch):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    for max_overlap in [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
        print('max_overlap: ', max_overlap)
    # Make sure it's in eval mode
        model.eval()

        # Lists to store detected and true boxes, labels, scores
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()
        true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

        with torch.no_grad():
            # Batches
            start=datetime.now()
            for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
                images = images.to(device)  # (N, 3, 300, 300)

                # Forward prop.
                predicted_locs, predicted_scores = model(images)

                # Detect objects in SSD output
                det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                        min_score=0.01, max_overlap=max_overlap,
                                                                                        top_k=200)
                # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

                # Store this batch's results for mAP calculation
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]
                difficulties = [d.to(device) for d in difficulties]

                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                true_boxes.extend(boxes)
                true_labels.extend(labels)
                true_difficulties.extend(difficulties)

            runtime = datetime.now()-start
            # Calculate mAP
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

        # Print AP for each class
        pp.pprint(APs)
        print(runtime)
        print('\nMean Average Precision (mAP) of ' + backbone + '(' + str(epoch) + ' epochs): %.3f' % mAP)


if __name__ == '__main__':
    for backbone in backbones:
        checkpoint = 'weight/checkpoint_ssd300'+backbone+'.pth.tar'
        # Load model checkpoint that is to be evaluated
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        epoch = checkpoint['epoch']
        model = model.to(device)

        # Switch to eval mode
        model.eval()
        evaluate(test_loader, model, epoch)
