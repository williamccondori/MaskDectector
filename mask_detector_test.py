import cv2
import torch
import numpy as np

from PIL import Image
#from mtcnn.mtcnn import mtcnn
from facenet_pytorch import MTCNN

import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def test(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])

    # Load image.
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    # Load model.
    #model = torch.load('models/result.model', map_location=device)
    model = torch.load('models/result.model', map_location=device)
    model = model.to(device)
    model.eval()

    outputs = model(image)
    prob, pred = torch.max(outputs, 1)

    labels = ['MASK', 'NO MASK']
    colors = [(0,255,0), (255,0,0)]
    return labels[int(pred)], colors[int(pred)]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = cv2.cvtColor(cv2.imread('multimedia/test3.jpg'), cv2.COLOR_BGR2RGB)

    mtcnn = MTCNN(image_size=256, device=device)
    
    (result) = mtcnn.detect(image)

    if not result[0] is None:
        for i in range(len(result[0])):
            if float(result[1][i]) >= 0.9:
                box = result[0][i].tolist()
                box = [ int(x) for x in box ]
                face = image[box[1]:box[3],box[0]:box[2]]

                label, color = test(Image.fromarray(face))

                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]) , color, 2)
                cv2.putText(image, label, (box[0],box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)

    """
    detector = MTCNN()
    results = detector.detect_faces(image)

    for result in results:
        box = result['box']
        face = image[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
    
        # Mask detector
        label, color = test(image)

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]) , color, 2)
        cv2.putText(image, label, (box[0],box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    """

    # Show image
    plt.imshow(image)
    plt.show()
    

if __name__ == "__main__":
    main()
