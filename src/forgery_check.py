
#----------------------------------------------------------------------------
#                          FORGED ID DETECTION SCRIPT
#
#
#   This script uses the best model saved from the training loop
#   and pulls the field extraction and validation methods for the OCR model.
#
#   
#   USAGE: python forgery_check.py <image_path>
#   
#   - remember to enclose the actual img path with quotes
#
#-----------------------------------------------------------------------------


import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import pytesseract
import re
import sys


pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_stddev  = [0.229, 0.224, 0.225]

clean_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((224, 224)), 
    transforms.RandomGrayscale(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_stddev)
])

def load_model(weights_path='models/forgery_detector.pth'):

    model = models.efficientnet_b0(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.classifier[1].in_features, 1),
        nn.Sigmoid()
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model



def extract_fields(image_path):

    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text

def validate_fields(text):

    anomalies = 0
    total_checks = 4

    name_matches = re.findall(r'[A-Z]{2,}(?:\s[A-Z]{2,})+', text)
    exclude = ['XYZ', 'LOGISTICS', 'Driver']
    valid_names = [m for m in name_matches if not any(word in m for word in exclude)]
    if not valid_names:
        anomalies += 1

    if not re.search(r'DRV\d{6}', text):
        anomalies += 1

    if not re.search(r'\+63\s?\d{3}\s?\d{3}\s?\d{4}', text):
        anomalies += 1

    if not re.search(r'\d{2}/\d{2}/\d{4}', text):
        anomalies += 1

    return anomalies / total_checks



def get_cnn_score(image_path, model):

    img = Image.open(image_path).convert('RGB')
    tensor = clean_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(tensor).item()

    return score


def ensemble_score(image_path, model, cnn_weight=0.7, ocr_weight=0.3):

    cnn_score = get_cnn_score(image_path, model)
    ocr_score = validate_fields(extract_fields(image_path))

    final_score = (cnn_weight * cnn_score) + (ocr_weight * ocr_score)

    verdict = 'LIKELY FORGED' if final_score >= 0.5 else 'LIKELY GENUINE'

    return {
        'cnn_score'     : round(cnn_score, 4),
        'ocr_score'     : round(ocr_score, 4),
        'ensemble_score': round(final_score, 4),
        'verdict'       : verdict
    }

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python forgery_check.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_model()
    result = ensemble_score(image_path, model)

    print(f"CNN Score      : {result['cnn_score']}")
    print(f"OCR Score      : {result['ocr_score']}")
    print(f"Ensemble Score : {result['ensemble_score']}")
    print(f"Verdict        : {result['verdict']}")