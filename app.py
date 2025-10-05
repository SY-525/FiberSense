from flask import Flask, jsonify, request, send_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
import timm
import base64
from flask_cors import CORS

class FiberEnsembleModelImproved(nn.Module):
    """Improved ensemble with enhanced regularization."""

    def __init__(self, num_classes, dropout_rate=0.5):
        super(FiberEnsembleModelImproved, self).__init__()
        self.num_classes = num_classes

        # Model 1: EfficientNet-B3
        print("Loading EfficientNet-B3...")
        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=True)
        eff_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Identity()

        # Model 2: ConvNeXt-Tiny
        print("Loading ConvNeXt-Tiny...")
        self.convnext = timm.create_model('convnext_tiny', pretrained=True)
        conv_features = self.convnext.head.fc.in_features
        self.convnext.head.fc = nn.Identity()

        self.eff_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(eff_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, num_classes)
        )

        self.conv_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, num_classes)
        )

        # Fusion classifier
        total_features = eff_features + conv_features
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(total_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, num_classes)
        )

        print(f"Improved ensemble model created with {num_classes} classes")
        print(f"Dropout rate: {dropout_rate}")

    def forward(self, x):
        # Extract features
        eff_features = self.efficientnet(x)
        conv_features = self.convnext(x)

        # Individual predictions
        eff_output = self.eff_classifier(eff_features)
        conv_output = self.conv_classifier(conv_features)

        # Fusion prediction
        combined_features = torch.cat([eff_features, conv_features], dim=1)
        fusion_output = self.fusion_classifier(combined_features)

        # Weighted ensemble
        final_output = 0.3 * eff_output + 0.3 * conv_output + 0.4 * fusion_output

        return final_output, eff_output, conv_output, fusion_output

def load_fiber_ensemble_model(checkpoint_path, num_classes=10, device='cpu'):
    """Load the FiberEnsembleModelImproved from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FiberEnsembleModelImproved(num_classes=num_classes, dropout_rate=0.5)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, checkpoint

def test_single_image_fiber_ensemble(checkpoint_path, pil_image, class_names=None, device='cpu', num_classes=10):
    """Test a single image using the FiberEnsembleModelImproved from checkpoint."""
    
    # Load ensemble model from checkpoint
    model, checkpoint = load_fiber_ensemble_model(checkpoint_path, num_classes, device)
    
    # Get class names
    if class_names is None:
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        elif 'classes' in checkpoint:
            class_names = checkpoint['classes']
        elif 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
            class_names = [None] * len(class_to_idx)
            for class_name, idx in class_to_idx.items():
                class_names[idx] = class_name
        else:
            class_names = [
                'polyester', 'nylon', 'viscose', 'polypropylene', 'acrylics', 
                'flax', 'hemp', 'acetate', 'lyocell', 'cotton'
            ]
            print("Warning: Using default class names.")
            
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Apply transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = test_transform(pil_image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Make predictions
    with torch.no_grad():
        final_output, eff_output, conv_output, fusion_output = model(image_tensor)
        
        final_probs = F.softmax(final_output, dim=1)
        eff_probs = F.softmax(eff_output, dim=1)
        conv_probs = F.softmax(conv_output, dim=1)
        fusion_probs = F.softmax(fusion_output, dim=1)
    
    # Get predictions
    final_confidence, final_predicted = torch.max(final_probs, 1)
    top3_prob, top3_indices = torch.topk(final_probs, 3, dim=1)

    _, eff_pred = torch.max(eff_probs, 1)
    _, conv_pred = torch.max(conv_probs, 1)
    _, fusion_pred = torch.max(fusion_probs, 1)

    # OPTIMIZED LAYOUT: Better proportions and spacing
    fig = plt.figure(figsize=(52, 42))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.25], width_ratios=[1.2, 1], 
                        hspace=0.28, wspace=0.32, 
                        left=0.06, right=0.96, top=0.96, bottom=0.04)

    # TOP LEFT: Input image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(pil_image)
    ax_img.set_title('Input Image', fontsize=95, fontweight='bold', pad=25)
    ax_img.axis('off')
    ax_img.set_aspect('equal')

    # TOP RIGHT: Top 3 Predictions (now narrower)
    ax_top3 = fig.add_subplot(gs[0, 1])
    top3_classes = [class_names[idx] for idx in top3_indices[0]]
    top3_probs_np = top3_prob[0].cpu().numpy()

    bars = ax_top3.barh(range(3), top3_probs_np, color='#4A7B9D', height=0.65)
    ax_top3.set_yticks(range(3))
    ax_top3.set_yticklabels(top3_classes, fontsize=90, weight='bold')
    ax_top3.set_xlabel('Confidence', fontsize=98, weight='bold')
    ax_top3.set_title('Ensemble Top 3 Predictions', fontsize=95, fontweight='bold', pad=30)
    ax_top3.tick_params(axis='x', labelsize=50)
    ax_top3.tick_params(axis='y', pad=10)
    ax_top3.invert_yaxis()
    ax_top3.set_xlim(0, 1.05)

    for i, prob in enumerate(top3_probs_np):
        ax_top3.text(prob + 0.03, i, f'{prob*100:.1f}%', va='center', fontsize=92, weight='bold')

    # BOTTOM LEFT: Ensemble weights pie chart
    ax_weights = fig.add_subplot(gs[1, 0])
    weights = [0.3, 0.3, 0.4]
    labels = ['EfficientNet\n(30%)', 'ConvNeXt\n(30%)', 'Fusion\n(40%)']
    colors = ['#D97652', '#4A7B9D', '#5C8D89'] 
    wedges, texts, autotexts = ax_weights.pie(weights, labels=labels, colors=colors, autopct='%1.0f%%', 
                                                startangle=90, textprops={'fontsize': 92})
    for autotext in autotexts:
        autotext.set_fontsize(98)
        autotext.set_weight('bold')
    ax_weights.set_title('Ensemble Weights', fontsize=95, fontweight='bold', pad=30)

    # BOTTOM RIGHT: Individual Model Predictions
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_pred.text(0.05, 0.92, 'Individual Model Predictions:', fontsize=92, fontweight='bold', transform=ax_pred.transAxes)
    ax_pred.text(0.05, 0.74, f'EfficientNet-B3: {class_names[eff_pred[0]]}', fontsize=95, transform=ax_pred.transAxes, weight='bold', color='#D97652')
    ax_pred.text(0.05, 0.64, f'Confidence: {torch.max(eff_probs, 1)[0][0]:.3f}', fontsize=90, transform=ax_pred.transAxes)
    ax_pred.text(0.05, 0.47, f'ConvNeXt-Tiny: {class_names[conv_pred[0]]}', fontsize=95, transform=ax_pred.transAxes, weight='bold', color='#4A7B9D')
    ax_pred.text(0.05, 0.37, f'Confidence: {torch.max(conv_probs, 1)[0][0]:.3f}', fontsize=90, transform=ax_pred.transAxes)
    ax_pred.text(0.05, 0.20, f'Fusion Model: {class_names[fusion_pred[0]]}', fontsize=95, transform=ax_pred.transAxes, weight='bold', color='#5C8D89')
    ax_pred.text(0.05, 0.10, f'Confidence: {torch.max(fusion_probs, 1)[0][0]:.3f}', fontsize=90, transform=ax_pred.transAxes)
    ax_pred.axis('off')

    # Print results
    predicted_class = class_names[final_predicted[0]]
    confidence_score = final_confidence[0].item()

    print(f"\n=== FIBER ENSEMBLE PREDICTION RESULTS ===")
    print(f"Final Ensemble Prediction: {predicted_class}")
    print(f"Final Ensemble Confidence: {confidence_score*100:.2f}%")
    
    print(f"\nIndividual Model Predictions:")
    print(f"  EfficientNet-B3: {class_names[eff_pred[0]]} ({torch.max(eff_probs, 1)[0][0]*100:.2f}%)")
    print(f"  ConvNeXt-Tiny: {class_names[conv_pred[0]]} ({torch.max(conv_probs, 1)[0][0]*100:.2f}%)")
    print(f"  Fusion Model: {class_names[fusion_pred[0]]} ({torch.max(fusion_probs, 1)[0][0]*100:.2f}%)")

    if 'epoch' in checkpoint:
        print(f"\nModel was saved at epoch: {checkpoint['epoch']}")

    return predicted_class, final_probs[0].cpu().numpy(), {
        'efficientnet': eff_probs[0].cpu().numpy(),
        'convnext': conv_probs[0].cpu().numpy(),
        'fusion': fusion_probs[0].cpu().numpy()
    }, fig
    
app = Flask(__name__)
CORS(app)

@app.route('/ai', methods=['POST'])
def ai():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image part"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        image = Image.open(file.stream).convert('RGB')
        
        predicted_class, final_probs, individual_probs, fig = test_single_image_fiber_ensemble(
            checkpoint_path="ensemble_checkpoint_epoch_70.pth", 
            pil_image=image
        )
        
        buf = BytesIO()
        # Minimal padding to reduce top whitespace
        fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype='image/png', as_attachment=False, download_name="result.png")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)