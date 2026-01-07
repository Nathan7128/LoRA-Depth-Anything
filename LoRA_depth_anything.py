import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForDepthEstimation, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model

# 1. Configuration du Dataset avec gestion des dimensions et des NaNs
class DatasetImages():

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, "images")
        self.depth_npy_path = os.path.join(dataset_path, "depth")
        self.images_dict = {}
        self.depth_npy_dict = {}

        self.load_images()
        self.load_depth_npy()

    def load_images(self):
        image_filenames = os.listdir(self.images_path)
        for image_name in image_filenames:
            image_file = os.path.join(self.images_path, image_name)
            self.images_dict[image_name] = Image.open(image_file)

    def load_depth_npy(self):
        depth_npy_filenames = os.listdir(self.depth_npy_path)
        for depth_npy_name in depth_npy_filenames:
            depth_npy_file = os.path.join(self.depth_npy_path, depth_npy_name)
            self.depth_npy_dict[depth_npy_name] = np.load(depth_npy_file)

dataset = DatasetImages("DATASET_DEVOIR")

class DepthDataset(Dataset):
    def __init__(self, pairs, images_path, depth_path, image_processor):
        self.pairs = pairs
        self.images_path = images_path
        self.depth_path = depth_path
        self.image_processor = image_processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, depth_name = self.pairs[idx]
        
        # Chargement
        image = Image.open(os.path.join(self.images_path, img_name)).convert('RGB')
        depth_numpy = np.load(os.path.join(self.depth_path, depth_name))
        
        # CORRECTION ICI : Gestion des canaux de profondeur
        # Si la depth a 3 canaux (H, W, 3), on ne garde que le premier (H, W)
        if len(depth_numpy.shape) == 3:
            depth_numpy = depth_numpy[:, :, 0]
        
        # Prétraitement de l'image
        inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Conversion en Tensor
        depth_tensor = torch.from_numpy(depth_numpy).float()
        
        # On a maintenant une forme (H, W). 
        # On ajoute (Batch, Channel) pour obtenir (1, 1, H, W) requis par interpolate
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        
        # Récupération de la taille cible
        target_size = inputs['pixel_values'].shape[-2:]
        
        # Interpolation
        depth_resized = F.interpolate(depth_tensor, size=target_size, mode='nearest')
        
        # On retire les dimensions pour revenir à (H, W) pour les labels
        depth_resized = depth_resized.squeeze()

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': depth_resized
        }

# 2. Chargement du Modèle et Processor
model_id = "depth-anything/Depth-Anything-V2-Small-hf"
image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id)

# 3. Configuration LoRA Correcte pour la Vision
# On cible tous les modules linéaires du Transformer pour un meilleur apprentissage
# On retire 'task_type' pour éviter l'erreur "input_ids"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense", "fc1", "fc2"], 
    lora_dropout=0.05,
    bias="none",
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# 4. Préparation des données (comme dans ton code original)
# Assure-toi que 'dataset' (ta classe DatasetImages) est bien instancié avant
image_files = sorted(os.listdir(dataset.images_path))
depth_files = sorted(os.listdir(dataset.depth_npy_path))
# Filtrer pour s'assurer que les fichiers correspondent bien si nécessaire
all_pairs = list(zip(image_files, depth_files))
random.shuffle(all_pairs)

split_idx = int(0.8 * len(all_pairs))
train_pairs = all_pairs[:split_idx]
eval_pairs = all_pairs[split_idx:]

train_dataset = DepthDataset(train_pairs, dataset.images_path, dataset.depth_npy_path, image_processor)
eval_dataset = DepthDataset(eval_pairs, dataset.images_path, dataset.depth_npy_path, image_processor)

# 5. Trainer Personnalisé pour gérer la Loss et les NaNs
class DepthTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        
        # L'output du modèle peut être légèrement différent de la taille d'entrée (padding)
        # On s'assure que la prédiction matche les labels
        if predicted_depth.shape[-2:] != labels.shape[-2:]:
            predicted_depth = F.interpolate(
                predicted_depth.unsqueeze(1), 
                size=labels.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)

        # Masquage des valeurs invalides (NaNs ou inf)
        # On suppose que la profondeur valide est > 0 et n'est pas NaN
        valid_mask = ~torch.isnan(labels) & ~torch.isinf(labels) & (labels > 0)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predicted_depth.device, requires_grad=True)

        # Calcul de la Loss (L1 Loss est souvent mieux pour la profondeur que MSE)
        loss = F.l1_loss(predicted_depth[valid_mask], labels[valid_mask])
        
        return (loss, outputs) if return_outputs else loss

# 6. Arguments d'entraînement
args = TrainingArguments(
    output_dir="output_depth_lora",
    remove_unused_columns=False, # Important pour garder 'labels'
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4, # Un peu plus bas pour LoRA
    per_device_train_batch_size=4, # Ajuste selon ta VRAM (128 est énorme pour des images)
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # Simule un batch plus grand
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    label_names=["labels"], # Indique au Trainer de ne pas supprimer cette colonne
)

# Fonction de collation simple
def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}

# 7. Lancement
trainer = DepthTrainer(
    model=lora_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

trainer.train()