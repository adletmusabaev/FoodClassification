"""
FIXED SOTA Model Training - Optimized for Small Dataset
Uses simpler architecture and more aggressive augmentation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2  # Lighter than EfficientNet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class FixedSOTAClassifier:
    """
    Optimized SOTA model for small datasets
    """
    
    def __init__(self, num_classes, img_size=160):  # 160 instead of 224
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def build_model(self, learning_rate=0.001):
        """Build optimized model"""
        
        # Use MobileNetV2 - lighter and works better on small datasets
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze all layers initially
        base_model.trainable = False
        
        # Build model
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Data augmentation (built-in)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomContrast(0.2)(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Simpler head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n" + "="*60)
        print("OPTIMIZED MODEL ARCHITECTURE")
        print("="*60)
        self.model.summary()
        
        return self.model
    
    def unfreeze_base_model(self, learning_rate=0.0001):
        """Unfreeze base model for fine-tuning"""
        
        # Find MobileNetV2 layer
        base_model = None
        for layer in self.model.layers:
            if 'mobilenet' in layer.name.lower():
                base_model = layer
                break
        
        if base_model is None:
            print(" Could not find base model")
            return
        
        # Unfreeze last 30 layers only
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n Base model partially unfrozen (last 30 layers)")
    
    def create_aggressive_augmentation(self):
        """Create very aggressive augmentation"""
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            brightness_range=[0.7, 1.3]
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        return train_datagen, test_datagen
    
    def prepare_data(self, train_dir, test_dir, batch_size=16):  # Smaller batch
        """Prepare data generators"""
        
        train_datagen, test_datagen = self.create_aggressive_augmentation()
        
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.class_names = list(train_gen.class_indices.keys())
        
        return train_gen, test_gen
    
    def get_callbacks(self, model_dir):
        """Get training callbacks"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',  # Changed to val_accuracy
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_gen, test_gen, epochs=50, model_dir='models_sota_fixed'):
        """Train the model"""
        
        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
        
        callbacks = self.get_callbacks(model_dir)
        
        # Phase 1: Train with frozen base (longer)
        print("\n Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=30,  # More epochs for phase 1
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen base
        print("\n Phase 2: Fine-tuning with partially unfrozen base...")
        self.unfreeze_base_model(learning_rate=0.00005)
        
        try:
            history2 = self.model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=20,
                callbacks=callbacks,
                initial_epoch=len(history1.history['loss']),
                verbose=1
            )
            
            # Combine histories
            self.history = {
                'loss': history1.history['loss'] + history2.history['loss'],
                'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
                'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            }
        except Exception as e:
            print(f"\n Phase 2 error: {e}")
            self.history = history1.history
        
        print("\n Training completed!")
        
        return self.history
    
    def evaluate(self, test_gen):
        """Evaluate model"""
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Metrics
        test_loss, test_acc = self.model.evaluate(test_gen, verbose=0)
        
        print(f"\n Test Metrics:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Predictions
        y_pred_probs = self.model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes
        
        # Classification report
        print("\n Classification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': y_pred,
            'true_labels': y_true,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, save_path='training_curves_fixed.png'):
        """Plot training curves"""
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(self.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history['loss'], label='Train', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n Training curves saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix_fixed.png'):
        """Plot confusion matrix"""
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Fixed SOTA Model', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Confusion matrix saved to {save_path}")
        plt.show()


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("FIXED SOTA TRAINING PIPELINE")
    print("="*60)
    
    # Configuration
    TRAIN_DIR = r"C:\Users\22206\Desktop\FoodClassification\dataset\train_augmented"
    TEST_DIR = r"C:\Users\22206\Desktop\FoodClassification\dataset\test"
    NUM_CLASSES = 4
    IMG_SIZE = 160  # Smaller than 224
    BATCH_SIZE = 16  # Smaller batch
    EPOCHS = 50
    MODEL_DIR = 'models_sota_fixed'
    RESULTS_DIR = 'results_sota_fixed'
    
    # Initialize
    classifier = FixedSOTAClassifier(
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE
    )
    
    # Build model
    classifier.build_model(learning_rate=0.001)
    
    # Prepare data
    train_gen, test_gen = classifier.prepare_data(TRAIN_DIR, TEST_DIR, BATCH_SIZE)
    
    print(f"\n Dataset Info:")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Test samples: {test_gen.samples}")
    print(f"  Classes: {classifier.class_names}")
    
    # Train
    history = classifier.train(train_gen, test_gen, epochs=EPOCHS, model_dir=MODEL_DIR)
    
    # Evaluate
    results = classifier.evaluate(test_gen)
    
    # Plot results
    classifier.plot_training_history(save_path=os.path.join(RESULTS_DIR, 'training_curves.png'))
    classifier.plot_confusion_matrix(results['confusion_matrix'], 
                                     save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    classifier.model.save(os.path.join(MODEL_DIR, 'final_sota_model.h5'))
    print(f"\n Final model saved to {MODEL_DIR}/final_sota_model.h5")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"\n Final Accuracy: {results['test_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()