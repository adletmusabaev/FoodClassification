"""
Ablation Study: SOTA-модель с и без аугментации
Использует final_sota_model.h5
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
import tensorflow as tf

# Убираем предупреждения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


class AblationStudy:
    def __init__(self, train_dir, test_dir, model_path, num_classes=4, img_size=160):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_path = model_path
        self.num_classes = num_classes
        self.img_size = img_size
        self.results = {}
        self.model = None

    def load_sota_model(self):
        """Загружает твою SOTA-модель"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"\nSOTA-модель не найдена!\n"
                f"   Путь: {self.model_path}\n"
                f"   Сначала запусти: py sota_training_fixed.py\n"
            )
        
        print(f"Загружаю SOTA-модель из: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

        self.model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
        print(f"SOTA-модель успешно загружена!")
        return self.model

    def create_data_generators(self, use_augmentation=True):
        """Генераторы с/без аугментации"""
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.2,
                zoom_range=0.3,
                horizontal_flip=True,
                brightness_range=[0.7, 1.3],
                fill_mode='nearest'
            )
            print("Аугментация: ВКЛЮЧЕНА")
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
            print("Аугментация: ВЫКЛЮЧЕНА")

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=16,
            class_mode='categorical',
            shuffle=True
        )

        test_gen = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=16,
            class_mode='categorical',
            shuffle=False
        )

        return train_gen, test_gen

    def run_experiment(self, experiment_name, use_augmentation=True, epochs=20):
        print(f"\n{'='*60}")
        print(f"ЭКСПЕРИМЕНТ: {experiment_name}")
        print(f"{'='*60}")

        # Используем одну и ту же SOTA-модель
        model = self.load_sota_model()

        train_gen, test_gen = self.create_data_generators(use_augmentation)

        # Early stopping
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
        ]

        print(f"Обучение на {len(train_gen)} батчах...")
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Оценка
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        print(f"Результат: Test Accuracy = {test_acc:.4f}")

        # Per-class accuracy
        y_pred = model.predict(test_gen, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_gen.classes
        class_names = list(train_gen.class_indices.keys())

        class_accuracies = {}
        for i, name in enumerate(class_names):
            mask = (y_true == i)
            if mask.sum() > 0:
                acc = (y_pred_classes[mask] == i).mean()
                class_accuracies[name] = float(acc)

        # Сохраняем результаты
        self.results[experiment_name] = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'training_accuracy': float(max(history.history['accuracy'])),
            'val_accuracy': float(max(history.history['val_accuracy'])),
            'epochs_trained': len(history.history['loss']),
            'class_accuracies': class_accuracies,
            'use_augmentation': use_augmentation
        }

        return history

    def run_all_experiments(self):
        print("\n" + "="*60)
        print("ЗАПУСК АБЛЯЦИИ: SOTA + АУГМЕНТАЦИЯ")
        print("="*60)

        experiments = [
            ("SOTA без аугментации", False),
            ("SOTA с аугментацией", True),
        ]

        for name, use_aug in experiments:
            self.run_experiment(name, use_augmentation=use_aug, epochs=20)

        print("\nВсе эксперименты завершены!")

    def visualize_results(self, save_path='ablation_sota_results.png'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Study: SOTA-модель с/без аугментации', fontsize=16, fontweight='bold')

        exp_names = list(self.results.keys())
        colors = ['#FF6B6B', '#4ECDC4']

        # 1. Общая точность
        accs = [self.results[e]['test_accuracy'] for e in exp_names]
        bars = axes[0,0].bar(exp_names, accs, color=colors, width=0.6)
        axes[0,0].set_ylabel('Test Accuracy')
        axes[0,0].set_title('Общая точность на тесте')
        axes[0,0].set_ylim(0, 1.1)
        for bar, acc in zip(bars, accs):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', fontweight='bold')

        # 2. Train vs Val
        train_acc = [self.results[e]['training_accuracy'] for e in exp_names]
        val_acc = [self.results[e]['val_accuracy'] for e in exp_names]
        x = np.arange(len(exp_names))
        width = 0.35
        axes[0,1].bar(x - width/2, train_acc, width, label='Train', color='skyblue')
        axes[0,1].bar(x + width/2, val_acc, width, label='Val', color='lightcoral')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].set_title('Train vs Validation')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(exp_names, rotation=10)
        axes[0,1].legend()

        # 3. Per-class
        if len(exp_names) >= 2:
            classes = list(self.results[exp_names[0]]['class_accuracies'].keys())
            no_aug = [self.results[exp_names[0]]['class_accuracies'].get(c, 0) for c in classes]
            with_aug = [self.results[exp_names[1]]['class_accuracies'].get(c, 0) for c in classes]
            x = np.arange(len(classes))
            axes[1,0].bar(x - 0.2, no_aug, 0.4, label=exp_names[0], color=colors[0])
            axes[1,0].bar(x + 0.2, with_aug, 0.4, label=exp_names[1], color=colors[1])
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].set_title('По классам')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(classes, rotation=45)
            axes[1,0].legend()

        # 4. Summary
        axes[1,1].axis('off')
        baseline = self.results[exp_names[0]]['test_accuracy']
        improved = self.results[exp_names[1]]['test_accuracy']
        diff = improved - baseline
        summary = f"""
        РЕЗУЛЬТАТЫ АБЛЯЦИИ

        Без аугментации:  {baseline:.3f}
        С аугментацией:   {improved:.3f}

        Разница:          {diff:+.3f}  ({diff/baseline*100:+.1f}%)

        Вывод:
        {'АУГМЕНТАЦИЯ ПОМОГЛА!' if diff > 0.02 else 'Эффект минимален'}
        {'Модель переобучилась' if baseline > 0.95 else ''}
        """
        axes[1,1].text(0.05, 0.7, summary, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow"))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nГрафик сохранён: {save_path}")
        plt.show()

    def save_results(self, output_dir='ablation_sota_results'):
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame([
            {
                'Эксперимент': k,
                'Test Acc': f"{v['test_accuracy']:.4f}",
                'Val Acc': f"{v['val_accuracy']:.4f}",
                'Эпох': v['epochs_trained'],
                'Аугм': '✓' if v['use_augmentation'] else '✗'
            }
            for k, v in self.results.items()
        ])
        df.to_csv(f"{output_dir}/results.csv", index=False)
        with open(f"{output_dir}/full_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены в: {output_dir}/")


def main():
    print("="*60)
    print("АБЛЯЦИЯ: SOTA-МОДЕЛЬ + АУГМЕНТАЦИЯ")
    print("="*60)

    # ПУТИ (измени, если нужно)
    TRAIN_DIR = r"C:\Users\22206\Desktop\FoodClassification\dataset\train"  # Маленький датасет!
    TEST_DIR = r"C:\Users\22206\Desktop\FoodClassification\dataset\test"
    MODEL_PATH = r"C:\Users\22206\Desktop\FoodClassification\scripts\models_sota_fixed\final_sota_model.h5"

    # Создай train_small: по 20-30 изображений на класс
    if not os.path.exists(TRAIN_DIR):
        print(f"ВНИМАНИЕ: Папка не найдена: {TRAIN_DIR}")
        print("Создай её и положи по 20-30 изображений на класс!")
        return

    study = AblationStudy(TRAIN_DIR, TEST_DIR, MODEL_PATH)
    study.run_all_experiments()
    study.visualize_results()
    study.save_results()


if __name__ == "__main__":
    main()