import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from PIL import Image

class CancerDetectionPipeline:
    def __init__(self, model_save_path='my_model3.h5', num_classes=4):
        self.model_save_path = model_save_path
        self.num_classes = num_classes
        self.class_labels = ["Benign", "InSitu", "Invasive", "Normal"]
        
        # Modern TensorFlow configurations
        K.set_image_data_format('channels_last')

    def extract_patches(self, img, need_rotations=False):
        """
        Extracts a grid of 12 patches (512x512) from a high-res biopsy image.
        Optionally applies 180-degree rotations for basic data augmentation.
        """
        img_array = np.asarray(img, dtype=np.uint8)
        patches = []
        
        # 3x4 grid cropping strategy
        for i in range(3):
            for j in range(4):
                crop = img_array[512*i : 512*(i+1), 512*j : 512*(j+1), :]
                patches.append(crop)
                if need_rotations:
                    patches.append(np.rot90(crop, k=2)) # Rotates 180 degrees cleanly
        return patches

    def _get_one_hot_label(self, folder_name):
        """Maps folder category names to a structured one-hot encoded vector."""
        mapping = {'b': [1,0,0,0], 'is': [0,1,0,0], 'iv': [0,0,1,0]}
        return mapping.get(folder_name, [0,0,0,1])

    def load_and_preprocess_dataset(self, data_directory, num_test_samples=2):
        """Loads raw images from folders, processes patches, and returns splits."""
        x_data, y_data = [], []
        image_count = 0

        for foldname in os.listdir(data_directory):
            fold_path = os.path.join(data_directory, foldname)
            if not os.path.isdir(fold_path):
                continue
                
            for filename in os.listdir(fold_path):
                img_path = os.path.join(fold_path, filename)
                try:
                    with Image.open(img_path) as img:
                        patches = self.extract_patches(img)
                    image_count += 1
                    
                    for patch in patches:
                        # Normalize pixel values directly to float32
                        x_data.append(patch.astype(np.float32) / 255.0)
                        y_data.append(self._get_one_hot_label(foldname))
                except Exception as e:
                    print(f"Skipping corrupt image {filename}: {e}")

        x_data, y_data = np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.int8)
        
        test_size = num_test_samples * 12
        train_size = len(x_data) - test_size

        return (
            x_data[:train_size], y_data[:train_size],
            x_data[train_size:], y_data[train_size:]
        )

    def build_cnn_architecture(self, input_shape=(512, 512, 3)):
        """Defines the custom ConvNet layers with systematic downsampling."""
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((3, 3), strides=3)(x)
        
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=2)(x)
        
        x = layers.Conv2D(64, (2, 2), activation='relu')(x)
        x = layers.ZeroPadding2D(padding=(2, 2))(x)
        x = layers.MaxPooling2D((2, 2), strides=2)(x)
        
        x = layers.Conv2D(64, (2, 2), activation='relu')(x)
        x = layers.ZeroPadding2D(padding=(2, 2))(x)
        x = layers.MaxPooling2D((3, 3), strides=3)(x)
        
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return models.Model(inputs=inputs, outputs=outputs, name='Cancer_Detection_CNN')

    def execute_training_loop(self, x_train, y_train, x_test, y_test, batch_size=16, epochs=10):
        """Compiles, manages execution loops, and stores the optimized model states."""
        model = self.build_cnn_architecture(input_shape=x_train.shape[1:])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        if os.path.exists(self.model_save_path):
            print(f"Loading weights from existing model checkpoint: {self.model_save_path}")
            model = models.load_model(self.model_save_path)

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
        model.save(self.model_save_path)
        return model

    def infer_single_image(self, img_path):
        """Runs patch ensemble inference across an un-cropped tissue image."""
        if Skinner := not os.path.exists(self.model_save_path):
            raise FileNotFoundError("Trained model checkpoint binary file not found.")
            
        model = models.load_model(self.model_save_path)
        with Image.open(img_path) as img:
            normalized_crops = np.array(self.extract_patches(img), dtype=np.float32) / 255.0
            
        accumulated_probabilities = np.zeros(self.num_classes)
        
        for patch in normalized_crops:
            input_tensor = np.expand_dims(patch, axis=0)
            predictions = model.predict(input_tensor, verbose=0)
            accumulated_probabilities += predictions[0]
            
        mean_probabilities = accumulated_probabilities / len(normalized_crops)
        predicted_class_idx = np.argmax(mean_probabilities)
        
        print("\n=== Diagnosis Report ===")
        for label, prob in zip(self.class_labels, mean_probabilities):
            print(f"{label}: {prob*100:.2f}%")
        print(f"\nFinal Aggregated Prediction: {self.class_labels[predicted_class_idx].upper()}")


if __name__ == "__main__":
    # Example clean pipeline initialization usage
    pipeline = CancerDetectionPipeline()
    # To execute training, pass local parameter directories cleanly:
    # x_train, y_train, x_test, y_test = pipeline.load_and_preprocess_dataset("./Samples")
    # pipeline.execute_training_loop(x_train, y_train, x_test, y_test)
