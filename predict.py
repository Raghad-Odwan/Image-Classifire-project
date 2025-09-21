import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


def process_image(image):
    """ Process image to fit the model requirements """
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image.numpy()


def predict(image_path, model, top_k=5):
    """ Predict the class of an image using a trained model. """
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(processed_image)
    top_values, top_indices = tf.nn.top_k(predictions, k=top_k)

    return top_values.numpy()[0], top_indices.numpy()[0]


def load_class_names(filepath):
    with open(filepath, 'r') as f:
        class_names = json.load(f)
    return class_names


def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model.')
    parser.add_argument('image_path', help='Path to input image.')
    parser.add_argument('model_path', help='Path to saved model (h5 or keras).')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', default=None, help='Path to a JSON file mapping labels to flower names.')

    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model_path,
                                       custom_objects={'KerasLayer': hub.KerasLayer})

    # Predict
    probs, classes = predict(args.image_path, model, args.top_k)

    # Map to class names if provided
    if args.category_names:
        class_names = load_class_names(args.category_names)
        classes = [class_names[str(c+1)] for c in classes]  # +1 because dataset labels start from 1

    # Print results
    for i in range(len(probs)):
        print(f"{classes[i]}: {probs[i]:.4f}")


if __name__ == '__main__':
    main()