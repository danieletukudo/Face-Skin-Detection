import sys  # Import the sys module for system-specific parameters and functions
from typing import Tuple, List, Union, Any  # Import type hints for better code readability and type checking
import cv2  # Import OpenCV for image processing
from numpy import ndarray, dtype, generic  # Import specific types from numpy for array handling
from ultralytics import YOLO  # Import the YOLO object detection model from the ultralytics library
import uuid  # Import the uuid module to generate unique identifiers


class recognize_image:  # Define a class for image recognition

    def __init__(self) -> None:  # Class constructor
        """Initialize the recognize_image class."""
        pass  # No initialization required for now

    def setup(self, model_path: str = 'deploy.prototxt',
              weights_path: str = 'res10_300x300_ssd_iter_140000.caffemodel') -> tuple[str, str]:
        """
        Setup method to return model and weights paths.

        Args:
            model_path (str): Path to the model file.
            weights_path (str): Path to the weights file.

        Returns:
            tuple[str, str]: A tuple containing the model path and weights path.
        """
        return model_path, weights_path  # Return the default paths for model and weights

    def YOLO_Model(self, image_path: str) -> tuple[list, Union[ndarray, Any]]:
        """
        Apply YOLO model on an image and return results.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple[list, ndarray]: A tuple containing the detection results and the image.
        """
        image = cv2.imread(image_path)  # Read the image from the given path
        model = YOLO("best.pt")  # Load the YOLO model with specified weights
        result = model(image_path, conf=0.1)  # Apply the model on the image with a confidence threshold
        return result, image  # Return the detection results and the image

    def detect_face(self, image_path: str) -> bool:
        """
        Detect faces in an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            bool: True if a face is detected, False otherwise.
        """
        image = cv2.imread(image_path)  # Read the image from the given path
        models = self.setup()  # Get the model and weights paths
        model_path = models[0]  # Extract the model path
        weights_path = models[1]  # Extract the weights path
        net = cv2.dnn.readNetFromCaffe(model_path, weights_path)  # Load the model using OpenCV's DNN module
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))  # Create a blob from the image
        net.setInput(blob)  # Set the blob as input to the network
        detections = net.forward()  # Perform forward pass to get detections

        # Loop over the detections and draw bounding boxes with confidence scores
        for i in range(detections.shape[2]):  # Iterate over each detection
            confidence = detections[0, 0, i, 2]  # Confidence score of the detection
            if confidence > 0.5:  # Check if confidence is above the threshold
                return True  # If confidence is high, face is detected
            else:
                return False  # If confidence is low, face is not detected

    def check_for_skin(self, image_path: str) -> str:
        """
        Check for skin (faces) in an image and process detections.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Unique image name if a face is detected, "face_not_found" otherwise.

        """
        detect_face = self.detect_face(image_path)  # Detect face in the image
        model, image = self.YOLO_Model(image_path)  # Get YOLO model results and the image

        if detect_face == True:  # If a face is detected
            print("Face seen")  # Print a message
            pass  # Continue processing

        else:  # If no face is detected
            return "face_not_found"  # Return a message indicating no face found

        for i, r in enumerate(model):  # Iterate over the model results
            detections = r.boxes.data.tolist()  # Get detection data as a list
            names = r.names  # Get class names
            classes = r.boxes.cls.tolist()  # Get class labels

            for labels, detection in zip(classes, detections):  # Iterate over detections and labels
                label = names[labels]  # Get the label name
                x1, y1, x2, y2, score, _ = detection  # Extract bounding box coordinates and score

                cv2.putText(image, str(label), (int(x1), int(y1 - 2)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                            (0, 0, 255), 1, cv2.LINE_AA)  # Draw label on the image
                cv2.putText(image, str(f"{score:.2f}"), (int(x1), int(y1 - 12)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                            (255, 0, 0), 1, cv2.LINE_AA)  # Draw score on the image
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2,
                              cv2.LINE_AA)  # Draw bounding box
        image_name = f"{uuid.uuid4()}"  # Generate a unique image name
        cv2.imwrite(f"static1/{image_name}.jpg", image)  # Save the processed image

        return image_name  # Return the unique image name


if __name__ == "__main__":  # Main block to run the script
    de = recognize_image()  # Create an instance of the recognize_image class
    result = de.check_for_skin(
        "img_1.png")  # Check for skin (face) in the given image

    print(result)  # Print the result
