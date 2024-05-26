from flask import Flask, request, jsonify, url_for, Blueprint
from main import recognize_image
import os
import uuid

app = Flask(__name__, static_url_path='/')
bp1 = Blueprint('bp1', __name__, static_folder='static1')
UPLOAD_FOLDER = "./static1"  # Directory to save uploaded images

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configure Flask to use the upload folder

# Create a Blueprint for serving static files

# Register the blueprint with the Flask app
app.register_blueprint(bp1)

# Allowed file extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.

    Args:
        filename (str): Name of the file to check.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Endpoint to process the uploaded image and detect faces
@app.route('/search', methods=['POST'])
def run_model():
    # try:
        input_image = request.files['file']  # Get the uploaded file
        if input_image.filename == '':  # Check if no file was uploaded
            app.logger.warning('No files inserted yet')
            return jsonify({'response': 'No files inserted yet'})

        if input_image and allowed_file(input_image.filename):  # Check if the file is allowed
            img_path = os.path.join(UPLOAD_FOLDER, f'{uuid.uuid4()}.jpg')  # Generate a unique path for the image

            input_image.save(img_path)  # Save the uploaded image

            model = recognize_image()  # Create an instance of the recognize_image class
            image_name = model.check_for_skin(image_path=img_path)  # Process the image

            if image_name == "face_not_found":  # Check if a face was not detected
                return jsonify(
                    {'response': 'No human face detected, insert a clear picture of a person looking at the camera'})

            else:
                # Generate a URL for the processed image
                img_url = url_for('bp1.static', filename=f'{image_name}.jpg',_external=True)
                app.logger.info(f'Generated URL: {img_url}')  # Debug: Log the generated URL

                os.remove(img_path)  # Remove the original uploaded image
                return jsonify({'image_url': img_url}), 202  # Return the URL of the processed image

        else:
            return jsonify({'response': 'Please upload the right picture file with an extension of png, jpg, or jpeg'})
    # except Exception as e:
    #     app.logger.error(f'Error: {str(e)}')  # Debug: Log any exceptions
    #     return jsonify({'response': 'invalid input'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7017, debug=True)  # Run the Flask app
