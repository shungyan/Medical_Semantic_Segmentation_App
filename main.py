from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

my_model = load_model('brats_3d_90epochs.hdf5', compile=False)

matplotlib.use('agg')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_num = 21
        test_img = np.load("image_9.npy")
        test_mask = np.load("mask_9.npy")
        test_mask_argmax = np.argmax(test_mask, axis=3)
        test_img_input = np.expand_dims(test_img, axis=0)
        test_prediction = my_model.predict(test_img_input)
        test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

        n_slice = int(request.form['slice_number'])

        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
        plt.subplot(232)
        plt.title('Testing Label')
        plt.imshow(test_mask_argmax[:, :, n_slice])
        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(test_prediction_argmax[:, :, n_slice])

        # Save the plot to a BytesIO object
        img_bytes_io = BytesIO()
        plt.savefig(img_bytes_io, format='png')
        img_bytes_io.seek(0)

        img_base64 = base64.b64encode(img_bytes_io.read()).decode('utf-8')
        plt.close()

        return render_template('index.html', img_base64=img_base64)
    else:
        return render_template('index.html', img_base64=None)

if __name__ == '__main__':
    app.run(debug=True)
