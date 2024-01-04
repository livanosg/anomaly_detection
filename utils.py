import cv2
import numpy as np


def inspect_data(dataset, model, threshold, conf):
    """
    Allows interactive inspection of images, either with or without a trained model.

    Note:
        This function displays images interactively and accepts user input for inspection.
    """
    paused = False
    window_name = "Inspection tool"
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    for input_image, label in dataset.unbatch().batch(1):
        window_title = " ".join([window_name, "(Paused)" if paused else len("(Paused)") * " "])
        y_prob = None
        y_pred = None
        text = ""
        if conf.model_type == "classifier":
            text = "Probability"
            y_prob = model.predict(input_image, verbose=0).squeeze()
            y_prob = y_prob[..., conf.pos_label]
            y_pred = np.greater_equal(y_prob, threshold).astype(int)
        elif conf.model_type == "autoencoder":
            text = "Reconstruction loss"
            recon_error = model.evaluate(input_image, input_image, verbose=0).squeeze()
            y_prob = recon_error
            y_pred = np.greater_equal(y_prob, threshold).astype(int)

        window_title = " ".join([window_title,
                                 f"Threshold: {np.round(threshold, 2)}",
                                 f"{text}: {np.round(y_prob, 3)}",
                                 f"Prediction: {conf.class_names[y_pred]}"])

        if y_pred == 1:
            color = (0, 0, 255)
        elif y_pred == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        image = input_image.numpy() * 255
        image = image.squeeze().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(img=image, text=conf.class_names[y_pred].capitalize(), org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=1)
        cv2.setWindowTitle(window_name, window_title)
        cv2.imshow(window_name, image)
        key = cv2.waitKey(10)
        if key == ord('q'):  # Quit
            break
        if key == ord('p'):  # Pause
            cv2.waitKey(0)
    cv2.destroyAllWindows()
