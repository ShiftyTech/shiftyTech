import webbrowser
import pyautogui
import time
import cv2
import tensorflow as tf

from compiler.mapping import pose_mapping
from move_net.estimation import input_size, movenet
from move_net.helper import draw_prediction_on_image
from move_net.image_utils import get_vector_from_frame
from move_net.search_milvus import search_for_pose

import configparser

from pymilvus import connections, utility
from pymilvus import Collection


def main():
    # Load milvus configs
    cfp = configparser.RawConfigParser()
    cfp.read('config_serverless.ini')

    operations = []
    to_skip: bool = False

    # Connect to milvus
    milvus_uri = cfp.get('example', 'uri')
    token = cfp.get('example', 'token')

    connections.connect("default",
                        uri=milvus_uri,
                        token=token)
    print(f"Connecting to DB: {milvus_uri}")

    # Check if the collection exists
    collection_name = "shifty_collection"
    check_collection = utility.has_collection(collection_name)
    print("Successfully connected to collection!")

    shifty_collection = Collection(collection_name)

    url = "https://replit.com/join/fmrnbraiil-rudrakshmonga1"
    webbrowser.open(url)

    time.sleep(5)

    vid = cv2.VideoCapture(1)
    frame_interval = 5  # Set the frame capture interval in seconds
    start_time = time.time()

    while True:
        current_time = time.time()
        time_left = frame_interval - (current_time - start_time)

        ret, frame = vid.read()

        # Create an image with the same dimensions as the frame for the text overlay
        overlay = frame.copy()

        # Add text indicating the time left until the next frame capture
        text = f"Next frame in {int(time_left) + 1}"
        cv2.putText(overlay, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)

        # Blend the overlay text onto the frame
        alpha = 0.6  # Adjust the transparency of the text
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Run MoveNet inference on the current frame
        input_image = tf.image.resize(frame, [input_size, input_size])
        input_image = tf.expand_dims(input_image, axis=0)
        keypoints_with_scores = movenet(input_image)
        # Draw the keypoints onto the frame
        frame = draw_prediction_on_image(frame, keypoints_with_scores)

        cv2.imshow('frame', frame)

        if time_left <= 0:
            # Capture a frame and reset the start time for the next frame capture
            ret, frame = vid.read()

            vector_representation = get_vector_from_frame(frame)
            closest_pose = search_for_pose(vector_representation, shifty_collection)

            closest_pose = closest_pose.replace('.jpg', '')

            operations.append(closest_pose)

            record = closest_pose

            if to_skip:
                try:
                    record = pose_mapping['not ' + record]
                except:
                    record = pose_mapping['not'] + pose_mapping[record]
                to_skip = False
            else:
                record = pose_mapping[record]

            if operations[-1] == 'not':
                to_skip = True
                # Add a red border around the frame
                border_thickness = 10
                print("Not hit")
                frame[:border_thickness, :] = [0, 0, 255]  # Top border
                frame[-border_thickness:, :] = [0, 0, 255]  # Bottom border
                frame[:, :border_thickness] = [0, 0, 255]  # Left border
                frame[:, -border_thickness:] = [0, 0, 255]  # Right border

                # Display the frame with the red border (Optional)
                cv2.imshow('frame', frame)
                cv2.waitKey(2)  # Display for a short time. Adjust as needed.

            if not to_skip:
                pyautogui.write(record)

            start_time = time.time()

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
