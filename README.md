# cv2WebApp
Basic Flask app for image capture from device camera

Intended use : Interface for ASL alphabet recognition

To use install OpenCV and MediaPipeline and the "off the shelf hand detection model"
 - The camera will turn on when ran ('python .\camera_App.py')
 - Place your hand in view and the MediaPipeline HandLandmark model will detect your had
 - Based on these coordinates the image will be cropped and sent to the prediction network
 - The prediction from the image given will then be displayed
