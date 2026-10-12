import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.highgui.HighGui;

import java.awt.Toolkit;

public class DriverDrowsinessDetection {

    private static final long WARNING_TIME_MS = 1500;

    public static void main(String[] args) {

        // Load OpenCV native libraries
        OpenCV.loadShared();

        System.out.println("Driver Drowsiness Detection");
        System.out.println("OpenCV version: " + Core.VERSION);

        // Load face and eye detection files
        CascadeClassifier faceDetector =
                new CascadeClassifier(
                    "haarcascade/haarcascade_frontalface_default.xml"
                );

        CascadeClassifier eyeDetector =
                new CascadeClassifier(
                    "haarcascade/haarcascade_eye_tree_eyeglasses.xml"
                );

        if (faceDetector.empty() || eyeDetector.empty()) {
            System.out.println("ERROR: Could not load XML files.");
            System.out.println(
                "Check the haarcascade folder and file names."
            );
            return;
        }

        // Open the laptop's built-in camera
        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println(
                "Camera 0 unavailable. Trying camera 1..."
            );
            camera.release();
            camera = new VideoCapture(1);
        }

        if (!camera.isOpened()) {
            System.out.println("ERROR: Could not access the camera.");
            System.out.println(
                "Allow camera permission and close other camera apps."
            );
            camera.release();
            return;
        }

        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);

        System.out.println("Laptop camera connected!");
        System.out.println("Press Q to quit.");

        Mat frame = new Mat();
        Mat gray = new Mat();

        long eyesMissingSince = 0;
        long lastBeepTime = 0;

        try {
            while (camera.read(frame)) {

                if (frame.empty()) {
                    continue;
                }

                Imgproc.cvtColor(
                    frame, gray, Imgproc.COLOR_BGR2GRAY
                );
                Imgproc.equalizeHist(gray, gray);

                MatOfRect faces = new MatOfRect();

                faceDetector.detectMultiScale(
                    gray,
                    faces,
                    1.1,
                    5,
                    0,
                    new Size(100, 100),
                    new Size()
                );

                Rect[] faceArray = faces.toArray();

                String status = "FACE NOT DETECTED";
                Scalar statusColor = new Scalar(0, 165, 255);

                if (faceArray.length == 0) {

                    // Missing face is not proof of closed eyes.
                    eyesMissingSince = 0;

                } else {

                    // Select the largest detected face.
                    Rect face = faceArray[0];

                    for (Rect candidate : faceArray) {
                        if (candidate.area() > face.area()) {
                            face = candidate;
                        }
                    }

                    Imgproc.rectangle(
                        frame,
                        new Point(face.x, face.y),
                        new Point(
                            face.x + face.width,
                            face.y + face.height
                        ),
                        new Scalar(255, 180, 0),
                        2
                    );

                    // Search for eyes in the upper half of the face.
                    int upperHeight = Math.max(1, face.height / 2);

                    Rect upperFace = new Rect(
                        face.x,
                        face.y,
                        face.width,
                        upperHeight
                    );

                    Mat eyeRegion = gray.submat(upperFace);
                    MatOfRect eyes = new MatOfRect();

                    eyeDetector.detectMultiScale(
                        eyeRegion,
                        eyes,
                        1.1,
                        5,
                        0,
                        new Size(15, 10),
                        new Size()
                    );

                    Rect[] eyeArray = eyes.toArray();

                    for (Rect eye : eyeArray) {
                        Imgproc.rectangle(
                            frame,
                            new Point(
                                upperFace.x + eye.x,
                                upperFace.y + eye.y
                            ),
                            new Point(
                                upperFace.x + eye.x + eye.width,
                                upperFace.y + eye.y + eye.height
                            ),
                            new Scalar(0, 255, 0),
                            2
                        );
                    }

                    long now = System.currentTimeMillis();

                    if (eyeArray.length == 0) {

                        if (eyesMissingSince == 0) {
                            eyesMissingSince = now;
                        }

                        long missingTime = now - eyesMissingSince;

                        if (missingTime >= WARNING_TIME_MS) {
                            status = "WARNING: POSSIBLE DROWSINESS!";
                            statusColor = new Scalar(0, 0, 255);

                            // Beep at most once every second.
                            if (now - lastBeepTime >= 1000) {
                                try {
                                    Toolkit.getDefaultToolkit().beep();
                                } catch (Exception e) {
                                    System.out.println(
                                        "Could not play warning sound."
                                    );
                                }
                                lastBeepTime = now;
                            }

                        } else {
                            status = "EYES NOT DETECTED - CHECK POSITION";
                            statusColor = new Scalar(0, 165, 255);
                        }

                    } else {
                        eyesMissingSince = 0;
                        status = "EYES DETECTED - MONITORING";
                        statusColor = new Scalar(0, 200, 0);
                    }

                    eyes.release();
                    eyeRegion.release();
                }

                Imgproc.putText(
                    frame,
                    status,
                    new Point(15, 30),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    statusColor,
                    2
                );

                Imgproc.putText(
                    frame,
                    "Press Q to quit",
                    new Point(15, frame.rows() - 15),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    new Scalar(255, 255, 255),
                    1
                );

                HighGui.imshow(
                    "Laptop Driver Drowsiness Detection",
                    frame
                );

                int key = HighGui.waitKey(1);

                if (key == 'q' || key == 'Q' || key == 27) {
                    break;
                }

                faces.release();
            }

        } finally {
            camera.release();
            frame.release();
            gray.release();
            HighGui.destroyAllWindows();
        }

        System.out.println("Detection stopped.");
    }
}