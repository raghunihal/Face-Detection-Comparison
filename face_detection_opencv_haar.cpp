#include <opencv2/opencv.hpp>

int main(int argc, char** argv) 
{
    cv::CommandLineParser parser(argc, argv);
    std::string image_file = parser.get<std::string>("-i");
    std::string video_file = parser.get<std::string>("-v");
    
    cv::CascadeClassifier face_detector;
    face_detector.load("haarcascade_frontalface_default.xml");
    
    if (!image_file.empty()) 
    {
        cv::Mat img = cv::imread(image_file);
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_detector.detectMultiScale(gray, faces, 1.3, 5);
        for (const auto& rect : faces) 
        {
            cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("face detection - opencv haar", img);
        cv::waitKey(0);
        cv::destroyAllWindows();
        return 0;
    }
    
    cv::VideoCapture webcam;
    if (!video_file.empty()) 
    {
        webcam.open(video_file);
    } 
	else 
	{
        webcam.open(0);
    }
    if (!webcam.isOpened()) 
    {
        std::cout << "Could not open webcam" << std::endl;
        return -1;
    }
    
    while (webcam.isOpened()) 
    {
        cv::Mat frame;
        bool status = webcam.read(frame);
        if (!status) 
        {
            std::cout << "Could not read frame" << std::endl;
            return -1;
        }
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        std::vector<cv::Rect> faces;
        face_detector.detectMultiScale(gray, faces, 1.3, 5);
        
        for (const auto& rect : faces) 
        {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        }
        
        cv::imshow("face detection - opencv haar", frame);
        
        if (cv::waitKey(1) & 0XFF == 'q') 
        {
            break;
        }
    }
    
    webcam.release();
    cv::destroyAllWindows();
    return 0;
}


