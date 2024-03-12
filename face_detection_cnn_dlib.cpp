#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/cnn_face_detector.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) 
{
    cv::CommandLineParser parser(argc, argv,
                                 "{i|image| |path to image file}"
                                 "{w|weights|./mmod_human_face_detector.dat|path to weights file}"
                                 );
    
    std::string imagePath = parser.get<std::string>("image");
    std::string weightsPath = parser.get<std::string>("weights");
    
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) 
    {
        std::cout << "Could not read input image" << std::endl;
        return 0;
    }
    
    dlib::frontal_face_detector hog_face_detector = dlib::get_frontal_face_detector();
    
    dlib::cnn_face_detection_model_v1 cnn_face_detector;
    dlib::deserialize(weightsPath) >> cnn_face_detector;
    
    double start = cv::getTickCount();
    
    std::vector<dlib::rectangle> faces_hog = hog_face_detector(dlib::cv_image<dlib::bgr_pixel>(image));
    double end = cv::getTickCount();
    std::cout << "Execution Time (in seconds) :" << std::endl;
    std::cout << "HOG : " << (end - start) / cv::getTickFrequency() << std::endl;
    
    for (const auto& face : faces_hog) 
    {
        int x = face.left();
        int y = face.top();
        int w = face.right() - x;
        int h = face.bottom() - y;
        
        cv::rectangle(image, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 2);
    }
    
    start = cv::getTickCount();
    
    std::vector<dlib::mmod_rect> faces_cnn = cnn_face_detector(dlib::cv_image<dlib::bgr_pixel>(image));
    end = cv::getTickCount();
    std::cout << "CNN : " << (end - start) / cv::getTickFrequency() << std::endl;
    
    for (const auto& face : faces_cnn) 
    {
        int x = face.rect.left();
        int y = face.rect.top();
        int w = face.rect.right() - x;
        int h = face.rect.bottom() - y;
        
        cv::rectangle(image, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 0, 255), 2);
    }
    
    int img_height = image.rows;
    int img_width = image.cols;
    cv::putText(image, "HOG", cv::Point(img_width - 50, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,cv::Scalar(0, 255, 0), 2);
    cv::putText(image, "CNN", cv::Point(img_width - 50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,cv::Scalar(0, 0, 255), 2);
    
    cv::imshow("face detection with dlib", image);
    cv::waitKey();    
    cv::imwrite("cnn_face_detection.png", image);    
    cv::destroyAllWindows();
    
    return 0;
}


