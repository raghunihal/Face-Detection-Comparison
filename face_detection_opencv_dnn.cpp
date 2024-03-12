#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) 
{
    cv::CommandLineParser parser(argc, argv,
        "{prototxt| |path to prototxt file}"
        "{model| |path to pre-trained caffemodel file}"
        "{threshold|0.5|probability threshold to ignore false detections}");

    std::string prototxtPath = parser.get<std::string>("prototxt");
    std::string modelPath = parser.get<std::string>("model");
    float threshold = parser.get<float>("threshold");

    cv::dnn::Net net = cv::dnn::readNetFromCaffe(prototxtPath, modelPath);
    cv::VideoCapture webcam(0);
    if (!webcam.isOpened()) 
	{
        std::cout << "Error opening webcam" << std::endl;
        return -1;
    }

    while (webcam.isOpened()) 
	{
        cv::Mat frame;
        bool status = webcam.read(frame);
        if (!status) 
		{
            std::cout << "Error reading frame" << std::endl;
            break;
        }

        int h = frame.rows;
        int w = frame.cols;
        cv::Mat blob = cv::dnn::blobFromImage(cv::resize(frame, cv::Size(300, 300)), 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
        net.setInput(blob);
        cv::Mat faces = net.forward();

        for (int i = 0; i < faces.size[2]; i++) 
		{
            float confidence = faces.at<float>(0, 0, i, 2);
            if (confidence < threshold) 
			{
                continue;
            }

            cv::Rect box;
            box.x = faces.at<float>(0, 0, i, 3) * w;
            box.y = faces.at<float>(0, 0, i, 4) * h;
            box.width = (faces.at<float>(0, 0, i, 5) - faces.at<float>(0, 0, i, 3)) * w;
            box.height = (faces.at<float>(0, 0, i, 6) - faces.at<float>(0, 0, i, 4)) * h;

            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

            std::string text = "face " + std::to_string(confidence * 100) + "%";
            cv::putText(frame, text, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("output", frame);
        if (cv::waitKey(1) == 'q') 
		{
            break;
        }
    }

    webcam.release();
    cv::destroyAllWindows();

    return 0;
}


