#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;

const size_t blobSize = 600;

const char* classNames[] = { "fingermark" };


Mat object_detection(Mat &frame)
{
	Size frame_size = frame.size();

	String weights = "./model_fpn/frozen_inference_graph.pb";
	String prototxt = "./model_fpn/graph.pbtxt";
	dnn::Net net = cv::dnn::readNetFromTensorflow(weights, prototxt);

	double start = (double)getTickCount();

	cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, Size(blobSize, blobSize));
	
	net.setInput(blob);
	Mat output = net.forward();
	//cout << "output size: " << output.size << endl;

	Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

	float confidenceThreshold = 0.70;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);
		//cout << "conf:" << confidence << endl;

		if (confidence > confidenceThreshold)
		{
			size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

			char conf[20];
			sprintf_s(conf, "%0.2f", confidence);

			Rect object((int)xLeftBottom, (int)yLeftBottom,
				(int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));

			rectangle(frame, object, Scalar(0, 255, 255), 2);
			String label = String(classNames[objectClass]) + ": " + conf;
			int baseLine = 0;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseLine);
			rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
				Size(labelSize.width, labelSize.height + baseLine)),
				Scalar(0, 255, 255), -1);
			putText(frame, label, Point(xLeftBottom, yLeftBottom),
				FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
		}
	}
	double end = (double)getTickCount();
	cout << "use_time :" << (end - start) * 1000.0 / cv::getTickFrequency() << " ms \n";
	return frame;
}

int main() 
{
	Mat frame = cv::imread("E:/Practice/TensorFlow/DataSet/fingermark/VOC2012/JPEGImages/1.jpg");
	Mat result = object_detection(frame);

	resize(result, result, Size(result.cols / 2, result.rows / 2), 0, 0);
	imshow("result", result);
	waitKey(0);

}