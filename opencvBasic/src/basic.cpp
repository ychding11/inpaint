#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
	if (argc != 2)
    {
		std::cout << "Usage:\n"  << argv[0] << " image_file_name" << std::endl;
		return 1;
	}

    // How to check read image ok?
	cv::Mat inputImage = cv::imread(argv[1]);
    printf("- Image size w=%d, h=%d.\n", inputImage.cols, inputImage.rows);
	int h = inputImage.rows;
	int w = inputImage.cols;

	int kw = 31;
	int kh = 21;

    cv::Mat patch1 (inputImage, cv::Rect(0, 0, kw, kh));
	cv::Mat patch2 (inputImage, cv::Rect(w - kw, h - kh, kw, kh));
	
    printf("- patch1 size w=%d, h=%d.\n", patch1.cols, patch1.rows);
    printf("- patch2 size w=%d, h=%d.\n", patch2.cols, patch2.rows);

	//std::cout << "\n inputImage = \n \t" <<  inputImage;
    //std::cout << "\n patch1 =\n\t " << patch1 ;
    //std::cout << "\n patch2 =\n\t " << patch2 ;

	cv::namedWindow("Input Image");
    cv::imshow("Input Image", inputImage);
    cv::waitKey(); // wait for key stroke forever.
	cv::imshow("Patch1", patch1);
	cv::imshow("Patch2", patch2);
	cv::waitKey(); // wait for key stroke forever.

	for (int i = 0; i < kh; ++i)
	{
		for (int j = 0; j < kw; ++j)
		{
			//patch1.at<cv::Scalar>(i, j) = cv::Scalar(255, 0, 0);
			//patch2.at<cv::Scalar>(i, j) = cv::Scalar(0, 0, 255);
			patch1.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
			patch2.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
		}
	}

    cv::imshow("Input Image", inputImage);
	cv::waitKey(); // wait for key stroke forever.
	return 0;
}
