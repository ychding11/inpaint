#include <iostream>
//#include <experimental/random>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>

static cv::Mat fetchPatch(const cv::Mat & image, int x, int y, int k)
{
    return cv::Mat(image, cv::Rect(x, y, k, k));
}

static void putPatch(cv::Mat & image, int x, int y, const cv::Mat & patch)
{
	assert(patch.cols == patch.rows);
	int k = patch.cols;
    cv::Mat dstPatch(image, cv::Rect(x, y, k, k));
	for (int i = 0; i < k; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			dstPatch.at<cv::Vec3b>(i, j) = patch.at<cv::Vec3b>(i, j);
		}
	}
}

static int randint(int low, int up)
{
	assert(low < up);
	//return std::experimental::randint(low, up);
	int ret = low + (rand() % static_cast<int>(up - low + 1));
	return ret;
}

cv::Mat naiveSynthesis(const cv::Mat & image, int k)
{
	int w = image.cols * 2;
	int h = image.rows * 2;
	srand(time(NULL));

	cv::Mat ret(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i <= h - k; i += k)
	{
		for (int j = 0; j <= w - k; j += k)
		{
			int x = randint(0, image.cols - k);
			int y = randint(0, image.rows - k);
			putPatch(ret, j, i, fetchPatch(image, x, y, k));
		}
	}
	return ret;
}

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
	int kh = 11;

    cv::Mat patch1 (inputImage, cv::Rect(0, 0, kw, kh));
	cv::Mat patch2 (inputImage, cv::Rect(w - kw, h - kh, kw, kh));
	
    printf("- patch1 size w=%d, h=%d.\n", patch1.cols, patch1.rows);
    printf("- patch2 size w=%d, h=%d.\n", patch2.cols, patch2.rows);

	cv::Mat dstM(cv::Size(inputImage.cols * 2, inputImage.rows), inputImage.type(), cv::Scalar::all(0));
    cv::Mat roiM (dstM, cv::Rect(0, 0, w, h));

	//std::cout << "\n inputImage = \n \t" <<  inputImage;
    //std::cout << "\n patch1 =\n\t " << patch1 ;
    //std::cout << "\n patch2 =\n\t " << patch2 ;

	cv::namedWindow("Input Image");
    cv::imshow("Input Image", inputImage);
	//cv::imshow("Patch1", patch1);
	//cv::imshow("Patch2", patch2);
    //cv::imshow("Synthesis Image", naiveSynthesis(inputImage, 7));
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
