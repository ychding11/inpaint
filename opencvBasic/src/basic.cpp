#include <iostream>
//#include <experimental/random>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <vector>
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

void displayColorSpace(const cv::Mat3b & image)
{
	int w = image.cols;
	int h = image.rows;
	cv::Mat3b plane1(h, w * 2);

	printf("- image data type: %d\n", image.type());
	cv::Mat3b YcbcrImage; image.copyTo(YcbcrImage);

	cv::cvtColor(YcbcrImage, YcbcrImage, CV_BGR2YCrCb);
	printf("- res data type: %d\n", YcbcrImage.type());

    //cv::imshow("src colorspace", image);
    //cv::imshow("dst colorspace", YcbcrImage);
	
	image.copyTo( plane1(cv::Rect(0,0, w, h)) );
	YcbcrImage.copyTo( plane1(cv::Rect(w,0, w, h)) );

	std::vector<cv::Mat1b> chns;
	cv::split(YcbcrImage, chns);
	cv::Mat1b plane2(h, w * 3);
	for (int i = 0; i < chns.size(); ++i)
	{
		printf("- chns[%d]= %dx%d, type=%d\n", i, chns[i].cols, chns[i].rows, chns[i].type());
		chns[i].copyTo( plane2(cv::Rect(w * i,0, w, h)) );
	}

    cv::imshow("multi-channels", plane1);
    cv::imshow("single-channels", plane2);
	cv::waitKey(); // wait for key stroke forever.
}

int main(int argc, char **argv)
{
	if (argc != 2)
    {
		std::cout << "Usage:\n"  << argv[0] << " image_file_name" << std::endl;
		return 1;
	}

    // How to check read image ok?
	cv::Mat3b inputImage = cv::imread(argv[1]);
    printf("- Image size w=%d, h=%d.\n", inputImage.cols, inputImage.rows);
	int h = inputImage.rows;
	int w = inputImage.cols;

	displayColorSpace(inputImage); exit(1);
	int kw = 3;
	int kh = 3;

    cv::Mat3b patch1 (inputImage, cv::Rect(0, 0, kw, kh));
	cv::Mat3b patch2 (inputImage, cv::Rect(w - kw, h - kh, kw, kh));
	
    printf("- patch1 size w=%d, h=%d.\n", patch1.cols, patch1.rows);
    printf("- patch2 size w=%d, h=%d.\n", patch2.cols, patch2.rows);

	cv::Mat3b dstM(cv::Size(inputImage.cols * 2, inputImage.rows), inputImage.type());
    cv::Mat3b roiM (dstM, cv::Rect(0, 0, w, h));
	inputImage.copyTo(roiM);

	for (int i = 0; i < kh; ++i)
	{
		for (int j = 0; j < kw; ++j)
		{
			patch1.at<cv::Vec3b>(i, j) = cv::Vec3b(1, 0, 2);
			patch2.at<cv::Vec3b>(i, j) = cv::Vec3b(2, 0, 1);
		}
	}

	cv::Mat3f t = patch1 - patch2;
	cv::Mat3f res;
	cv::sqrt(t.mul(t), res);

    std::cout << "\n\n patch1 =\n " << patch1 ;
    std::cout << "\n\n patch2 =\n " << patch2 ;

    std::cout << "\n\n t =\n " << t ;
    std::cout << "\n\n res =\n " << res ;
    std::cout << "\n\n norm =\n " << cv::norm(patch1, patch2) ;

	cv::namedWindow("Input Image");
    cv::imshow("Input Image", dstM);
	cv::waitKey(); // wait for key stroke forever.
	return 0;
}
