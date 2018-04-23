#include <inpaint/criminisi_inpainter.h>
#include <iostream>
#include <opencv2/opencv.hpp>

struct ImageInfo
{
	ImageInfo() : leftMouseDown(false), rightMouseDown(false), patchSize(9) {}
	cv::Mat image;
	cv::Mat targetMask;
    cv::Mat sourceMask;
	cv::Mat displayImage;
	cv::Mat sourceImage;
	bool leftMouseDown;
    bool rightMouseDown;
	int  patchSize;
};

cv::Point point1, point2;
bool beginDraw = false;

void onMouse(int eventType, int x, int y, int flags, void* data)
{
    // reinterpret generic data type to specificed type.
	ImageInfo &ii = *reinterpret_cast<ImageInfo*>(data);

	if ( (eventType == CV_EVENT_LBUTTONDOWN) || (eventType == CV_EVENT_MOUSEMOVE && flags == CV_EVENT_FLAG_LBUTTON) )
    {
		cv::Mat &mask    = ii.targetMask;
		cv::Scalar color = cv::Scalar(0,250,0);
		cv::circle(mask, cv::Point(x, y), ii.displayImage.rows / 60, cv::Scalar(255), -1);
		ii.displayImage.setTo(color, mask);
    }
	else if ( (eventType == CV_EVENT_RBUTTONDOWN) || (eventType == CV_EVENT_MOUSEMOVE && flags == CV_EVENT_FLAG_RBUTTON) )
	{
		cv::Mat &mask    =  ii.sourceMask;
		cv::Scalar color =  cv::Scalar(0,250,250);
		cv::circle(mask, cv::Point(x, y), ii.displayImage.rows / 60, cv::Scalar(255), -1);
		ii.displayImage.setTo(color, mask);
	}
//   cv::imshow("Source mask", ii.sourceMask);
//   cv::imshow("Target mask", ii.targetMask);
}

void mainLoop(ImageInfo &ii);

int main(int argc, char **argv)
{
	if (argc != 2)
    {
		std::cout << argv[0] << " image_file_name" << std::endl;
		return -1;
	}

    // How to check read image ok?
	cv::Mat inputImage = cv::imread(argv[1]);
    printf("- Image size w=%d, h=%d.\n", inputImage.cols, inputImage.rows);

	// resize the image, if oversize.
    while (inputImage.cols > 720 || inputImage.rows > 800)
    {
        cv::resize( inputImage, inputImage, cv::Size(round(0.618 * inputImage.cols), round(0.618 * inputImage.rows)) );
        printf("- Resized Image size w=%d, h=%d.\n", inputImage.cols, inputImage.rows);
    }

	ImageInfo ii;
	ii.image        = inputImage.clone();
	ii.sourceImage  = inputImage.clone();
	ii.displayImage = ii.image.clone();
    ii.targetMask.create(ii.image.size(), CV_8UC1);
	ii.targetMask.setTo(0); // a black picture.
    ii.sourceMask.create(ii.image.size(), CV_8UC1);
	ii.sourceMask.setTo(0); // a black picture.

	//cv::namedWindow("Image Inpaint", cv::WINDOW_NORMAL);
	cv::namedWindow("Image Inpaint");
	cv::setMouseCallback("Image Inpaint", onMouse, &ii);
	cv::createTrackbar("Patchsize", "Image Inpaint", &ii.patchSize, 50);

    mainLoop(ii);
	return 0;
}

void mainLoop(ImageInfo &ii)
{
	Inpaint::CriminisiInpainter inpainter;
	bool loop = true;
	bool inpainterInitialized = false;
	int  iterations = 0;
	while (loop)
	{
		cv::imshow("Image Inpaint", ii.displayImage);
		int key = cv::waitKey(10); // wait for key strok for 10 ms.
		switch (key)
		{
			case 'x':
			{
				loop = false;
				break;
			}
			case 'i':
			{
				if (!inpainterInitialized)
				{
					inpainter.setSourceImage(ii.image);
					inpainter.setTargetMask(ii.targetMask);
					inpainter.setSourceMask(ii.sourceMask);
					inpainter.setPatchSize(ii.patchSize);
					inpainter.initialize();
					inpainterInitialized = true;
				}
				break;
			}
			case 'r':
			{
				ii.image        = ii.sourceImage.clone();
				ii.displayImage = ii.sourceImage.clone();
				ii.targetMask.create(ii.image.size(), CV_8UC1);
				ii.targetMask.setTo(0);
				ii.sourceMask.create(ii.image.size(), CV_8UC1);
				ii.sourceMask.setTo(0);
				inpainterInitialized = false;
				break;
			}
			default:
			{
				break;
			}
		}

		if (inpainterInitialized)
		{
			if (inpainter.hasMoreSteps())
			{
				++iterations;
				inpainter.step();
				if (!(iterations & 0xf))
				{
					inpainter.image().copyTo( ii.displayImage );
					ii.displayImage.setTo(cv::Scalar(0, 250, 0), inpainter.targetRegion());
				}
			}
			else
			{
				ii.image        = inpainter.image().clone();
				ii.displayImage = ii.image.clone();
				ii.targetMask   = inpainter.targetRegion().clone();
				inpainterInitialized = false;
			}
		}

	}

	cv::imshow("Source", ii.sourceImage);
	if (!inpainter.image().empty())
		cv::imshow("Final Result", inpainter.image()); // bug here.
	cv::waitKey(); // wait for key stroke forever.

	printf("- Total iterations = %d.\n", iterations);
}