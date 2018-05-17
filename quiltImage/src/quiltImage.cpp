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

// [low, up]
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

static double L2Norm(const cv::Mat & ov1, const cv::Mat & ov2)
{
	assert( (ov1.rows == ov2.rows) && (ov1.cols == ov2.cols));
	double ret = 0.0;
	int w = ov1.cols;
	int h = ov1.rows;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			double dist = cv::norm(ov1.at<cv::Vec3b>(i, j), ov2.at<cv::Vec3b>(i, j));
			ret += dist;
		}
	}
	//printf("- L2Norm  = %lf\n", ret);
	return ret;
}

static cv::Mat getBestVertical(const cv::Mat & image, const cv::Mat & ov, double  bestErr, const int k)
{
	int x = 0;
	int y = 0;

	int w = image.cols;
	int h = image.rows;
	int ow = ov.cols;
	bestErr = 100000.0;
	for (int i = 0; i <= h - k; ++i)
	{
		for (int j = 0; j < w - k; ++j)
		{
			cv::Mat patch = image(cv::Rect(j, i, k, k));
			double err = L2Norm(patch(cv::Rect(0, 0, ow, k) ), ov);
			if (err < bestErr)
			{
				bestErr = err;
				x = j; y = i;
			}
		}
	}
	//printf("- Vertical best patch(%d, %d), %lf\n", x, y, bestErr);
	return image(cv::Rect(x, y, k, k));
}

static cv::Mat getBestHorizonal(const cv::Mat & image, const cv::Mat & ov, double  bestErr, const int k)
{
	int x = 0;
	int y = 0;

	int w = image.cols;
	int h = image.rows;
	int oh = ov.rows;
	bestErr = 100000.0;
	for (int i = 0; i <= h - k; ++i)
	{
		for (int j = 0; j < w - k; ++j)
		{
			cv::Mat patch = image(cv::Rect(j, i, k, k));
			double err = L2Norm(patch(cv::Rect(0, 0, k, oh)), ov);
			if (err < bestErr)
			{
				bestErr = err;
				x = j; y = i;
			}
		}
	}
	//printf("- Horizonal best patch(%d, %d), %lf\n", x, y, bestErr);
	return image(cv::Rect(x, y, k, k));
}

static cv::Mat getBestBoth(const cv::Mat & image, const cv::Mat & vov, const cv::Mat & hov, double  bestErr, const int k)
{
	assert(hov.rows == vov.cols);
	int x = 0;
	int y = 0;

	int w = image.cols;
	int h = image.rows;
	int o = hov.rows;
	bestErr = 100000.0;

	for (int i = 0; i <= h - k; ++i)
	{
		for (int j = 0; j < w - k; ++j)
		{
			cv::Mat patch = image(cv::Rect(j, i, k, k));
			double err = L2Norm(patch(cv::Rect(0, 0, k, o)), hov) + L2Norm(patch(cv::Rect(0, 0, o, k)), vov);
			if (err < bestErr)
			{
				bestErr = err;
				x = j; y = i;
			}
		}
	}
	//printf("- Both best patch(%d, %d), %lf\n", x, y, bestErr);
	return image(cv::Rect(x, y, k, k));
}

static void putOverlapHorizonal(cv::Mat & image, const cv::Mat & patch, int x, int y, int oh)
{
	assert(patch.rows == patch.cols);
	int k = patch.rows;
	cv::Mat roi = image(cv::Rect(x, y + oh /2, k, k - oh / 2));
	patch(cv::Rect(0, oh / 2, k, k - oh / 2)).copyTo(roi);
}

static void putOverlapVertical(cv::Mat & image, const cv::Mat & patch, int x, int y, int ow)
{
	assert(patch.rows == patch.cols);
	int k = patch.rows;
	cv::Mat roi = image(cv::Rect(x + ow / 2, y, k - ow / 2, k));
	patch(cv::Rect(ow / 2, 0, k - ow / 2, k)).copyTo(roi);
}

static void putOverlapBoth(cv::Mat & image, const cv::Mat & patch, int x, int y, int o)
{
	assert(patch.rows == patch.cols);
	int k = patch.rows;
	cv::Mat roi = image(cv::Rect(x + o / 2, y + o / 2, k - o / 2, k - o / 2));
	patch(cv::Rect(o / 2, o / 2, k - o / 2, k - o / 2)).copyTo(roi);
}

cv::Mat L2Synthesis(const cv::Mat & image, int k = 13, int o = 2)
{
	int w = image.cols * 2;
	int h = image.rows * 2;
	double bestErr = 100000000000.0;
	srand(time(NULL));

	cv::Mat ret(h, w, CV_8UC3, cv::Scalar::all(0));
	cv::Mat roi(ret, cv::Rect(0, 0, k, k));
	image(cv::Rect(0, 0, k, k)).copyTo(roi);
	cv::Mat vov = image(cv::Rect(k - o, 0, o, k));
	cv::Mat hov = image(cv::Rect(0, k - o, k, o));

	for (int j = k - o; j <= w - k; j += (k -o) )
	{
		cv::Mat patch = getBestVertical(image, vov, bestErr, k);
	    vov = patch(cv::Rect(k - o, 0, o, k));
		putOverlapVertical( ret, patch, j, 0, o);
	}
	for (int i = k - o; i <= h - k; i += (k - o) )
	{
		cv::Mat patch = getBestHorizonal(image, hov, bestErr, k);
	    hov = patch(cv::Rect(0, k - o, k, o));
		putOverlapHorizonal( ret, patch, 0, i, o);
	}

	for (int i = k - o; i <= h - k; i += (k - o) )
	{
		for (int j = k - o; j <= w - k; j += (k - o) )
		{
			hov = ret(cv::Rect(j, i, k, o));
			vov = ret(cv::Rect(j, i, o, k));
			cv::Mat patch = getBestBoth(image, vov, hov, bestErr, k);
			putOverlapBoth( ret, patch, j, i, o);
		}
	}
	return ret;
}

void sumChannels(const cv::Mat3f &a, cv::Mat1f &b)
 {
	CV_Assert(a.type() == CV_32FC3);
	int rows = a.rows;
	int cols = b.cols;
	b.create(a.size());

	for (int i = 0; i < rows; ++i)
	{
		const float *sptr = a.ptr<float>(i);
			  float *dptr = b.ptr<float>(i);
		for (int j = 0; j < cols; ++j)
		{
			dptr[j] = sptr[0] + sptr[1] + sptr[2];
			sptr += 3;
		}
	}
}

static void minCutVertical(const cv::Mat3f& ov1, const cv::Mat3f & ov2, std::vector<int>& cut)
{
	cv::Mat3f diff = ov1 - ov2;
	diff = diff.mul(diff);
	cv::Mat3f res;
	cv::sqrt(diff, res);
	cv::Mat1f a;
	sumChannels(res, a);

	struct elem { int i; float v; };
	int h = ov1.rows;
	int w = ov1.cols;
	CV_Assert(h == cut.size());
	std::vector<std::vector<int> > track(h, std::vector<int>(w, 0) );

	for (int i = 0; i < w; ++i)
	{
		track[0][i] = i;
	}
	
	for (int i = 1; i < h; i++)
	{
		const float *p = a.ptr<float>(i - 1);
		      float *c = a.ptr<float>(i);
		for (int j = 0; j < w; ++j)
		{
			if (j == 0)
			{
				p[j] < p[j + 1] ? track[i][j] = j, c[j] += p[j] : track[i][j] = j + 1, c[j] += p[j + 1];
			}
			else if (j == w - 1)
			{
				p[j] < p[j - 1] ? track[i][j] = j, c[j] += p[j] : track[i][j] = j - 1, c[j] += p[j - 1];
			}
			else
			{
				p[j] < p[j - 1] ? (p[j] < p[j + 1] ? track[i][j] = j, c[j] += p[j] : track[i][j] = j + 1, c[j] += p[j + 1]) : \
					              (p[j - 1] < p[j + 1] ? track[i][j] = j - 1, c[j] += p[j - 1] : track[i][j] = j + 1, c[j] += p[j + 1]);
			}
		}
	}

    const float *p = a.ptr<float>(h - 1);
	int index = 0;
	float v = p[0];
	for (int i = 1; i < w; ++i)
	{
		if (p[i] < v ) v = p[i], index = i;
	}

	cut[h - 1] = index;
	for (int i = h - 1; i > 0; --i)
	{
		index = track[i][index];
		cut[i - 1] = index;
	}

}

static void minCutHorizonal(const cv::Mat3f& ov1, const cv::Mat3f & ov2, std::vector<int>& cut)
{ }

static void quiltOverlapVertical(cv::Mat & image, const cv::Mat & patch, int x, int y, int ow)
{
	CV_Assert(patch.rows == patch.cols);
	int k = patch.rows;
	std::vector<int> cut(k, 0);

	cv::Mat3b dst = image(cv::Rect(x, y, ow, k));
	cv::Mat3b src = patch(cv::Rect(0, 0, ow, k));

	cv::Mat3f ov1 = image(cv::Rect(x, y, ow, k));
	cv::Mat3f ov2 = patch(cv::Rect(0, 0, ow, k));
	minCutVertical(ov1, ov2, cut);

	for (int row = 0; row < k; row++)
	{
		int b = cut[row];
		for (int col = b; col < ow; ++col)
		{
			dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(row, col);
		}
	}
}

static void quiltOverlapHorizonal(cv::Mat & image, const cv::Mat & patch, int x, int y, int oh)
{
	CV_Assert(patch.rows == patch.cols);
	int k = patch.rows;
	std::vector<int> cut(k, 0);

	cv::Mat3b dst = image(cv::Rect(x, y, k, oh));
	cv::Mat3b src = patch(cv::Rect(0, 0, k, oh));

	cv::Mat3f t1 = image(cv::Rect(x, y, k, oh));
	cv::Mat3f t2 = patch(cv::Rect(0, 0, k, oh));

	cv::Mat3f ov1;
	cv::Mat3f ov2;

	cv::transpose(t1, ov1);
	cv::transpose(t2, ov2);

	minCutVertical(ov1, ov2, cut);

	for (int col = 0; col < k; ++col)
	{
		int b = cut[col];
		for (int row = b; row < oh; ++row)
		{
			dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(row, col);
		}
	}
}

static void quiltOverlapBoth(cv::Mat & image, const cv::Mat & patch, int x, int y, int o)
{
	CV_Assert(patch.rows == patch.cols);
	int k = patch.rows;
	std::vector<int> cut(k, 0);

	cv::Mat3b dst = image(cv::Rect(x, y, k, o));
	cv::Mat3b src = patch(cv::Rect(0, 0, k, o));

	cv::Mat3f t1 = image(cv::Rect(x, y, k, o));
	cv::Mat3f t2 = patch(cv::Rect(0, 0, k, o));

	cv::Mat3f ov1;
	cv::Mat3f ov2;

	cv::transpose(t1, ov1);
	cv::transpose(t2, ov2);

	minCutVertical(ov1, ov2, cut);

	for (int col = 0; col < k; ++col)
	{
		int b = cut[col];
		for (int row = b; row < o; ++row)
		{
			dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(row, col);
		}
	}

	dst = image(cv::Rect(x, y + o, o, k - o));
	src = patch(cv::Rect(0, o, o, k - o));

	ov1 = image(cv::Rect(x, y + o, o, k - o));
	ov2 = patch(cv::Rect(0, o, o, k - o));
	
	std::vector<int> cut1(k - o, 0);
	minCutVertical(ov1, ov2, cut1);

	for (int row = 0; row < k - o; ++row)
	{
		int b = cut1[row];
		for (int col = b; col < o; ++col)
		{
			dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(row, col);
		}
	}
}

static void quiltOverlapVertical2(cv::Mat & targetOvRegion, const cv::Mat & ovRegion)
{
	int k  = ovRegion.rows;
	int ow = ovRegion.cols;
	std::vector<int> cut(k, 0);

	cv::Mat3b dst = targetOvRegion; 
	cv::Mat3b src = ovRegion;

	cv::Mat3f ov1 = dst;
	cv::Mat3f ov2 = src;
	minCutVertical(ov1, ov2, cut);

	for (int row = 0; row < k; row++)
	{
		int b = cut[row];
		for (int col = b; col < ow; ++col)
		{
			dst.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(row, col);
		}
	}
}

static cv::Mat guessBestVertical(const cv::Mat & image, const cv::Mat & ovRegion, double  bestErr)
{
	int x = 0;
	int y = 0;

	int w = image.cols;
	int h = image.rows;
	int ovWidth  = ovRegion.cols;
	int ovHeight = ovRegion.rows, k = ovHeight;

	struct coord { int x; int y; };
    std::vector<coord> candidates;

	bestErr = 1000.0;

	for (int i = 0; i <= h - k; ++i)
	{
		for (int j = 0; j < w - k; ++j)
		{
			cv::Mat leftOvRegion = image(cv::Rect(j, i, ovWidth, ovHeight));
			double err = cv::norm(ovRegion, leftOvRegion);
			//double err = L2Norm(ovRegion, leftOvRegion);
			double threshold = 1.1 * bestErr;
			if (err < threshold)
			{
				candidates.push_back({j, i});
				if (err < bestErr) bestErr = err;
			}
		}
	}
	int n = candidates.size();
    CV_Assert(n <= 0);
	int index = randint(0, n - 1);
	x = candidates[index].x;
	y = candidates[index].y;

	//printf("- Vertical best patch(%d, %d), %lf\n", x, y, bestErr);
	return image(cv::Rect(x, y, k, k));
}

cv::Mat minCutSynthesis(const cv::Mat & image, int k = 13, int o = 2)
{
	int w = image.cols * 2;
	int h = image.rows * 2;
	int xbegin = k - o, xend = w - k;
	int ybegin = k - o, yend = h - k;

	double bestErr = 100000000000.0;
	srand(time(NULL));

	cv::Mat ret(h, w, CV_8UC3, cv::Scalar::all(0));
	cv::Mat firstPatch(ret, cv::Rect(0, 0, k, k));
	image(cv::Rect(0, 0, k, k)).copyTo(firstPatch);

	cv::Mat &vov = image(cv::Rect(k - o, 0, o, k));
	cv::Mat &hov = image(cv::Rect(0, k - o, k, o));

	for (int x = xbegin, y = 0; x <= xend; x += (k -o) )
	{
		cv::Mat patch = guessBestVertical(image, vov, bestErr);
	    vov = patch(cv::Rect(k - o, 0, o, k));
		cv::Mat nonOV  = patch(cv::Rect(o, 0, k - o, k));
		cv::Mat leftOV = patch(cv::Rect(0, 0, o, k));
        cv::Mat targetOvRegion = ret(cv::Rect(x, y, o, k));
		nonOV.copyTo( ret(cv::Rect(x + o, y, k - o, k)) );
		quiltOverlapVertical2( targetOvRegion, leftOV );

		//cv::Mat src = patch(cv::Rect(o, 0, k - o, k));
		//cv::Mat dst = ret(cv::Rect(j + o, 0, k - o, k));
		//src.copyTo(dst);
	}

	for (int y = ybegin; y <= yend; y += (k - o) )
	{
		cv::Mat patch = getBestHorizonal(image, hov, bestErr, k);
	    hov = patch(cv::Rect(0, k - o, k, o));
		quiltOverlapHorizonal( ret, patch, 0, y, o);
		patch(cv::Rect(0, o, k, k - o)).copyTo(ret(cv::Rect(0, y + o,  k, k - o)) );
	}

	for (int y = ybegin; y <= yend; y += (k - o) )
	{
		for (int x = xbegin; x <= xend; x += (k - o) )
		{
			hov = ret(cv::Rect(x, y, k, o));
			vov = ret(cv::Rect(x, y, o, k));
			cv::Mat patch = getBestBoth(image, vov, hov, bestErr, k);
			quiltOverlapBoth( ret, patch, x, y, o);
		    patch(cv::Rect(o, o, k - o, k - o)).copyTo(ret(cv::Rect(x + o, y + o, k - o, k - o)) );
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
	cv::Mat dstM(cv::Size(w * 3, h * 2), inputImage.type(), cv::Scalar::all(0));
	cv::Mat roiM(dstM, cv::Rect(0, 0, w, h));
	inputImage.copyTo(roiM);
	roiM = dstM(cv::Rect(w, 0, w * 2, h * 2));

	//naiveSynthesis(inputImage, 11).copyTo(roiM);
	//L2Synthesis(inputImage, 19, 3).copyTo(roiM);
	minCutSynthesis(inputImage, 24, 4).copyTo(roiM);
    cv::imshow("Synthesis Image", dstM); cv::waitKey(); // wait for key stroke forever.
	return 0;
}
