#include "Cloning.h"
#include "opencv2/highgui.hpp"
#include <iostream>

#include <omp.h>

using namespace std;
using namespace cv;

float*** mycv::Cloning::source_MVC_0 = nullptr, *** mycv::Cloning::source_MVC_1 = nullptr, ***mycv::Cloning::source_MVC_2 = nullptr;
int mycv::Cloning::num_MVC_dim = 0, mycv::Cloning::prev_num_MVC_dim = 0;
mycv::Cloning::MVC_Coords* mycv::Cloning::mvc = nullptr;
vector<int> mycv::Cloning::vPart_MVC = vector<int>();
bool mycv::Cloning::alreadyPrecalculated = false;

void saveArray(const char* fileName, float*** array_name, int outerLength, int middleLength, int innerLength)
{
	std::ofstream output(fileName);

	for (size_t i = 0; i<outerLength; i++)
	{
		for (size_t j = 0; j < middleLength; j++)
		{
			for (size_t k = 0; k < innerLength; k++)
			{
				output << array_name[i][j][k] << std::endl;
			}
		}
	}
}
void mycv::Cloning::MVCSeamlessClone(Mat source, Mat target, Mat mask, Point center, Mat & clone)
{
	if (!mycv::Cloning::alreadyPrecalculated)
	{
		mycv::Cloning::num_MVC_dim = 2 * (source.cols + source.rows) - 4;
		mycv::Cloning::mvc = new MVC_Coords[num_MVC_dim];

		int mvc_indicator = 0;
		for (int n = (source.cols - 2); n > -1; n--)
		{
			mycv::Cloning::mvc[mvc_indicator].i = 0;
			mycv::Cloning::mvc[mvc_indicator].j = n;
			mvc_indicator++;
		}
		for (int m = 1; m < source.rows; m++)
		{
			mycv::Cloning::mvc[mvc_indicator].i = m;
			mycv::Cloning::mvc[mvc_indicator].j = 0;
			mvc_indicator++;
		}
		for (int n = 1; n < source.cols; n++)
		{
			mycv::Cloning::mvc[mvc_indicator].i = source.rows - 1;
			mycv::Cloning::mvc[mvc_indicator].j = n;
			mvc_indicator++;
		}
		for (int m = (source.rows - 2); m > -1; m--)
		{
			mycv::Cloning::mvc[mvc_indicator].i = m;
			mycv::Cloning::mvc[mvc_indicator].j = source.cols - 1;
			mvc_indicator++;
		}

		for (size_t indicator = 0; indicator <= num_MVC_dim; indicator += 20)
		{
			mycv::Cloning::vPart_MVC.push_back(indicator);
		}

		mycv::Cloning::prev_num_MVC_dim = mycv::Cloning::num_MVC_dim;
		mycv::Cloning::num_MVC_dim = mycv::Cloning::vPart_MVC.size();

		int err_theshold = 0.01;
		float**** source_MVC = new float***[3];
		float*** w_t = new float**[3];
		for (size_t c = 0; c < 3; c++)
		{
			source_MVC[c] = new float**[source.rows];
			//		clone_MVC[c] = new float*[source.rows];
			w_t[c] = new float*[source.rows];
			for (size_t i = 0; i < source.rows; i++)
			{
				source_MVC[c][i] = new float*[source.cols];
				//			clone_MVC[c][i] = new float[source.cols];
				w_t[c][i] = new float[source.cols];
				for (size_t j = 0; j < source.cols; j++)
				{
					source_MVC[c][i][j] = new float[num_MVC_dim];
					for (size_t indicator = 0; indicator < num_MVC_dim; indicator++)
					{
						source_MVC[c][i][j][indicator] = 0.0;
					}
				}
			}
		}

		bool check = true;
		for (int c = 0; c < 3; c++)
		{
			for (int i = 1; i < source.rows - 1; i++)
			{
				for (int j = 1; j < source.cols - 1; j++)
				{
					//Top right to Top left
					w_t[c][i][j] = 0;
					for (int indicator = 0; indicator < num_MVC_dim; indicator++)
					{
						//					int a = mvc[indicator].i;
						float b_, c_, w_i;
						float cos_alpha_i_min_1, cos_alpha_i, tan_alpha_i_min_1_over_2, tan_alpha_i_over_2;
						int prevIndicator = (indicator - 1) < 0 ? vPart_MVC[(indicator - 1) + num_MVC_dim] : vPart_MVC[(indicator - 1)];
						int currIndicator = vPart_MVC[indicator];
						int nextIndicator = vPart_MVC[(indicator + 1) % num_MVC_dim];
						int i_prev = mvc[prevIndicator].i, i_curr = mvc[currIndicator].i, i_next = mvc[nextIndicator].i, j_prev = mvc[prevIndicator].j, j_curr = mvc[currIndicator].j, j_next = mvc[nextIndicator].j;
						b_ = (i - i_prev)*(i - i_prev) + (j - j_prev)*(j - j_prev);
						c_ = (i - i_curr)*(i - i_curr) + (j - j_curr)*(j - j_curr);
						cos_alpha_i_min_1 = (b_ + c_ - ((i_prev - i_curr)*(i_prev - i_curr) + (j_prev - j_curr)*(j_prev - j_curr))) / (2 * sqrt(b_ * c_));
						b_ = c_;
						c_ = (i - i_next)*(i - i_next) + (j - j_next)*(j - j_next);
						cos_alpha_i = (b_ + c_ - ((i_next - i_curr)*(i_next - i_curr) + (j_next - j_curr)*(j_next - j_curr))) / (2 * sqrt(b_ * c_));
						tan_alpha_i_min_1_over_2 = sqrt((1 - cos_alpha_i_min_1) / (1 + cos_alpha_i_min_1));
						tan_alpha_i_over_2 = sqrt((1 - cos_alpha_i) / (1 + cos_alpha_i));
						w_i = (tan_alpha_i_over_2 + tan_alpha_i_min_1_over_2) / sqrt(b_);
						w_t[c][i][j] += w_i;
						source_MVC[c][i][j][indicator] = w_i;
					}
				}
			}
		}

		for (int c = 0; c < 3; c++)
		{
			for (int i = 0; i < source.rows; i++)
			{
				for (int j = 0; j < source.cols; j++)
				{
					for (size_t indicator = 0; indicator < (num_MVC_dim); indicator++)
					{
						source_MVC[c][i][j][indicator] = source_MVC[c][i][j][indicator] / w_t[c][i][j];
					}
				}
			}
		}

		source_MVC_0 = source_MVC[0];
		source_MVC_1 = source_MVC[1];
		source_MVC_2 = source_MVC[2];
		delete[] source_MVC;
		delete[] w_t;

		Cloning::alreadyPrecalculated = true;

		//saveArray("1.txt", source_MVC_0, source.rows, source.cols, num_MVC_dim);
		//saveArray("2.txt", source_MVC_1, source.rows, source.cols, num_MVC_dim);
		//saveArray("3.txt", source_MVC_2, source.rows, source.cols, num_MVC_dim);
	}

	float** diff_boundary = new float*[3];
	Mat source_per_channel[3], target_per_channel[3], target_ROI;
	source.convertTo(source, CV_32F, 1.0 / 255);
	target.convertTo(target, CV_32F, 1.0 / 255);
	split(source, source_per_channel);
	target(Rect(center.x - source.cols / 2, center.y - source.rows / 2, source.cols, source.rows)).copyTo(target_ROI);
	split(target_ROI, target_per_channel);

	for (int c = 0; c < 3; c++)
	{
		diff_boundary[c] = new float[num_MVC_dim];
		for (size_t indicator = 0; indicator < num_MVC_dim; indicator++)
		{
			diff_boundary[c][indicator] = target_per_channel[c].ptr<float>(mvc[vPart_MVC[indicator]].i)[mvc[vPart_MVC[indicator]].j] - source_per_channel[c].ptr<float>(mvc[vPart_MVC[indicator]].i)[mvc[vPart_MVC[indicator]].j];
		}
	}
#pragma omp parallel for
	for (int i = 0; i < source.rows; i++)
	{
		for (int j = 0; j < source.cols; j++)
		{
			float temp0 = 0.0, temp1 = 0.0, temp2 = 0.0;
			for (int indicator = 0; indicator < (num_MVC_dim); indicator++)
			{
				temp0 = temp0 + source_MVC_0[i][j][indicator] * diff_boundary[0][indicator];
				temp1 = temp1 + source_MVC_1[i][j][indicator] * diff_boundary[1][indicator];
				temp2 = temp2 + source_MVC_2[i][j][indicator] * diff_boundary[2][indicator];
			}
			source_per_channel[0].ptr<float>(i)[j] = source_per_channel[0].ptr<float>(i)[j] + temp0;
			source_per_channel[1].ptr<float>(i)[j] = source_per_channel[1].ptr<float>(i)[j] + temp1;
			source_per_channel[2].ptr<float>(i)[j] = source_per_channel[2].ptr<float>(i)[j] + temp2;
		}
	}

	delete[] diff_boundary;

	cv::merge(&source_per_channel[0], 3, source);
	source(Rect(1, 1, source.cols - 2, source.rows - 2)).copyTo(target(Rect(center.x - source.cols / 2, center.y - source.rows / 2, source.cols - 2, source.rows - 2)));
	target.copyTo(clone);
	clone.convertTo(clone, CV_8U, 255 / 1);
}

void mycv::Cloning::computeGradientX(const Mat &img, Mat &gx)
{
	Mat kernel = Mat::zeros(1, 3, CV_8S);
	kernel.at<char>(0, 2) = 1;
	kernel.at<char>(0, 1) = -1;

	if (img.channels() == 3)
	{
		filter2D(img, gx, CV_32F, kernel);
	}
	else if (img.channels() == 1)
	{
		Mat tmp[3];
		for (int chan = 0; chan < 3; ++chan)
		{
			filter2D(img, tmp[chan], CV_32F, kernel);
		}
		merge(tmp, 3, gx);
	}
}

void mycv::Cloning::computeGradientY(const Mat &img, Mat &gy)
{
	Mat kernel = Mat::zeros(3, 1, CV_8S);
	kernel.at<char>(2, 0) = 1;
	kernel.at<char>(1, 0) = -1;

	if (img.channels() == 3)
	{
		filter2D(img, gy, CV_32F, kernel);
	}
	else if (img.channels() == 1)
	{
		Mat tmp[3];
		for (int chan = 0; chan < 3; ++chan)
		{
			filter2D(img, tmp[chan], CV_32F, kernel);
		}
		merge(tmp, 3, gy);
	}
}

void mycv::Cloning::computeLaplacianX(const Mat &img, Mat &laplacianX)
{
	Mat kernel = Mat::zeros(1, 3, CV_8S);
	kernel.at<char>(0, 0) = -1;
	kernel.at<char>(0, 1) = 1;
	filter2D(img, laplacianX, CV_32F, kernel);
}

void mycv::Cloning::computeLaplacianY(const Mat &img, Mat &laplacianY)
{
	Mat kernel = Mat::zeros(3, 1, CV_8S);
	kernel.at<char>(0, 0) = -1;
	kernel.at<char>(1, 0) = 1;
	filter2D(img, laplacianY, CV_32F, kernel);
}

void mycv::Cloning::dst(const Mat& src, Mat& dest, bool invert)
{
	Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

	int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE : DFT_ROWS;

	src.copyTo(temp(Rect(1, 0, src.cols, src.rows)));

	for (int j = 0; j < src.rows; ++j)
	{
		float * tempLinePtr = temp.ptr<float>(j);
		const float * srcLinePtr = src.ptr<float>(j);
		for (int i = 0; i < src.cols; ++i)
		{
			tempLinePtr[src.cols + 2 + i] = -srcLinePtr[src.cols - 1 - i];
		}
	}

	Mat planes[] = { temp, Mat::zeros(temp.size(), CV_32F) };
	Mat complex;

	merge(planes, 2, complex);
	dft(complex, complex, flag);
	split(complex, planes);
	temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);

	for (int j = 0; j < src.cols; ++j)
	{
		float * tempLinePtr = temp.ptr<float>(j);
		for (int i = 0; i < src.rows; ++i)
		{
			float val = planes[1].ptr<float>(i)[j + 1];
			tempLinePtr[i + 1] = val;
			tempLinePtr[temp.cols - 1 - i] = -val;
		}
	}

	Mat planes2[] = { temp, Mat::zeros(temp.size(), CV_32F) };

	merge(planes2, 2, complex);
	dft(complex, complex, flag);
	split(complex, planes2);

	temp = planes2[1].t();
	dest = Mat::zeros(src.size(), CV_32F);
	temp(Rect(0, 1, src.cols, src.rows)).copyTo(dest);
}

void mycv::Cloning::idst(const Mat& src, Mat& dest)
{
	dst(src, dest, true);
}

void mycv::Cloning::solve(const Mat &img, Mat& mod_diff, Mat &result)
{
	const int w = img.cols;
	const int h = img.rows;

	Mat res;
	dst(mod_diff, res);

	for (int j = 0; j < h - 2; j++)
	{
		float * resLinePtr = res.ptr<float>(j);
		for (int i = 0; i < w - 2; i++)
		{
			resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
		}
	}

	idst(res, mod_diff);

	unsigned char *  resLinePtr = result.ptr<unsigned char>(0);
	const unsigned char * imgLinePtr = img.ptr<unsigned char>(0);
	const float * interpLinePtr = NULL;

	//first col
	for (int i = 0; i < w; ++i)
	result.ptr<unsigned char>(0)[i] = img.ptr<unsigned char>(0)[i];

	for (int j = 1; j < h - 1; ++j)
	{
		resLinePtr = result.ptr<unsigned char>(j);
		imgLinePtr = img.ptr<unsigned char>(j);
		interpLinePtr = mod_diff.ptr<float>(j - 1);

		//first row
		resLinePtr[0] = imgLinePtr[0];

		for (int i = 1; i < w - 1; ++i)
		{
			//saturate cast is not used here, because it behaves differently from the previous implementation
			//most notable, saturate_cast rounds before truncating, here it's the opposite.
			float value = interpLinePtr[i - 1];
			if (value < 0.)
			resLinePtr[i] = 0;
			else if (value > 255.0)
			resLinePtr[i] = 255;
			else
			resLinePtr[i] = static_cast<unsigned char>(value);
		}

		//last row
		resLinePtr[w - 1] = imgLinePtr[w - 1];
	}

	//last col
	resLinePtr = result.ptr<unsigned char>(h - 1);
	imgLinePtr = img.ptr<unsigned char>(h - 1);
	for (int i = 0; i < w; ++i)
	resLinePtr[i] = imgLinePtr[i];
}

void mycv::Cloning::poissonSolver(const Mat &img, Mat &laplacianX, Mat &laplacianY, Mat &result)
{
	const int w = img.cols;
	const int h = img.rows;

	Mat lap = Mat(img.size(), CV_32FC1);

	lap = laplacianX + laplacianY;

	Mat bound = img.clone();

	rectangle(bound, Point(1, 1), Point(img.cols - 2, img.rows - 2), Scalar::all(0), -1);
	Mat boundary_points;
	Laplacian(bound, boundary_points, CV_32F);

	boundary_points = lap - boundary_points;

	Mat mod_diff = boundary_points(Rect(1, 1, w - 2, h - 2));

	solve(img, mod_diff, result);
}

void mycv::Cloning::initVariables(const Mat &destination, const Mat &binaryMask)
{
	destinationGradientX = Mat(destination.size(), CV_32FC3);
	destinationGradientY = Mat(destination.size(), CV_32FC3);
	patchGradientX = Mat(destination.size(), CV_32FC3);
	patchGradientY = Mat(destination.size(), CV_32FC3);

	binaryMaskFloat = Mat(binaryMask.size(), CV_32FC1);
	binaryMaskFloatInverted = Mat(binaryMask.size(), CV_32FC1);

	//init of the filters used in the dst
	const int w = destination.cols;
	filter_X.resize(w - 2);
	for (int i = 0; i < w - 2; ++i)
	filter_X[i] = 2.0f * std::cos(static_cast<float>(CV_PI) * (i + 1) / (w - 1));

	const int h = destination.rows;
	filter_Y.resize(h - 2);
	for (int j = 0; j < h - 2; ++j)
	filter_Y[j] = 2.0f * std::cos(static_cast<float>(CV_PI) * (j + 1) / (h - 1));
}

void mycv::Cloning::computeDerivatives(const Mat& destination, const Mat &patch, const Mat &binaryMask)
{
	initVariables(destination, binaryMask);

	computeGradientX(destination, destinationGradientX);
	computeGradientY(destination, destinationGradientY);

	computeGradientX(patch, patchGradientX);
	computeGradientY(patch, patchGradientY);

	Mat Kernel(Size(3, 3), CV_8UC1);
	Kernel.setTo(Scalar(1));
	erode(binaryMask, binaryMask, Kernel, Point(-1, -1), 3);

	binaryMask.convertTo(binaryMaskFloat, CV_32FC1, 1.0 / 255.0);
}

void mycv::Cloning::scalarProduct(Mat mat, float r, float g, float b)
{
	vector <Mat> channels;
	split(mat, channels);
	multiply(channels[2], r, channels[2]);
	multiply(channels[1], g, channels[1]);
	multiply(channels[0], b, channels[0]);
	merge(channels, mat);
}

void mycv::Cloning::arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result) const
{
	vector <Mat> lhs_channels;
	vector <Mat> result_channels;

	split(lhs, lhs_channels);
	split(result, result_channels);

	for (int chan = 0; chan < 3; ++chan)
	multiply(lhs_channels[chan], rhs, result_channels[chan]);

	merge(result_channels, result);
}

void mycv::Cloning::poisson(const Mat &destination)
{
	Mat laplacianX = Mat(destination.size(), CV_32FC3);
	Mat laplacianY = Mat(destination.size(), CV_32FC3);

	laplacianX = destinationGradientX + patchGradientX;
	laplacianY = destinationGradientY + patchGradientY;

	computeLaplacianX(laplacianX, laplacianX);
	computeLaplacianY(laplacianY, laplacianY);

	split(laplacianX, rgbx_channel);
	split(laplacianY, rgby_channel);

	split(destination, output);

	for (int chan = 0; chan < 3; ++chan)
	{
		poissonSolver(output[chan], rgbx_channel[chan], rgby_channel[chan], output[chan]);
	}
}

void mycv::Cloning::evaluate(const Mat &I, const Mat &wmask, const Mat &cloned)
{
	bitwise_not(wmask, wmask);

	wmask.convertTo(binaryMaskFloatInverted, CV_32FC1, 1.0 / 255.0);

	arrayProduct(destinationGradientX, binaryMaskFloatInverted, destinationGradientX);
	arrayProduct(destinationGradientY, binaryMaskFloatInverted, destinationGradientY);

	poisson(I);

	merge(output, cloned);
}

void mycv::Cloning::normalClone(const Mat &destination, const Mat &patch, const Mat &binaryMask, Mat &cloned, int flag)
{
	const int w = destination.cols;
	const int h = destination.rows;
	const int channel = destination.channels();
	const int n_elem_in_line = w * channel;

	computeDerivatives(destination, patch, binaryMask);

	switch (flag)
	{
		case NORMAL_CLONE:
			arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
			arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);
			break;

		case MIXED_CLONE:
		{
			AutoBuffer<int> maskIndices(n_elem_in_line);
			for (int i = 0; i < n_elem_in_line; ++i)
			maskIndices[i] = i / channel;

			for (int i = 0; i < h; i++)
			{
				float * patchXLinePtr = patchGradientX.ptr<float>(i);
				float * patchYLinePtr = patchGradientY.ptr<float>(i);
				const float * destinationXLinePtr = destinationGradientX.ptr<float>(i);
				const float * destinationYLinePtr = destinationGradientY.ptr<float>(i);
				const float * binaryMaskLinePtr = binaryMaskFloat.ptr<float>(i);

				for (int j = 0; j < n_elem_in_line; j++)
				{
					int maskIndex = maskIndices[j];

					if (abs(patchXLinePtr[j] - patchYLinePtr[j]) >
					abs(destinationXLinePtr[j] - destinationYLinePtr[j]))
					{
						patchXLinePtr[j] *= binaryMaskLinePtr[maskIndex];
						patchYLinePtr[j] *= binaryMaskLinePtr[maskIndex];
					}
					else
					{
						patchXLinePtr[j] = destinationXLinePtr[j]
						* binaryMaskLinePtr[maskIndex];
						patchYLinePtr[j] = destinationYLinePtr[j]
						* binaryMaskLinePtr[maskIndex];
					}
				}
			}
		}
		break;

		case MONOCHROME_TRANSFER:
			Mat gray = Mat(patch.size(), CV_8UC1);
			cvtColor(patch, gray, COLOR_BGR2GRAY);

			computeGradientX(gray, patchGradientX);
			computeGradientY(gray, patchGradientY);

			arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
			arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);
			break;
	}

	evaluate(destination, binaryMask, cloned);
}

mycv::Stitcher mycv::Stitcher::createDefault(bool try_use_gpu)
{
	mycv::Stitcher stitcher;
	stitcher.setRegistrationResol(0.6);
	stitcher.setSeamEstimationResol(0.1);
	stitcher.setCompositingResol(ORIG_RESOL);
	stitcher.setPanoConfidenceThresh(1);
	stitcher.setWaveCorrection(true);
	stitcher.setWaveCorrectKind(cv::detail::WAVE_CORRECT_HORIZ);
	stitcher.setFeaturesMatcher(cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_use_gpu));
	stitcher.setBundleAdjuster(cv::makePtr<cv::detail::BundleAdjusterRay>());

#ifdef HAVE_OPENCV_CUDALEGACY
	if (try_use_gpu && cuda::getCudaEnabledDeviceCount() > 0)
	{
#ifdef HAVE_OPENCV_XFEATURES2D
		stitcher.setFeaturesFinder(makePtr<detail::SurfFeaturesFinderGpu>());
#else
		stitcher.setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
		stitcher.setWarper(makePtr<SphericalWarperGpu>());
		stitcher.setSeamFinder(makePtr<detail::GraphCutSeamFinderGpu>());
	}
	else
#endif
	{
#ifdef HAVE_OPENCV_XFEATURES2D
		stitcher.setFeaturesFinder(cv::makePtr<cv::detail::SurfFeaturesFinder>());
#else
		stitcher.setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
		stitcher.setWarper(cv::makePtr<cv::SphericalWarper>());
		stitcher.setSeamFinder(cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR));
	}

	stitcher.setExposureCompensator(cv::makePtr<cv::detail::BlocksGainCompensator>());
	stitcher.setBlender(cv::makePtr<cv::detail::MultiBandBlender>(try_use_gpu));

	stitcher.work_scale_ = 1;
	stitcher.seam_scale_ = 1;
	stitcher.seam_work_aspect_ = 1;
	stitcher.warped_image_scale_ = 1;

	return stitcher;
}

mycv::Stitcher::Status mycv::Stitcher::estimateTransform(cv::InputArrayOfArrays images)
{
	return mycv::Stitcher::estimateTransform(images, std::vector<std::vector<Rect> >());
}

mycv::Stitcher::Status mycv::Stitcher::estimateTransform(cv::InputArrayOfArrays images, const std::vector<std::vector<cv::Rect>>& rois)
{
	images.getUMatVector(imgs_);
	rois_ = rois;

//	std::cout << "INIT ESTIMATE" << std::endl;

	mycv::Stitcher::Status status;

	if ((status = mycv::Stitcher::matchImages()) != OK)
		return status;
//	std::cout << "SUCCESS MATCH" << std::endl;

	if ((status = mycv::Stitcher::estimateCameraParams()) != OK)
		return status;

	return OK;
}

mycv::Stitcher::Status mycv::Stitcher::composePanorama(cv::OutputArray pano)
{
	return mycv::Stitcher::composePanorama(std::vector<UMat>(), pano);
}

mycv::Stitcher::Status mycv::Stitcher::composePanorama(cv::InputArrayOfArrays images, cv::OutputArray pano)
{
	std::vector<cv::UMat> imgs;
	images.getUMatVector(imgs);
	if (!imgs.empty())
	{
		CV_Assert(imgs.size() == imgs_.size());

		cv::UMat img;
		seam_est_imgs_.resize(imgs.size());

		for (size_t i = 0; i < imgs.size(); ++i)
		{
			imgs_[i] = imgs[i];
			resize(imgs[i], img, Size(), seam_scale_, seam_scale_);
			seam_est_imgs_[i] = img.clone();
		}

		std::vector<cv::UMat> seam_est_imgs_subset;
		std::vector<cv::UMat> imgs_subset;

		for (size_t i = 0; i < indices_.size(); ++i)
		{
			imgs_subset.push_back(imgs_[indices_[i]]);
			seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
		}

		seam_est_imgs_ = seam_est_imgs_subset;
		imgs_ = imgs_subset;
	}

	cv::UMat pano_;

#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	std::vector<cv::Point> corners(imgs_.size());
	std::vector<cv::UMat> masks_warped(imgs_.size());
	std::vector<cv::UMat> images_warped(imgs_.size());
	std::vector<cv::Size> sizes(imgs_.size());
	std::vector<cv::UMat> masks(imgs_.size());

	// Prepare image masks
	for (size_t i = 0; i < imgs_.size(); ++i)
	{
		masks[i].create(seam_est_imgs_[i].size(), CV_8U);
		masks[i].setTo(cv::Scalar::all(255));
	}

	// Warp images and their masks
	cv::Ptr<cv::detail::RotationWarper> w = warper_->create(float(warped_image_scale_ * seam_work_aspect_));
	for (size_t i = 0; i < imgs_.size(); ++i)
	{
		cv::Mat_<float> K;
		cameras_[i].K().convertTo(K, CV_32F);
		K(0, 0) *= (float)seam_work_aspect_;
		K(0, 2) *= (float)seam_work_aspect_;
		K(1, 1) *= (float)seam_work_aspect_;
		K(1, 2) *= (float)seam_work_aspect_;

		corners[i] = w->warp(seam_est_imgs_[i], K, cameras_[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		w->warp(masks[i], K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}


	// Compensate exposure before finding seams
	exposure_comp_->feed(corners, images_warped, masks_warped);
	for (size_t i = 0; i < imgs_.size(); ++i)
		exposure_comp_->apply(int(i), corners[i], images_warped[i], masks_warped[i]);

	// Find seams
	std::vector<cv::UMat> images_warped_f(imgs_.size());
	for (size_t i = 0; i < imgs_.size(); ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

//	namedWindow("Test");
//	imshow("Test", masks_warped[0]);
//	waitKey(0);
	seam_finder_->find(images_warped_f, corners, masks_warped);
//	imshow("Test", masks_warped[0]);
//	waitKey(0);
//	destroyWindow("Test");

	// Release unused memory
	seam_est_imgs_.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

#if ENABLE_LOG
	t = getTickCount();
#endif

	cv::UMat img_warped, img_warped_s;
	cv::UMat dilated_mask, seam_mask, mask, mask_warped;

	//double compose_seam_aspect = 1;
	double compose_work_aspect = 1;
	bool is_blender_prepared = false;

	double compose_scale = 1;
	bool is_compose_scale_set = false;

	cv::UMat full_img, img;
	for (size_t img_idx = 0; img_idx < imgs_.size(); ++img_idx)
	{
#if ENABLE_LOG
		int64 compositing_t = getTickCount();
#endif

		// Read image and resize it if necessary
		full_img = imgs_[img_idx];
		if (!is_compose_scale_set)
		{
			if (compose_resol_ > 0)
				compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
			is_compose_scale_set = true;

			// Compute relative scales
			//compose_seam_aspect = compose_scale / seam_scale_;
			compose_work_aspect = compose_scale / work_scale_;

			// Update warped image scale
			warped_image_scale_ *= static_cast<float>(compose_work_aspect);
			w = warper_->create((float)warped_image_scale_);

			// Update corners and sizes
			for (size_t i = 0; i < imgs_.size(); ++i)
			{
				// Update intrinsics
				cameras_[i].focal *= compose_work_aspect;
				cameras_[i].ppx *= compose_work_aspect;
				cameras_[i].ppy *= compose_work_aspect;

				// Update corner and size
				cv::Size sz = full_img_sizes_[i];
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_img_sizes_[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes_[i].height * compose_scale);
				}

				cv::Mat K;
				cameras_[i].K().convertTo(K, CV_32F);
				cv::Rect roi = w->warpRoi(sz, K, cameras_[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (std::abs(compose_scale - 1) > 1e-1)
		{
#if ENABLE_LOG
			int64 resize_t = getTickCount();
#endif
			resize(full_img, img, cv::Size(), compose_scale, compose_scale);
		}
		else
			img = full_img;
		full_img.release();
		cv::Size img_size = img.size();

		cv::Mat K;
		cameras_[img_idx].K().convertTo(K, CV_32F);

#if ENABLE_LOG
		int64 pt = getTickCount();
#endif
		// Warp the current image
		w->warp(img, K, cameras_[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
#if ENABLE_LOG
		pt = getTickCount();
#endif

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		w->warp(mask, K, cameras_[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
#if ENABLE_LOG
		pt = getTickCount();
#endif

		// Compensate exposure
		exposure_comp_->apply((int)img_idx, corners[img_idx], img_warped, mask_warped);
#if ENABLE_LOG
		pt = getTickCount();
#endif

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

		// Make sure seam mask has proper size
		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());

		bitwise_and(seam_mask, mask_warped, mask_warped);

#if ENABLE_LOG
		pt = getTickCount();
#endif

		if (!is_blender_prepared)
		{
			blender_->prepare(corners, sizes);
			is_blender_prepared = true;
		}

#if ENABLE_LOG
		int64 feed_t = getTickCount();
#endif
		// Blend the current image
		blender_->feed(img_warped_s, mask_warped, corners[img_idx]);
//		std::cout << corners[img_idx] << std::endl;
		/*
		cv::namedWindow("maskWindow");
		imshow("maskWindow", mask_warped);
		waitKey();
		destroyWindow("maskWindow");
		*/
	}

#if ENABLE_LOG
	int64 blend_t = getTickCount();
#endif
	cv::UMat result, result_mask;
	blender_->blend(result, result_mask);

	// Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
	// so convert it to avoid user confusing
	result.convertTo(pano, CV_8U);

	return OK;
}

mycv::Stitcher::Status mycv::Stitcher::stitch(cv::InputArrayOfArrays images, cv::OutputArray pano)
{
	mycv::Stitcher::Status status = mycv::Stitcher::estimateTransform(images);
	if (status != OK)
		return status;
//	std::cout << "SUCCESS ESTIMATE" << std::endl;
	return mycv::Stitcher::composePanorama(pano);
}

mycv::Stitcher::Status mycv::Stitcher::stitch(cv::InputArrayOfArrays images, const std::vector<std::vector<cv::Rect>>& rois, cv::OutputArray pano)
{
	mycv::Stitcher::Status status = mycv::Stitcher::estimateTransform(images, rois);
	if (status != OK)
		return status;
//	std::cout << "SUCCESS ESTIMATE" << std::endl;
	return mycv::Stitcher::composePanorama(pano);
}

mycv::Stitcher::Status mycv::Stitcher::matchImages()
{
	if ((int)imgs_.size() < 2)
	{
		LOGLN("Need more images");
		return ERR_NEED_MORE_IMGS;
	}

	work_scale_ = 1;
	seam_work_aspect_ = 1;
	seam_scale_ = 1;
	bool is_work_scale_set = false;
	bool is_seam_scale_set = false;
	UMat full_img, img;
	features_.resize(imgs_.size());
	seam_est_imgs_.resize(imgs_.size());
	full_img_sizes_.resize(imgs_.size());

	LOGLN("Finding features...");
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	for (size_t i = 0; i < imgs_.size(); ++i)
	{
		full_img = imgs_[i];
		full_img_sizes_[i] = full_img.size();

		if (registr_resol_ < 0)
		{
			img = full_img;
			work_scale_ = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale_ = std::min(1.0, std::sqrt(registr_resol_ * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale_, work_scale_);
		}
		if (!is_seam_scale_set)
		{
			seam_scale_ = std::min(1.0, std::sqrt(seam_est_resol_ * 1e6 / full_img.size().area()));
			seam_work_aspect_ = seam_scale_ / work_scale_;
			is_seam_scale_set = true;
		}

		if (rois_.empty())
			(*features_finder_)(img, features_[i]);
		else
		{
			std::vector<Rect> rois(rois_[i].size());
			for (size_t j = 0; j < rois_[i].size(); ++j)
			{
				Point tl(cvRound(rois_[i][j].x * work_scale_), cvRound(rois_[i][j].y * work_scale_));
				Point br(cvRound(rois_[i][j].br().x * work_scale_), cvRound(rois_[i][j].br().y * work_scale_));
				rois[j] = Rect(tl, br);
			}
			(*features_finder_)(img, features_[i], rois);
		}
		features_[i].img_idx = (int)i;
		LOGLN("Features in image #" << i + 1 << ": " << features_[i].keypoints.size());

		resize(full_img, img, Size(), seam_scale_, seam_scale_);
		seam_est_imgs_[i] = img.clone();
	}

	// Do it to save memory
	features_finder_->collectGarbage();
	full_img.release();
	img.release();

	LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	LOG("Pairwise matching");
#if ENABLE_LOG
	t = getTickCount();
#endif
	(*features_matcher_)(features_, pairwise_matches_, matching_mask_);
	features_matcher_->collectGarbage();
	LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	// Leave only images we are sure are from the same panorama
	indices_ = detail::leaveBiggestComponent(features_, pairwise_matches_, (float)conf_thresh_);
	std::vector<UMat> seam_est_imgs_subset;
	std::vector<UMat> imgs_subset;
	std::vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices_.size(); ++i)
	{
		imgs_subset.push_back(imgs_[indices_[i]]);
		seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
		full_img_sizes_subset.push_back(full_img_sizes_[indices_[i]]);
	}
	seam_est_imgs_ = seam_est_imgs_subset;
	imgs_ = imgs_subset;
	full_img_sizes_ = full_img_sizes_subset;

	if ((int)imgs_.size() < 2)
	{
		LOGLN("Need more images");
		return ERR_NEED_MORE_IMGS;
	}

	return OK;
}

mycv::Stitcher::Status mycv::Stitcher::estimateCameraParams()
{
	detail::HomographyBasedEstimator estimator;
	if (!estimator(features_, pairwise_matches_, cameras_))
		return ERR_HOMOGRAPHY_EST_FAIL;

	for (size_t i = 0; i < cameras_.size(); ++i)
	{
		Mat R;
		cameras_[i].R.convertTo(R, CV_32F);
		cameras_[i].R = R;
		//LOGLN("Initial intrinsic parameters #" << indices_[i] + 1 << ":\n " << cameras_[i].K());
	}

	bundle_adjuster_->setConfThresh(conf_thresh_);
	if (!(*bundle_adjuster_)(features_, pairwise_matches_, cameras_))
		return ERR_CAMERA_PARAMS_ADJUST_FAIL;

	// Find median focal length and use it as final image scale
	std::vector<double> focals;
	for (size_t i = 0; i < cameras_.size(); ++i)
	{
		//LOGLN("Camera #" << indices_[i] + 1 << ":\n" << cameras_[i].K());
		focals.push_back(cameras_[i].focal);
	}

	std::sort(focals.begin(), focals.end());
	if (focals.size() % 2 == 1)
		warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct_)
	{
		std::vector<Mat> rmats;
		for (size_t i = 0; i < cameras_.size(); ++i)
			rmats.push_back(cameras_[i].R.clone());
		detail::waveCorrect(rmats, wave_correct_kind_);
		for (size_t i = 0; i < cameras_.size(); ++i)
			cameras_[i].R = rmats[i];
	}

	return OK;
}
