#include "Cloning.h"
#include <iostream>

float*** Cloning::source_MVC_0 = nullptr, *** Cloning::source_MVC_1 = nullptr, ***Cloning::source_MVC_2 = nullptr;
int Cloning::num_MVC_dim = 0, Cloning::prev_num_MVC_dim = 0;
MVC_Coords* Cloning::mvc = nullptr;
vector<int> Cloning::vPart_MVC = vector<int>();
bool Cloning::alreadyPrecalculated = false;

void Cloning::MVCSeamlessClone(Mat source, Mat target, Mat mask, Point center, Mat & clone)
{
	if (!Cloning::alreadyPrecalculated)
	{
		Cloning::num_MVC_dim = 2 * (source.cols + source.rows) - 4;
		Cloning::mvc = new MVC_Coords[num_MVC_dim];

		int mvc_indicator = 0;
		for (int n = (source.cols - 2); n > -1; n--)
		{
			Cloning::mvc[mvc_indicator].i = 0;
			Cloning::mvc[mvc_indicator].j = n;
			mvc_indicator++;
		}
		for (int m = 1; m < source.rows; m++)
		{
			Cloning::mvc[mvc_indicator].i = m;
			Cloning::mvc[mvc_indicator].j = 0;
			mvc_indicator++;
		}
		for (int n = 1; n < source.cols; n++)
		{
			Cloning::mvc[mvc_indicator].i = source.rows - 1;
			Cloning::mvc[mvc_indicator].j = n;
			mvc_indicator++;
		}
		for (int m = (source.rows - 2); m > -1; m--)
		{
			Cloning::mvc[mvc_indicator].i = m;
			Cloning::mvc[mvc_indicator].j = source.cols - 1;
			mvc_indicator++;
		}

		for (size_t indicator = 0; indicator <= num_MVC_dim; indicator += 20)
		{
			Cloning::vPart_MVC.push_back(indicator);
		}

		Cloning::prev_num_MVC_dim = Cloning::num_MVC_dim;
		Cloning::num_MVC_dim = Cloning::vPart_MVC.size();

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
	}

	float** diff_boundary = new float*[3];
	Mat source_per_channel[3], target_per_channel[3], target_ROI;
	source.convertTo(source, CV_32F, 1.0 / 255);
	target.convertTo(target, CV_32F, 1.0 / 255);
	split(source, source_per_channel);
//	std::cout << source.size() << endl;
//	std::cout << target.size() << endl;
//	std::cout << (center.x - source.cols / 2) << endl;
//	std::cout << (center.y - source.rows / 2) << endl;
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

/*
void Cloning::computeGradientX(const Mat &img, Mat &gx)
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

void Cloning::computeGradientY(const Mat &img, Mat &gy)
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

void Cloning::computeLaplacianX(const Mat &img, Mat &laplacianX)
{
Mat kernel = Mat::zeros(1, 3, CV_8S);
kernel.at<char>(0, 0) = -1;
kernel.at<char>(0, 1) = 1;
filter2D(img, laplacianX, CV_32F, kernel);
}

void Cloning::computeLaplacianY(const Mat &img, Mat &laplacianY)
{
Mat kernel = Mat::zeros(3, 1, CV_8S);
kernel.at<char>(0, 0) = -1;
kernel.at<char>(1, 0) = 1;
filter2D(img, laplacianY, CV_32F, kernel);
}

void Cloning::dst(const Mat& src, Mat& dest, bool invert)
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

void Cloning::idst(const Mat& src, Mat& dest)
{
dst(src, dest, true);
}

void Cloning::solve(const Mat &img, Mat& mod_diff, Mat &result)
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

void Cloning::poissonSolver(const Mat &img, Mat &laplacianX, Mat &laplacianY, Mat &result)
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

void Cloning::initVariables(const Mat &destination, const Mat &binaryMask)
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

void Cloning::computeDerivatives(const Mat& destination, const Mat &patch, const Mat &binaryMask)
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

void Cloning::scalarProduct(Mat mat, float r, float g, float b)
{
vector <Mat> channels;
split(mat, channels);
multiply(channels[2], r, channels[2]);
multiply(channels[1], g, channels[1]);
multiply(channels[0], b, channels[0]);
merge(channels, mat);
}

void Cloning::arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result) const
{
vector <Mat> lhs_channels;
vector <Mat> result_channels;

split(lhs, lhs_channels);
split(result, result_channels);

for (int chan = 0; chan < 3; ++chan)
multiply(lhs_channels[chan], rhs, result_channels[chan]);

merge(result_channels, result);
}

void Cloning::poisson(const Mat &destination)
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

void Cloning::evaluate(const Mat &I, const Mat &wmask, const Mat &cloned)
{
bitwise_not(wmask, wmask);

wmask.convertTo(binaryMaskFloatInverted, CV_32FC1, 1.0 / 255.0);

arrayProduct(destinationGradientX, binaryMaskFloatInverted, destinationGradientX);
arrayProduct(destinationGradientY, binaryMaskFloatInverted, destinationGradientY);

poisson(I);

merge(output, cloned);
}

void Cloning::normalClone(const Mat &destination, const Mat &patch, const Mat &binaryMask, Mat &cloned, int flag)
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
*/


