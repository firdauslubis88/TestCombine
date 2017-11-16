#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>

#define THRESH_FACTOR 6

// 8 possible rotation and each one is 3 X 3 
const int mRotationPatterns[8][9] = {
	1,2,3,
	4,5,6,
	7,8,9,

	4,1,2,
	7,5,3,
	8,9,6,

	7,4,1,
	8,5,2,
	9,6,3,

	8,7,4,
	9,5,1,
	6,3,2,

	9,8,7,
	6,5,4,
	3,2,1,

	6,9,8,
	3,5,7,
	2,1,4,

	3,6,9,
	2,5,8,
	1,4,7,

	2,3,6,
	1,5,9,
	4,7,8
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };


class gms_matcher
{
public:
	// OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches 
	gms_matcher(const std::vector<cv::KeyPoint> &vkp1, const cv::Size size1, const std::vector<cv::KeyPoint> &vkp2, const cv::Size size2, const std::vector<cv::DMatch> &vDMatches);
	~gms_matcher() {};


private:

	// Normalized Points
	std::vector<cv::Point2f> mvP1, mvP2;

	// Matches
	std::vector<std::pair<int, int> > mvMatches;

	// Number of Matches
	size_t mNumberMatches;

	// Grid Size
	cv::Size mGridSizeLeft, mGridSizeRight;
	int mGridNumberLeft;
	int mGridNumberRight;


	// x	  : left grid idx
	// y      :  right grid idx
	// value  : how many matches from idx_left to idx_right
	cv::Mat mMotionStatistics;

	// 
	std::vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
	std::vector<int> mCellPairs;

	// Every Matches has a cell-pair 
	// first  : grid_idx_left
	// second : grid_idx_right
	std::vector<std::pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	std::vector<bool> mvbInlierMask;

	//
	cv::Mat mGridNeighborLeft;
	cv::Mat mGridNeighborRight;


public:

	// Get Inlier Mask
	// Return number of inliers 
	int GetInlierMask(std::vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);

private:

	// Normalize Key Points to Range(0 - 1)
	void NormalizePoints(const std::vector<cv::KeyPoint> &kp, const cv::Size &size, std::vector<cv::Point2f> &npts);

	// Convert OpenCV DMatch to Match (pair<int, int>)
	void ConvertMatches(const std::vector<cv::DMatch> &vDMatches, std::vector<std::pair<int, int> > &vMatches);

	int GetGridIndexLeft(const cv::Point2f &pt, int type);

	int GetGridIndexRight(const cv::Point2f &pt);

	// Assign Matches to Cell Pairs 
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
	std::vector<int> GetNB9(const int idx, const cv::Size& GridSize);

	//
	void InitalizeNiehbors(cv::Mat &neighbor, const cv::Size& GridSize);

	void SetScale(int Scale);

	// Run 
	int run(int RotationType);
};

// utility
inline cv::Mat DrawInlier(cv::Mat &src1, cv::Mat &src2, std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, std::vector<cv::DMatch> &inlier, int type) {
	const int height = std::max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	src1.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
			cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
			line(output, left, right, cv::Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
			cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
			cv::line(output, left, right, cv::Scalar(255, 0, 0));
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
			cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, cv::Scalar(0, 255, 255), 2);
			circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
		}
	}

	return output;
}

inline void imresize(cv::Mat &src, int height, double& ratio) {
	ratio = src.rows * 1.0 / height;
	int width = static_cast<int>(src.cols * 1.0 / ratio);
	resize(src, src, cv::Size(width, height));
}




