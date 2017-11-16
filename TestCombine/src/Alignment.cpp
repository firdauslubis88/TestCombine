#include "Alignment.h"
//#define USE_SIFT_EXTRACTOR

bool Alignment::alreadyCreated = false;
bool Alignment::alreadyChanged = false;
bool Alignment::ptzAlreadyChanged = false;
Ptr<SURF> Alignment::detector = Ptr<SURF>();
int Alignment::minHessian = 200;
float Alignment::comparisonThreshold = 0.7;
Mat Alignment::hBig = Mat();
int Alignment::xReturn = 0;
int Alignment::yReturn = 0;
Ptr<Map> Alignment::mapPtr;
int Alignment::counter = 0;
float Alignment::widthRatio = 0;
float Alignment::heightRatio = 0;


Alignment::Alignment()
{
	comparisonThreshold = 0.7;
}

Alignment::Alignment(int minHessian)
{
	comparisonThreshold = 0.7;
}

#ifdef USE_PTZ_ADJUSTMENT
void Alignment::ptzAlign(const Mat refImage, const Mat inputImage, const int x, const int y, const int mask_width, const int mask_height)
{
	if (!Alignment::ptzAlreadyChanged)
	{
		if (!alreadyCreated)
		{
			Alignment::detector = SURF::create(Alignment::minHessian);
			alreadyCreated = true;
		}
		Alignment::xReturn = 0;
		Alignment::yReturn = 0;
		while (1)
		{
			bool doneX = false, doneY = false;
			cout << "Inside this" << endl;
			std::vector<KeyPoint> keypoints_ref, keypoints_input;
			Mat descriptors_ref, descriptors_input;
			Rect ROIRef = Rect(x, y, mask_width, mask_height);
			Mat ROIRefMask = Mat(inputImage.size(), CV_8UC1, Scalar::all(0));
			ROIRefMask(ROIRef).setTo(Scalar::all(255));

			detector->detectAndCompute(refImage, ROIRefMask, keypoints_ref, descriptors_ref);
			detector->detectAndCompute(inputImage, Mat(), keypoints_input, descriptors_input);
			Mat temp_img_kp;
			drawKeypoints(refImage, keypoints_ref, temp_img_kp, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

			FlannBasedMatcher matcher;
			vector<vector<DMatch>> kmatches;
			vector<DMatch> good_matches;

			matcher.knnMatch(descriptors_ref, descriptors_input, kmatches, 2);
			std::sort(kmatches.begin(), kmatches.end());
			int lenghtKMatches = kmatches.size();
			for (int i = 0; i < lenghtKMatches; i++)
			{
				double dist1 = kmatches[i][0].distance;
				double dist2 = kmatches[i][1].distance;
				double comp = dist1 / dist2;

				if (comp < comparisonThreshold)
				{
					good_matches.push_back(kmatches[i][0]);
				}
			}

			vector<Point2f> ptsTemp, ptsTemp2;
			for (int i = 0; i < good_matches.size(); i++)
			{
				Point3f pnt;
				//-- Get the keypoints from the good matches
				ptsTemp.push_back(keypoints_ref[good_matches[i].queryIdx].pt);
				ptsTemp2.push_back(keypoints_input[good_matches[i].trainIdx].pt);
			}
			vector<Point2f> ptsROI, ptsROI2;
			ptsROI = ptsTemp;
			ptsROI2 = ptsTemp2;

			Mat h;
			int localXReturn = 0, localYReturn = 0;
			Point2f inputMean, refMean, inputTotal = Point2f(0, 0), refTotal = Point2f(0, 0);
			if (ptsROI.size() >= 3)
			{
				for (int i = 0; i < ptsROI.size(); i++)
				{
					refTotal += ptsROI[i];
					inputTotal += ptsROI2[i];
				}
				refMean = refTotal / (int)ptsROI.size();
				inputMean = inputTotal / (int)ptsROI2.size();
				if (inputMean.x < refMean.x)
				{
					localXReturn = -1;
				}
				else
				{
					localXReturn = 1;
				}
				if (inputMean.y < refMean.y)
				{
					localYReturn = 1;
				}
				else
				{
					localYReturn = -1;
				}

				if (!doneX)
				{
					if ((abs(inputMean.x - refMean.x) < 200))
					{
						Alignment::xReturn = 0;
						doneX = true;
						cout << "DONE X" << endl;

					}
					else
					{
						Alignment::xReturn = localXReturn;
					}
				}
				if (!doneY)
				{
					if ((abs(inputMean.y - refMean.y) < 200))
					{
						Alignment::yReturn = 0;
						doneY = true;
						cout << "DONE Y" << endl;
					}
					else
					{
						Alignment::yReturn = localYReturn;
					}
				}
				if (doneX && doneY)
				{
					Alignment::ptzAlreadyChanged = true;
					cout << "DONE ALL" << endl;
					break;
				}
			}
		}
	}
}
#endif // USE_PTZ_ADJUSTMENT

void Alignment::align(Mat refImage, Mat inputImage, int x, int y, int mask_width, int mask_height)
{
	Mat outputImage;
	int countTest = 0;
	if (!Alignment::alreadyChanged && countTest < 5)
	{
		if (!alreadyCreated)
		{
			Alignment::detector = SURF::create(Alignment::minHessian);
			alreadyCreated = true;
		}
#ifdef USE_PTZ_ADJUSTMENT
		//		cout << "ALIGNING AGAIN" << endl;
		if (!Alignment::ptzAlreadyChanged)
		{
			Alignment::ptzAlign(refImage, inputImage, x, y, mask_width, mask_height);
		}
#endif // USE_PTZ_ADJUSTMENT
		double ratio;
		imresize(refImage, 360, ratio);
		imresize(inputImage, 360, ratio);

		std::vector<KeyPoint> keypoints_ref, keypoints_input;
		Mat descriptors_ref, descriptors_input;
		std::vector<DMatch> good_matches;
		Rect ROIRef = Rect(x/ratio, y/ratio, mask_width/ratio, mask_height/ratio);
		Mat ROIRefMask = Mat(inputImage.size(), CV_8UC1, Scalar::all(0));
		ROIRefMask(ROIRef).setTo(Scalar::all(255));

#ifdef USE_SIFT_EXTRACTOR
		detector->detectAndCompute(refImage, ROIRefMask, keypoints_ref, descriptors_ref);
		detector->detectAndCompute(inputImage, Mat(), keypoints_input, descriptors_input);

		FlannBasedMatcher matcher;
		vector<vector<DMatch>> kmatches;

		matcher.knnMatch(descriptors_ref, descriptors_input, kmatches, 2);
		std::sort(kmatches.begin(), kmatches.end());
		int lenghtKMatches = kmatches.size();
		for (int i = 0; i < lenghtKMatches; i++)
		{
			double dist1 = kmatches[i][0].distance;
			double dist2 = kmatches[i][1].distance;
			double comp = dist1 / dist2;

			if (comp < comparisonThreshold)
			{
				good_matches.push_back(kmatches[i][0]);
			}
		}
#else
		Ptr<ORB> orb = ORB::create(1000);
		orb->setFastThreshold(0);
		orb->detectAndCompute(refImage, ROIRefMask, keypoints_ref, descriptors_ref);
		orb->detectAndCompute(inputImage, Mat(), keypoints_input, descriptors_input);

		BFMatcher matcher(NORM_HAMMING);
		std::vector<DMatch> matches_all;
		matcher.match(descriptors_ref, descriptors_input, matches_all);

		// GMS filter
		int num_inliers = 0;
		std::vector<bool> vbInliers;
		gms_matcher gms(keypoints_ref, refImage.size(), keypoints_input, inputImage.size(), matches_all);
		num_inliers = gms.GetInlierMask(vbInliers, false, false);

//		std::cout << "Get total " << num_inliers << " matches." << std::endl;

		// draw matches
		for (size_t i = 0; i < vbInliers.size(); ++i)
		{
			if (vbInliers[i] == true)
			{
				good_matches.push_back(matches_all[i]);
			}
		}

#endif
		std::vector<Point2f> ptsTemp, ptsTemp2;
		for (int i = 0; i < good_matches.size(); i++)
		{
			Point3f pnt;
			//-- Get the keypoints from the good matches
			ptsTemp.push_back(keypoints_ref[good_matches[i].queryIdx].pt * ratio);
			ptsTemp2.push_back(keypoints_input[good_matches[i].trainIdx].pt * ratio);
		}
		std::vector<Point2f> ptsROI, ptsROI2;
		ptsROI = ptsTemp;
		ptsROI2 = ptsTemp2;

		Mat h;
		if (ptsROI.size() >= 3)
		{
			std::vector<Mat> ptsxy(2), pts2xy(2);
			Rect ROI, ROI2;
			double smallestX, largestX, smallestY, largestY;
			h = estimateRigidTransform(ptsROI2, ptsROI, false);
//			h = findHomography(ptsROI2, ptsROI, CV_RANSAC);
			if (!h.empty())
			{
				h.copyTo(Alignment::hBig);
				Alignment::alreadyChanged = true;
#ifdef ALIGNMENT_CHECK
				Mat img_matches;
				drawMatches(refImage, keypoints_ref, inputImage, keypoints_input,
					good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				std::ostringstream name;
				name << "bin/data/matches_im_" << Alignment::counter << ".jpg";
				imwrite(name.str(), img_matches);
				Alignment::counter++;
#endif // ALIGNMENT_CHECK

			}
		}
		countTest++;
	}
}

Mat Alignment::align_direct(Mat refImage, Mat inputImage, int x, int y, int mask_width, int mask_height)
{
	Mat outputImage;
	if (!Alignment::hBig.empty())
	{
		warpAffine(inputImage, outputImage, Alignment::hBig, Size(refImage.cols, refImage.rows));
//		warpPerspective(inputImage, outputImage, Alignment::hBig, Size(refImage.cols, refImage.rows));
	}
	else
	{
		outputImage = inputImage;
	}
	//	cout << Alignment::hBig << endl;
	return outputImage;
}

void Alignment::align_reglib(Mat refImage, Mat inputImage, int x, int y, int mask_width, int mask_height)
{
	if (!Alignment::alreadyChanged)
	{
#ifdef USE_PTZ_ADJUSTMENT
		//		cout << "ALIGNING AGAIN" << endl;
		if (!Alignment::ptzAlreadyChanged)
		{
			Alignment::ptzAlign(refImage, inputImage, x, y, mask_width, mask_height);
		}
#endif // USE_PTZ_ADJUSTMENT
		// Register using pixel differences
		Mat img1, img2;
		refImage.convertTo(img1, CV_8UC3);
		inputImage.convertTo(img2, CV_8UC3);

		MapperGradProj mapper;
		MapperPyramid mappPyr(mapper);
		//		Ptr<Map> mapPtr;
		Mat img3;
		img1.copyTo(img3);
		Rect ROIRef = Rect(x, y, mask_width, mask_height);
		Mat ROIRefMask = Mat(img1.size(), CV_8UC3, Scalar::all(0));
		ROIRefMask(ROIRef).setTo(Scalar::all(255));

		Mat img4;
		bitwise_and(img3, ROIRefMask, img4);
		img4.convertTo(img4, CV_64FC3, 1.0 / 255);
		namedWindow("IMG4");
		imshow("IMG4", img4);
		waitKey(0);
		inputImage.convertTo(img2, CV_64FC3, 1.0 / 255);
		//		img1(ROIRef).copyTo(img3);
		mappPyr.calculate(img2, img4, Alignment::mapPtr);

		// Print result
		//		MapProjec* mapProj2 = dynamic_cast<MapProjec*>(mapPtr.get());
		//		Alignment::mapProj = mapProj2;
		//		Alignment::mapProj->normalize();

		Alignment::alreadyChanged = true;
	}
}

Mat Alignment::align_direct_reglib(Mat refImage, Mat inputImage, int x, int y, int mask_width, int mask_height)
{
	Mat outputImage;
	MapProjec* mapProj = dynamic_cast<MapProjec*>(Alignment::mapPtr.get());
	if (mapProj != nullptr)
	{
		//		cout << mapProj->getProjTr() << endl;
		mapProj->normalize();
		mapProj->warp(inputImage, outputImage);
	}
	else
	{
		outputImage = inputImage;
	}
	//	cout << Alignment::hBig << endl;
	return outputImage;
}

Alignment::~Alignment()
{
}
