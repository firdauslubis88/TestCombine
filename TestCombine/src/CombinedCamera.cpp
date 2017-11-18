#include "CombinedCamera.h"

ofxCvColorImage CombinedCamera::ldCvImage, CombinedCamera::hdCvImage, CombinedCamera::combinedCvImage, CombinedCamera::ldCvImage2, CombinedCamera::hdCvImage2, CombinedCamera::combinedCvImage2;
bool CombinedCamera::skipCloning = false, CombinedCamera::skipAligning = false, CombinedCamera::alreadyInitialized = false;


CombinedCamera::CombinedCamera(int image_width,int image_height)
{
	if (!CombinedCamera::alreadyInitialized)
	{
		ldCvImage.allocate(image_width, image_height);
		hdCvImage.allocate(image_width, image_height);
		combinedCvImage.allocate(image_width, image_height);
		ldCvImage2.allocate(image_width, image_height);
		hdCvImage2.allocate(image_width, image_height);
		combinedCvImage2.allocate(image_width, image_height);
		CombinedCamera::alreadyInitialized = true;
	}
}


CombinedCamera::~CombinedCamera()
{
}
/*
void CombinedCamera::seamlessClone(InputArray _src, InputArray _dst, InputArray _mask, Point p, OutputArray _blend, int flags)
{
	const Mat src = _src.getMat();
	const Mat dest = _dst.getMat();
	const Mat mask = _mask.getMat();
	_blend.create(dest.size(), CV_8UC3);
	Mat blend = _blend.getMat();

	int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
	int h = mask.size().height;
	int w = mask.size().width;

	Mat gray = Mat(mask.size(), CV_8UC1);
	Mat dst_mask = Mat::zeros(dest.size(), CV_8UC1);
	Mat cs_mask = Mat::zeros(src.size(), CV_8UC3);
	Mat cd_mask = Mat::zeros(dest.size(), CV_8UC3);

	if (mask.channels() == 3)
		cvtColor(mask, gray, COLOR_BGR2GRAY);
	else
		gray = mask;

	for (int i = 0; i<h; i++)
	{
		for (int j = 0; j<w; j++)
		{
			if (gray.at<uchar>(i, j) == 255)
			{
				minx = std::min(minx, i);
				maxx = std::max(maxx, i);
				miny = std::min(miny, j);
				maxy = std::max(maxy, j);
			}
		}
	}

	int lenx = maxx - minx;
	int leny = maxy - miny;

	Mat patch = Mat::zeros(Size(leny, lenx), CV_8UC3);

	int minxd = p.y - lenx / 2;
	int maxxd = p.y + lenx / 2;
	int minyd = p.x - leny / 2;
	int maxyd = p.x + leny / 2;

	CV_Assert(minxd >= 0 && minyd >= 0 && maxxd <= dest.rows && maxyd <= dest.cols);

	Rect roi_d(minyd, minxd, leny, lenx);
	Rect roi_s(miny, minx, leny, lenx);

	Mat destinationROI = dst_mask(roi_d);
	Mat sourceROI = cs_mask(roi_s);

	gray(roi_s).copyTo(destinationROI);
	src(roi_s).copyTo(sourceROI, gray(roi_s));
	src(roi_s).copyTo(patch, gray(roi_s));

	destinationROI = cd_mask(roi_d);
	cs_mask(roi_s).copyTo(destinationROI);


	Cloning obj;
	obj.normalClone(dest, cd_mask, dst_mask, blend, flags);

}
*/
ofPixels CombinedCamera::combine_direct(ofPixels ldPixel, ofImage hdImage, int image_width, int image_height, int x, int y, int width, int height)
{
	if (!CombinedCamera::alreadyInitialized)
	{
		ldCvImage.allocate(image_width, image_height);
		hdCvImage.allocate(image_width, image_height);
		combinedCvImage.allocate(image_width, image_height);
		ldCvImage2.allocate(image_width, image_height);
		hdCvImage2.allocate(image_width, image_height);
		combinedCvImage2.allocate(image_width, image_height);
		CombinedCamera::alreadyInitialized = true;
	}
	ofImage ldImage;

	/*How to integrate openFramework class with opencv class (from oF -> openCV):
	1. ofPixels -> ofImage
	2. ofImage -> ofxCvColorImage
	3. ofxCvColorImage -> Mat
	*/
	ldImage.setFromPixels(ldPixel);
	ldImage.setImageType(OF_IMAGE_COLOR);

	ldCvImage2.setFromPixels(ldImage.getPixels());
	hdCvImage2.setFromPixels(hdImage.getPixels());

	//Preparing (transfering) ofImage data type into ofxOpenCV image data type
	Mat tempMatHdCvImage = cvarrToMat(hdCvImage2.getCvImage());
	Mat tempMatLdCvImage = cvarrToMat(ldCvImage2.getCvImage());
	Mat source, target;
	Point cloneCenter;
	target = tempMatLdCvImage;

	//Aligning the images
	Mat aligned = Alignment::align_direct(tempMatLdCvImage, tempMatHdCvImage, x, y, width, height);
	aligned.copyTo(source);

	if (CombinedCamera::skipCloning)
	{
		source(Rect(x, y, width, height)).copyTo(target(Rect(x, y, width, height)));
		IplImage temp = target;
		IplImage* pTemp = &temp;
		combinedCvImage2 = pTemp;
	}
	else
	{
		//Stitching/blending the images
		Mat clone_mask, clone;
		clone_mask = Mat(tempMatLdCvImage.rows, tempMatLdCvImage.cols, CV_8UC1);
		clone_mask.setTo(Scalar(0));
		Rect clone_mask_ROI = Rect(x, y, width, height);
		clone_mask(clone_mask_ROI).setTo(Scalar(255));
		cloneCenter = Point(x + width / 2, y + height / 2);
		//seamlessClone(source, target, clone_mask, cloneCenter, clone, 1);
		mycv::Cloning::MVCSeamlessClone(source(Rect(x, y, width, height)), target, clone_mask, cloneCenter, clone);
		/*How to integrate openFramework class with opencv class (from openCV -> oF):
		1. Mat -> IplImage
		2. IplImage -> IplImage*
		3. IplImage* -> ofxCvColorImage
		*/
		IplImage temp = clone;
		IplImage* pTemp = &temp;
		combinedCvImage2 = pTemp;
	}

	return combinedCvImage2.getPixels();
}
/*
ofPixels CombinedCamera::combine(ofPixels ldPixel, ofImage hdImage, int image_width, int image_height, int x, int y, int width, int height)
{
	if (!CombinedCamera::alreadyInitialized)
	{
		ldCvImage.allocate(image_width, image_height);
		hdCvImage.allocate(image_width, image_height);
		combinedCvImage.allocate(image_width, image_height);
		ldCvImage2.allocate(image_width, image_height);
		hdCvImage2.allocate(image_width, image_height);
		combinedCvImage2.allocate(image_width, image_height);
		CombinedCamera::alreadyInitialized = true;
	}
	ofImage ldImage;

	ldImage.setFromPixels(ldPixel);
	ldImage.setImageType(OF_IMAGE_COLOR);

	ldCvImage.setFromPixels(ldImage.getPixels());
	hdCvImage.setFromPixels(hdImage.getPixels());

	//Preparing (transfering) ofImage data type into ofxOpenCV image data type
	Mat tempMatHdCvImage = cvarrToMat(hdCvImage.getCvImage());
	Mat tempMatLdCvImage = cvarrToMat(ldCvImage.getCvImage());
	Mat source, target;
	Point cloneCenter;
	target = tempMatLdCvImage;

	if (skipAligning)
	{
		tempMatHdCvImage.copyTo(source);
	}
	else
	{
		//Aligning the images
		Mat aligned = Alignment::align(tempMatLdCvImage, tempMatHdCvImage, x, y, width, height);
		aligned.copyTo(source);
	}
	//		tempMatHdCvImage(Rect(x, y, width, height)).copyTo(source); //Use this instead above if you want to skip alignment process

	if (skipCloning)
	{
		source(Rect(x, y, width, height)).copyTo(target(Rect(x, y, width, height)));
		IplImage temp = target;
		IplImage* pTemp = &temp;
		combinedCvImage = pTemp;
	}
	else
	{
		//Stitching/blending the images
		Mat clone_mask, clone;
		clone_mask = Mat(tempMatLdCvImage.rows, tempMatLdCvImage.cols, CV_8UC1);
		clone_mask.setTo(Scalar(0));
		Rect clone_mask_ROI = Rect(x, y, width, height);
		clone_mask(clone_mask_ROI).setTo(Scalar(255));
		cloneCenter = Point(x + width / 2, y + height / 2);
		//		seamlessClone(source, target, clone_mask, cloneCenter, clone, 1);
		Cloning::MVCSeamlessClone(source(Rect(x, y, width, height)), target, clone_mask, cloneCenter, clone);
		IplImage temp = clone;
		IplImage* pTemp = &temp;
		combinedCvImage = pTemp;
	}

	return combinedCvImage.getPixels();
}
*/
void CombinedCamera::combine_align(ofPixels ldPixel, ofImage hdImage, int image_width, int image_height, int x, int y, int width, int height)
{
	if (!CombinedCamera::alreadyInitialized)
	{
		ldCvImage.allocate(image_width, image_height);
		hdCvImage.allocate(image_width, image_height);
		combinedCvImage.allocate(image_width, image_height);
		ldCvImage2.allocate(image_width, image_height);
		hdCvImage2.allocate(image_width, image_height);
		combinedCvImage2.allocate(image_width, image_height);
		CombinedCamera::alreadyInitialized = true;
	}
	if (Alignment::alreadyChanged)return;
	ofImage ldImage;

	ldImage.setFromPixels(ldPixel);
	ldImage.setImageType(OF_IMAGE_COLOR);

	ldCvImage.setFromPixels(ldImage.getPixels());
	hdCvImage.setFromPixels(hdImage.getPixels());

	//Preparing (transfering) ofImage data type into ofxOpenCV image data type
	Mat tempMatHdCvImage = cvarrToMat(hdCvImage.getCvImage());
	Mat tempMatLdCvImage = cvarrToMat(ldCvImage.getCvImage());
	Mat source, target;
	Point cloneCenter;
	target = tempMatLdCvImage;
	float start = ofGetElapsedTimef();
	Alignment::align(tempMatLdCvImage, tempMatHdCvImage, x, y, width, height);
	float elapsed = ofGetElapsedTimef() - start;
	if (Alignment::AlignmentMethod == lubis::SIFT)
	{
		std::cout << "minHessian " << Alignment::minHessian << ":\t" << elapsed << std::endl;
	}
	else if (Alignment::AlignmentMethod == lubis::GMS)
	{
		std::cout << "orbCount " << Alignment::orbCount << ":\t" << elapsed << std::endl;
	}
	/*
	if (skipCloning)
	{
		source(Rect(x,y,width,height)).copyTo(target(Rect(x, y, width, height)));
		IplImage temp = target;
		IplImage* pTemp = &temp;
		combinedCvImage = pTemp;
	}
	else
	{
		//Stitching/blending the images
		Mat clone_mask, clone;
		clone_mask = Mat(tempMatLdCvImage.rows, tempMatLdCvImage.cols, CV_8UC1);
		clone_mask.setTo(Scalar(0));
		Rect clone_mask_ROI = Rect(x, y, width, height);
		clone_mask(clone_mask_ROI).setTo(Scalar(255));
		cloneCenter = Point(x + width / 2, y + height / 2);
//		seamlessClone(source, target, clone_mask, cloneCenter, clone, 1);
		Cloning::MVCSeamlessClone(source(Rect(x,y,width,height)), target, clone_mask, cloneCenter, clone);
		IplImage temp = clone;
		IplImage* pTemp = &temp;
		combinedCvImage = pTemp;
	}

	*outputPixels = combinedCvImage.getPixels();
	*/
}

void CombinedCamera::setSkipCloning(bool value)
{
	CombinedCamera::skipCloning = value;
}

void CombinedCamera::setSkipAligning(bool value)
{
	CombinedCamera::skipAligning = value;
}

void CombinedCamera::restartAligning()
{
	Alignment::alreadyChanged = false;
}

void CombinedCamera::setSiftMinHessian(int hess)
{
	Alignment::minHessian = hess;
}

void CombinedCamera::setOrbCount(int orb)
{
	Alignment::orbCount = orb;
}

void CombinedCamera::setAlignmentMethod(lubis::ALIGNMENT_METHOD meth)
{
	Alignment::AlignmentMethod = meth;
}

