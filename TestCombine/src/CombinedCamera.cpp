#include "CombinedCamera.h"

ofxCvColorImage CombinedCamera::ldCvImage, CombinedCamera::hdCvImage, CombinedCamera::combinedCvImage, CombinedCamera::ldCvImage2, CombinedCamera::hdCvImage2, CombinedCamera::combinedCvImage2;
bool CombinedCamera::skipCloning = false, CombinedCamera::skipAligning = false, CombinedCamera::alreadyInitialized = false;
double CombinedCamera::minHessianTime = 0.0f; double CombinedCamera::orbCountTime = 0.0f; double CombinedCamera::withoutCloningTime = 0.0f;  double CombinedCamera::withCloningTime = 0.0f;

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
#ifdef LOGTEST
	float start = ofGetElapsedTimef();
#endif
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
#ifdef LOGTEST
	float elapsed = ofGetElapsedTimef() - start;
	if (CombinedCamera::skipCloning)
	{
		//std::cout << "Merging View without cloning time:\t" << time << std::endl;
		CombinedCamera::withoutCloningTime = elapsed;
	}
	else
	{
		//std::cout << "Merging View with cloning time:\t" << time << std::endl;
		CombinedCamera::withCloningTime = elapsed;
	}
#endif
	return combinedCvImage2.getPixels();
}

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
#ifdef LOGTEST
	double start = ofGetElapsedTimef();
#endif
	Alignment::align(tempMatLdCvImage, tempMatHdCvImage, x, y, width, height);
#ifdef LOGTEST
	double elapsed = (ofGetElapsedTimef() - start);
	if (Alignment::AlignmentMethod == lubis::SIFT)
	{
		//std::cout << "minHessian time:\t" << Alignment::minHessian << ":\t" << elapsed << std::endl;
		CombinedCamera::minHessianTime = elapsed;
	}
	else if (Alignment::AlignmentMethod == lubis::GMS)
	{
		//std::cout << "orbCount time:\t" << Alignment::orbCount << ":\t" << elapsed << std::endl;
		CombinedCamera::orbCountTime = elapsed;
	}
#endif
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

