#pragma once
#include "ofImage.h"
#include "ofxOpenCv.h"
#include "Alignment.h"
#include "Cloning.h"

using namespace cv;

class CombinedCamera
{
public:
	CombinedCamera() = default;
	CombinedCamera(int image_width, int image_height);
	~CombinedCamera();

//	void seamlessClone(InputArray _src, InputArray _dst, InputArray _mask, Point p, OutputArray _blend, int flags);

	static ofPixels combine_direct(ofPixels ldPixels, ofImage hdImage, int image_width, int image_height, int x, int y, int width, int height);
//	ofPixels combine(ofPixels ldPixels, ofImage hdImage, int image_width, int image_height, int x, int y, int width, int height);
	static void combine_align(ofPixels ldPixels, ofImage hdImage, int image_width, int image_height, int x, int y, int width, int height);
	static void setSkipCloning(bool value);
	static void setSkipAligning(bool value);
	static void restartAligning();

private:
	std::shared_ptr<ofxCvImage> combinedImage;
	static ofxCvColorImage ldCvImage, hdCvImage, combinedCvImage, ldCvImage2, hdCvImage2, combinedCvImage2;
	static bool skipCloning, skipAligning, alreadyInitialized;
};
