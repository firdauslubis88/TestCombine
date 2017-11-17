#include "ofApp.h"

#include "CombinedCamera.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ldImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	hdImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	combinedImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	ldPixel.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	ldPixel = ldImage.getPixels();

//	ldImage.load("ldImage_2.jpg");
//	hdImage.load("hdImage_2.jpg");

	std::vector<ofVideoDevice> devices = ldVideoGrabber.listDevices();
	for each (ofVideoDevice device in devices)
	{
		if (device.deviceName == "USB2.0 HD UVC WebCam")
		{
			hdVideoGrabber.setDeviceID(device.id);
			hdVideoGrabber.setup(1280, 720);
		}
		else if (device.deviceName == "Creative GestureCam")
		{
			ldVideoGrabber.setDeviceID(device.id);
			ldVideoGrabber.setup(640, 320);
		}
	}
	
	camSwitch = lubis::NONE;
	i = 0; j = 0;
}

//--------------------------------------------------------------
void ofApp::update(){
	ldVideoGrabber.update();
	hdVideoGrabber.update();
}

//--------------------------------------------------------------
void ofApp::draw(){
	if (camSwitch == lubis::LD)
	{
		ldVideoGrabber.draw(0, 0, ofGetWidth(), ofGetHeight());
	}
	else if (camSwitch == lubis::HD)
	{
		hdVideoGrabber.draw(0, 0, ofGetWidth(), ofGetHeight());
	}

//	if (i == 0)
//	{
//		double start = (double)cv::getTickCount();
//		CombinedCamera::combine_align(ldPixel, hdImage, ofGetWidth(), 1080, ofGetWidth() / 3, ofGetHeight() / 3, ofGetWidth() / 3, ofGetHeight() / 3);
//		double end = (double)cv::getTickCount();
//		double timeSpan = (end - start) / cv::getTickFrequency();
//		std::cout << "GMS:\t" << timeSpan << std::endl;
//		i++;
//	}
//	double start; double timeSpan;
//	if (j > -1)
//	{
//		start = (double)cv::getTickCount();
//	}
//	combinedImage.setFromPixels(CombinedCamera::combine_direct(ldPixel, hdImage, ofGetWidth(), 1080, ofGetWidth() / 3, ofGetHeight() / 3, ofGetWidth() / 3, ofGetHeight() / 3));
//	if (j > -1)
//	{
//		double end = (double)cv::getTickCount();
//		timeSpan = (end - start) / cv::getTickFrequency();
//		j++;
//		std::cout << "setPixels:\t" << timeSpan << std::endl;
//	}
//	combinedImage.draw(0, 0, ofGetWidth(), 1080);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if (key == 'l')
	{
		camSwitch = lubis::LD;
	}
	if (key == 'h')
	{
		camSwitch = lubis::HD;
	}
	if (key == 'c')
	{
		camSwitch = lubis::COMBINE;
	}
	if (key == 'n')
	{
		camSwitch = lubis::NONE;
	}
	{

	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
