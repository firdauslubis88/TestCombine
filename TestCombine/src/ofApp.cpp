#include "ofApp.h"

#include "CombinedCamera.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ldImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	hdImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	combinedImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);

	ldImage.load("ldImage_2.jpg");
	hdImage.load("hdImage_2.jpg");

	ldPixel.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	ldPixel = ldImage.getPixels();
	
	i = 0; j = 0;
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
	if (i == 0)
	{
		double start = (double)cv::getTickCount();
		CombinedCamera::combine_align(ldPixel, hdImage, ofGetWidth(), 1080, ofGetWidth() / 3, ofGetHeight() / 3, ofGetWidth() / 3, ofGetHeight() / 3);
		double end = (double)cv::getTickCount();
		double timeSpan = (end - start) / cv::getTickFrequency();
		std::cout << "GMS:\t" << timeSpan << std::endl;
		i++;
	}
	double start; double timeSpan;
	if (j == 0)
	{
		start = (double)cv::getTickCount();
	}
	combinedImage.setFromPixels(CombinedCamera::combine_direct(ldPixel, hdImage, ofGetWidth(), 1080, ofGetWidth() / 3, ofGetHeight() / 3, ofGetWidth() / 3, ofGetHeight() / 3));
	if (j == 0)
	{
		double end = (double)cv::getTickCount();
		timeSpan = (end - start) / cv::getTickFrequency();
		j++;
		std::cout << "setPixels:\t" << timeSpan << std::endl;
	}
	combinedImage.draw(0, 0, ofGetWidth(), 1080);

//	hdImage.draw(0, 0, ofGetWidth(), ofGetHeight());
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

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
