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
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
	CombinedCamera::combine_align(ldPixel, hdImage, ofGetWidth(), 1080, ofGetWidth() / 3, ofGetHeight() / 3, ofGetWidth() / 3, ofGetHeight() / 3);
	combinedImage.setFromPixels(CombinedCamera::combine_direct(ldPixel, hdImage, ofGetWidth(), 1080, ofGetWidth() / 3, ofGetHeight() / 3, ofGetWidth() / 3, ofGetHeight() / 3));
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
