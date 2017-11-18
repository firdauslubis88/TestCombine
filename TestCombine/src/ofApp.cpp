#include "ofApp.h"

#include "CombinedCamera.h"

//#define LOGTEST

//--------------------------------------------------------------
void ofApp::setup(){
	hdWidth = 1280; hdHeight = 720;
	ldImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	hdImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	combinedImage.allocate(hdWidth, hdHeight, OF_IMAGE_COLOR);
	ldPixel.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);

	std::vector<ofVideoDevice> devices = ldVideoGrabber.listDevices();
	for each (ofVideoDevice device in devices)
	{
		if (device.deviceName == "USB2.0 HD UVC WebCam")
		{
			hdVideoGrabber.setDeviceID(device.id);
			hdVideoGrabber.setup(hdWidth, hdHeight);
		}
		else if (device.deviceName == "Creative GestureCam")
		{
			ldVideoGrabber.setDeviceID(device.id);
			ldVideoGrabber.setup(640, 320);
		}
	}
	
	minHessianSlider.setup("SIFT Hessian Value", 200, 50, 1000);
	orbCountSlider.setup("ORB count Value", 20000, 1000, 20000);
	siftButton.setup("SIFT");
	siftButton.addListener(this, &ofApp::onToggle);
	gmsButton.setup("GMS");
	gmsButton.addListener(this, &ofApp::onToggle);
	gui.setup();
	gui.add(&siftButton);
	gui.add(&minHessianSlider);
	gui.add(&gmsButton);
	gui.add(&orbCountSlider);

	camSwitch = lubis::NONE;
	i = 0; j = 0;
}

//--------------------------------------------------------------
void ofApp::update(){
	ldVideoGrabber.update();
	hdVideoGrabber.update();
	CombinedCamera::setSiftMinHessian(minHessianSlider);
	CombinedCamera::setOrbCount(orbCountSlider);
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
	else if (camSwitch == lubis::COMBINE)
	{
		ldPixel = ldVideoGrabber.getPixels();
		ldPixel.resize(hdWidth, hdHeight);
		hdImage.setFromPixels(hdVideoGrabber.getPixels());
		CombinedCamera::combine_align(ldPixel, hdImage, hdWidth, hdHeight, hdWidth/3, hdHeight/3, hdWidth/3, hdHeight/3);
		combinedImage.setFromPixels(CombinedCamera::combine_direct(ldPixel, hdImage, hdWidth, hdHeight, hdWidth/3, hdHeight/3, hdWidth/3, hdHeight/3));
		combinedImage.draw(0, 0, ofGetWidth(), ofGetHeight());
#ifdef LOGTEST
//		std::cout << "Combined Image Width():\t" << combinedImage.getWidth() << std::endl;
//		std::cout << "Combined Image Height():\t" << combinedImage.getHeight() << std::endl;
#endif
	}
#ifdef LOGTEST
	std::cout << ofGetFrameRate() << std::endl;
#endif
	gui.draw();
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
	if (key == 'r')
	{
		CombinedCamera::restartAligning();
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

void ofApp::onToggle(const void* sender)
{
	ofxButton* button = (ofxButton*)sender;
	if (button->getName() == "GMS")
	{
		CombinedCamera::setAlignmentMethod(lubis::GMS);
	}
	else if (button->getName() == "SIFT")
	{
		CombinedCamera::setAlignmentMethod(lubis::SIFT);
	}
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
