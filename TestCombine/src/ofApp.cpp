#include "ofApp.h"

#include "CombinedCamera.h"

//#define LOGTEST

//--------------------------------------------------------------
void ofApp::setup(){
	ofDisableArbTex();
	ofEnableDepthTest();

	hdWidth = 960; hdHeight = 540;
	ldImage.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	hdImage.allocate(hdWidth, hdHeight, OF_IMAGE_COLOR);
	combinedImage.allocate(hdWidth, hdHeight, OF_IMAGE_COLOR);
	ldPixel.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	fbo.allocate(ofGetWidth(), ofGetHeight());
	mesh = ofSpherePrimitive(2000, 24).getMesh();
	for (size_t i = 0; i < mesh.getNumNormals(); i++)
	{
		mesh.setNormal(i, mesh.getNormal(i) * ofVec3f(-1));
	}

	std::vector<ofVideoDevice> devices = ldVideoGrabber.listDevices();
	for each (ofVideoDevice device in devices)
	{
		if (device.deviceName == "PTZ Pro Camera")
		{
			hdVideoGrabber.setDeviceID(device.id);
			hdVideoGrabber.setup(hdWidth, hdHeight);
			hdCameraConnected = true;
		}
		else if (device.deviceName == "THETA UVC FullHD Blender")
		{
			ldVideoGrabber.setDeviceID(device.id);
			ldVideoGrabber.setup(1920, 960);
			ldCameraConnected = true;
		}
	}
	
	cam.setAspectRatio(16. / 9.);
	cam.setFov(90.);
	cam.rotate(90, 1, 0, 0);
	cam.rotate(90, 0, 0, 1);

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
	if (ldCameraConnected)
	{
		ldVideoGrabber.update();
	}
	if (hdCameraConnected)
	{
		hdVideoGrabber.update();
	}
	CombinedCamera::setSiftMinHessian(minHessianSlider);
	CombinedCamera::setOrbCount(orbCountSlider);
}

//--------------------------------------------------------------
void ofApp::draw(){
	if (ldCameraConnected)
	{
		fbo.begin();
		cam.begin();
		ofClear(0);
		ldVideoGrabber.bind();
		mesh.draw();
		ldVideoGrabber.unbind();
		cam.end();
		fbo.end();
	}
	if (camSwitch == lubis::LD)
	{
		fbo.draw(0, 0, ofGetWidth(), ofGetHeight());
	}
	else if (camSwitch == lubis::HD)
	{
		hdVideoGrabber.draw(0, 0, ofGetWidth(), ofGetHeight());
	}
	else if (camSwitch == lubis::COMBINE)
	{
		fbo.readToPixels(ldPixel);
		ldPixel.resize(hdWidth, hdHeight);
		hdImage.setFromPixels(hdVideoGrabber.getPixels());
		CombinedCamera::combine_align(ldPixel, hdImage, hdWidth, hdHeight, hdWidth/3, hdHeight/3, hdWidth/3, hdHeight/3);
		combinedImage.setFromPixels(CombinedCamera::combine_direct(ldPixel, hdImage, hdWidth, hdHeight, hdWidth/3, hdHeight/3, hdWidth/3, hdHeight/3));
		combinedImage.draw(0, 0, ofGetWidth(), ofGetHeight());

		ofDrawBitmapStringHighlight("Press 'q' for using clone", ofPoint(10, 120));
		ofDrawBitmapStringHighlight("Press 'w' for not using clone", ofPoint(10, 150));
		ofDrawBitmapStringHighlight("Press 'r' to restart aligning", ofPoint(10, 180));
#ifdef LOGTEST
		ofDrawBitmapStringHighlight("minHessianTime:\t\t" + std::to_string(CombinedCamera::minHessianTime), ofPoint(10, 210));
		ofDrawBitmapStringHighlight("orbCountTime:\t\t" + std::to_string(CombinedCamera::orbCountTime), ofPoint(10, 240));
		ofDrawBitmapStringHighlight("withoutCloningTime:\t" + std::to_string(CombinedCamera::withoutCloningTime), ofPoint(10, 270));
		ofDrawBitmapStringHighlight("withCloningTime:\t" + std::to_string(CombinedCamera::withCloningTime), ofPoint(10, 300));
#endif
		ofDrawBitmapStringHighlight("FPS:\t\t\t" + std::to_string(ofGetFrameRate()), ofPoint(10, 350));
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if (key == 'l' && ldCameraConnected)
	{
		camSwitch = lubis::LD;
	}
	if (key == 'h' && hdCameraConnected)
	{
		camSwitch = lubis::HD;
	}
	if (key == 'c' && ldCameraConnected && hdCameraConnected)
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
	if (key == 'w')
	{
		CombinedCamera::setSkipCloning(true);
	}
	if (key == 'q')
	{
		CombinedCamera::setSkipCloning(false);
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
