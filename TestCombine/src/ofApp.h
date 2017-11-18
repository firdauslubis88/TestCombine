#pragma once

#include "ofMain.h"
#include "ofxGui.h"

namespace lubis
{
	enum CAMSWITCH {
		NONE,
		LD,
		HD,
		COMBINE
	};
}

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		void onToggle(const void* sender);
		
		ofImage ldImage, hdImage, combinedImage;
		ofPixels ldPixel;

		ofxPanel gui;
		ofxIntSlider minHessianSlider, orbCountSlider;
		ofxButton siftButton, gmsButton;

		ofVideoGrabber ldVideoGrabber, hdVideoGrabber;

		lubis::CAMSWITCH camSwitch;
		int i, j, hdWidth, hdHeight;
};
