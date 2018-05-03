#ifndef __EYETRACKER_H__
#define __EYETRACKER_H__

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <mutex>
#include "constants.h"

namespace Gorilla
{
    class EyeTracker 
    {
    public:
        EyeTracker();
        void Initialize();
        std::pair<cv::Point,cv::Point> GetEyeCenter(void);
        void GetDelta(std::mutex& m_mutex, std::pair<int,int>& delta);
    
    private:
        bool rectInImage(cv::Rect rect, cv::Mat image);
        bool inMat(cv::Point p, int rows, int cols);
        cv::Mat matrixMagnitude(const cv::Mat & matX, const cv::Mat & matY);
        double computeDynamicThreshold(const cv::Mat & mat, double stdDevFactor);

        cv::Point findEyeCenter(cv::Mat face, cv::Rect eye);

        void scaleToFastSize(const cv::Mat &src, cv::Mat& dst);
        cv::Mat computeMatXGradient(const cv::Mat &mat);
        void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out);

        cv::Point unscalePoint(cv::Point p, cv::Rect origSize);
        cv::Mat floodKillEdges(cv::Mat &mat);
        bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat);

        cv::Point leftPupil;
        cv::Point rightPupil;
        cv::Mat frame;

        cv::String face_cascade_name = "/home/gorilla/Documents/RenderingSystem/MasterThesis/RenderingSystem/data/res/haarcascade_frontalface_alt.xml";
        cv::CascadeClassifier face_cascade;

    };

};

#endif //__EYETRACKER_H__