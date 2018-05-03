#include "commonHeaders.h"
#include "constants.h"
#include <vector>
#include <utility>
#include "EyeTracker.h"

using namespace Gorilla;

EyeTracker::EyeTracker()
{
}

void EyeTracker::Initialize()
{
    if(!face_cascade.load(face_cascade_name))
    {
        throw std::runtime_error("--(!)Error loading face cascade, please check face_cascade_name in source code");
    }
    std::pair<cv::Point, cv::Point> EyeCenters = GetEyeCenter();
    leftPupil = EyeCenters.first;
    rightPupil = EyeCenters.second;
}

std::pair<cv::Point, cv::Point> EyeTracker::GetEyeCenter(void)
{
#if CV_MAJOR_VERSION < 3
    CvCapture* capture = cvCaptureFromCAM(-1);
    if(capture)
        frame = cvQueryFrame(capture);
#else 
    cv::VideoCapture capture(-1);
    if(capture.isOpened())
        capture.read(frame);
#endif 

    cv::flip(frame, frame, 1);

    //detect faces
    std::vector<cv::Rect> faces;
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];

    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150));

    //findEyes
    if(faces.size() == 0)
        return std::make_pair(leftPupil, rightPupil);
    cv::Rect face = faces[0];
    cv::Mat faceROI = frame_gray(face);
    cv::Mat debugFace = faceROI;

    if(kSmoothFaceImage)
    {
        double sigma = kSmoothFaceFactor * face.width;
        GaussianBlur(faceROI, faceROI, cv::Size(0,0), sigma);
    }

    int eye_region_width = face.width * (kEyePercentWidth/100.0);
    int eye_region_height = face.height * (kEyePercentHeight/100.0);
    int eye_region_top = face.height * (kEyePercentTop/100.0);
    cv::Rect leftEyeRegion(face.width * (kEyePercentSide/100.0), eye_region_top, eye_region_width, eye_region_height);
    cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0), eye_region_top, eye_region_width, eye_region_height); 
    cv::Point leftPupilCenter = findEyeCenter(faceROI, leftEyeRegion);
    cv::Point rightPupilCenter = findEyeCenter(faceROI, rightEyeRegion);
    return std::make_pair(leftPupilCenter, rightPupilCenter);
}

void EyeTracker::GetDelta(std::mutex& m_mutex, std::pair<int,int>& delta)
{
    #if CV_MAJOR_VERSION < 3
    CvCapture* capture = cvCaptureFromCAM(-1);
    if(capture)
    {
        while(true)
        {
            frame = cvQueryFrame(capture);
#else 
    cv::VideoCapture capture(-1);
    if(capture.isOpened())
    {
        while(true)
        {
            capture.read(frame);
#endif 

            cv::flip(frame, frame, 1);

            //detect faces
            std::vector<cv::Rect> faces;
            std::vector<cv::Mat> rgbChannels(3);
            cv::split(frame, rgbChannels);
            cv::Mat frame_gray = rgbChannels[2];

            face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150));

            //findEyes
            if(faces.size() == 0)
            {

                std::lock_guard<std::mutex> lg(m_mutex);
                delta.first = 0;
                delta.second =  0;
                continue;
            }
            cv::Rect face = faces[0];
            cv::Mat faceROI = frame_gray(face);
            cv::Mat debugFace = faceROI;

            if(kSmoothFaceImage)
            {
                double sigma = kSmoothFaceFactor * face.width;
                GaussianBlur(faceROI, faceROI, cv::Size(0,0), sigma);
            }

            int eye_region_width = face.width * (kEyePercentWidth/100.0);
            int eye_region_height = face.height * (kEyePercentHeight/100.0);
            int eye_region_top = face.height * (kEyePercentTop/100.0);
            cv::Rect leftEyeRegion(face.width * (kEyePercentSide/100.0), eye_region_top, eye_region_width, eye_region_height);
            cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0), eye_region_top, eye_region_width, eye_region_height); 
            cv::Point leftPupilCenter = findEyeCenter(faceROI, leftEyeRegion);
            cv::Point rightPupilCenter = findEyeCenter(faceROI, rightEyeRegion);
            std::lock_guard<std::mutex> lg(m_mutex);
            delta.first = ((leftPupilCenter.x - leftPupil.x) + (rightPupilCenter.x - rightPupil.x)) / 2;
            delta.second =  ((rightPupilCenter.y - leftPupil.y)  + (rightPupilCenter.y - rightPupil.y)) / 2;
            leftPupil = leftPupilCenter;
            rightPupil = rightPupilCenter;

            //printf("%d,%d\n", delta.first, delta.second);
        }
    }
}

bool EyeTracker::rectInImage(cv::Rect rect, cv::Mat image)
{
    return rect.x > 0 && rect.y > 0 && rect.x+rect.width < image.cols &&
    rect.y+rect.height < image.rows;
}
bool EyeTracker::inMat(cv::Point p, int rows, int cols)
{
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}
cv::Mat EyeTracker::matrixMagnitude(const cv::Mat & matX, const cv::Mat & matY)
{
    cv::Mat mags(matX.rows,matX.cols,CV_64F);
    for (int y = 0; y < matX.rows; ++y) {
      const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
      double *Mr = mags.ptr<double>(y);
      for (int x = 0; x < matX.cols; ++x) {
        double gX = Xr[x], gY = Yr[x];
        double magnitude = sqrt((gX * gX) + (gY * gY));
        Mr[x] = magnitude;
      }
    }
    return mags;
}
double EyeTracker::computeDynamicThreshold(const cv::Mat & mat, double stdDevFactor)
{
    cv::Scalar stdMagnGrad, meanMagnGrad;
    cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}

void EyeTracker::scaleToFastSize(const cv::Mat &src, cv::Mat& dst)
{
    cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

cv::Mat EyeTracker::computeMatXGradient(const cv::Mat &mat) {
    cv::Mat out(mat.rows,mat.cols,CV_64F);
    
    for (int y = 0; y < mat.rows; ++y) {
      const uchar *Mr = mat.ptr<uchar>(y);
      double *Or = out.ptr<double>(y);
      
      Or[0] = Mr[1] - Mr[0];
      for (int x = 1; x < mat.cols - 1; ++x) {
        Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
      }
      Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
    }
    
    return out;
}

void EyeTracker::testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
    // for all possible centers
    for (int cy = 0; cy < out.rows; ++cy) {
      double *Or = out.ptr<double>(cy);
      const unsigned char *Wr = weight.ptr<unsigned char>(cy);
      for (int cx = 0; cx < out.cols; ++cx) {
        if (x == cx && y == cy) {
          continue;
        }
        // create a vector from the possible center to the gradient origin
        double dx = x - cx;
        double dy = y - cy;
        // normalize d
        double magnitude = sqrt((dx * dx) + (dy * dy));
        dx = dx / magnitude;
        dy = dy / magnitude;
        double dotProduct = dx*gx + dy*gy;
        dotProduct = std::max(0.0,dotProduct);
        // square and multiply by the weight
        if (kEnableWeight) {
          Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
        } else {
          Or[cx] += dotProduct * dotProduct;
        }
      }
    }
  }

cv::Point EyeTracker::findEyeCenter(cv::Mat face, cv::Rect eye)
{
    cv::Mat eyeROIUnscaled = face(eye);
    cv::Mat eyeROI;
    scaleToFastSize(eyeROIUnscaled, eyeROI);

    cv::Mat gradientX = computeMatXGradient(eyeROI);
    cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();

    cv::Mat mags = matrixMagnitude(gradientX, gradientY);

    double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);

    for(int y = 0; y < eyeROI.rows; ++y)
    {
        double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        const double *Mr = mags.ptr<double>(y);
        for(int x = 0; x < eyeROI.cols; ++x)
        {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = Mr[x];
            if(magnitude > gradientThresh)
            {
                Xr[x] = gX/magnitude;
                Yr[x] = gY/magnitude;
            }
            else
            {
                Xr[x] = 0.0;
                Yr[x] = 0.0;
            }
        }
    }

    cv::Mat weight;
    GaussianBlur(eyeROI, weight, cv::Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
    for(int y = 0; y < weight.rows; ++y)
    {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for(int x = 0; x < weight.cols; ++x)
        {
            row[x] = (255 - row[x]);
        }
    }

    cv::Mat outSum = cv::Mat::zeros(eyeROI.rows, eyeROI.cols, CV_64F);

    for(int y = 0; y < weight.rows; ++y)
    {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        for(int x = 0; x < weight.cols; ++x)
        {
            double gX = Xr[x], gY = Yr[x];
            if(gX == 0.0 && gY == 0.0)
            {
                continue;
            }
            testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
        }
    }

    double numGradients = (weight.rows * weight.cols);
    cv::Mat out;
    outSum.convertTo(out, CV_32F, 1.0f/numGradients);

    cv::Point maxP;
    double maxVal;
    cv::minMaxLoc(out, NULL, &maxVal, NULL, &maxP);

    if(kEnablePostProcess)
    {
        cv::Mat floodClone;
        double floodThresh = maxVal * kPostProcessThreshold;
        cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);

        cv::Mat mask = floodKillEdges(floodClone);
        cv::minMaxLoc(out, NULL, &maxVal, NULL, &maxP, mask);
    }
    return unscalePoint(maxP, eye);
}

cv::Point EyeTracker::unscalePoint(cv::Point p, cv::Rect origSize)
{
    float ratio = (((float)kFastEyeWidth)/origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return cv::Point(x,y);
}

bool EyeTracker::floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat)
{
    return inMat(np, mat.rows, mat.cols);
}
  
  // returns a mask
cv::Mat EyeTracker::floodKillEdges(cv::Mat &mat)
{
    rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
    
    cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
    std::queue<cv::Point> toDo;
    toDo.push(cv::Point(0,0));
    while (!toDo.empty()) {
      cv::Point p = toDo.front();
      toDo.pop();
      if (mat.at<float>(p) == 0.0f) {
        continue;
      }
      // add in every direction
      cv::Point np(p.x + 1, p.y); // right
      if (floodShouldPushPoint(np, mat)) toDo.push(np);
      np.x = p.x - 1; np.y = p.y; // left
      if (floodShouldPushPoint(np, mat)) toDo.push(np);
      np.x = p.x; np.y = p.y + 1; // down
      if (floodShouldPushPoint(np, mat)) toDo.push(np);
      np.x = p.x; np.y = p.y - 1; // up
      if (floodShouldPushPoint(np, mat)) toDo.push(np);
      // kill it
      mat.at<float>(p) = 0.0f;
      mask.at<uchar>(p) = 0;
    }
    return mask;
}