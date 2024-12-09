/**
 * Python interface for python local feature detectors like:
 *  - SuperPoint: https://github.com/magicleap/SuperPointPretrainedNetwork
 *  desc:superpoint的python函数转换为c++接口
 */

#ifndef PYDETECTOR_H
#define PYDETECTOR_H

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>

#include <Interpreter/Python/PythonInterface.h>
#include <Python.h>

namespace rtabmap
{

class PyDetector
{
public:
	PyDetector(std::string pyPath);
	virtual ~PyDetector();

public:
	virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi, const cv::Mat & mask = cv::Mat());
	virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

private:
  PyObject * pModule_;
  PyObject * pFunc_;
  std::string path_;
  bool cuda_;
  cv::Mat descriptors_;
};

}

#endif
