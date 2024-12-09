/**
 * Python interface for SuperGlue: https://github.com/magicleap/SuperGluePretrainedNetwork
 */

#include <Interpreter/Python/PyDetector.h>
#include <sys/stat.h>
#include <unistd.h>

#include <pybind11/embed.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>

#include <Interpreter/Utils/UtilsBase.h>
#include <Interpreter/Utils/UtilsLog.h>

namespace rtabmap
{

PyDetector::PyDetector(std::string pyPath) :
		pModule_(0),
		pFunc_(0),
		path_(pyPath),
		cuda_(false)
{
	std::cout << "pyPath=" << pyPath << ">" << std::endl;

	struct stat buffer;
	if (stat(pyPath.c_str(), &buffer) != 0){
		std::cout << "Cannot initialize Python detector, the path is not valid:" << pyPath << std::endl;
		return;
	}

	pybind11::gil_scoped_acquire acquire;
	
	std::string matcherPythonDir = UtilsBase::getDir(path_);
	if(!matcherPythonDir.empty())
	{
		PyRun_SimpleString("import sys");
		PyRun_SimpleString(UtilsBase::uFormat("sys.path.append(\"%s\")", matcherPythonDir.c_str()).c_str());
	}

	_import_array();

	std::string scriptName = UtilsBase::uSplit(UtilsBase::getName(path_), '.').front();
	PyObject * pName = PyUnicode_FromString(scriptName.c_str());
	std::cout << ("PyImport_Import() beg") << std::endl;
	pModule_ = PyImport_Import(pName);
	std::cout << ("PyImport_Import() end") << std::endl;

	Py_DECREF(pName);

	if(!pModule_)
	{
		std::cout << "Module " << scriptName << " could not be imported! File=" << path_ << std::endl;
		std::cout << getPythonTraceback() << std::endl;
	}
}

PyDetector::~PyDetector()
{
	pybind11::gil_scoped_acquire acquire;

	if(pFunc_)
	{
		Py_DECREF(pFunc_);
	}
	if(pModule_)
	{
		Py_DECREF(pModule_);
	}
}

std::vector<cv::KeyPoint> PyDetector::generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi, const cv::Mat & mask)
{
	std::cout << "try PyDetector::generateKeypointsImpl" << std::endl;
	descriptors_ = cv::Mat();
	assert(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
	// UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat imgRoi(image, roi);


	if(!pModule_)
	{
		std::cout << ("Python detector module not loaded!") << std::endl;
		return keypoints;
	}

	Timer timerAll("generateKeypointsImpl time");
	Timer timer;
	pybind11::gil_scoped_acquire acquire;
	if(!pFunc_)
	{

		PyObject * pFunc = PyObject_GetAttrString(pModule_, "init");
		if(pFunc)
		{
			if(PyCallable_Check(pFunc))
			{
				PyObject * result = PyObject_CallFunction(pFunc, "i", cuda_?1:0);

				if(result == NULL)
				{
					UERROR("Call to \"init(...)\" in \"%s\" failed!", path_.c_str());
					UERROR("%s", getPythonTraceback().c_str());
					return keypoints;
				}
				Py_DECREF(result);

				pFunc_ = PyObject_GetAttrString(pModule_, "detect");
				if(pFunc_ && PyCallable_Check(pFunc_))
				{
					// we are ready!
				}
				else
				{
					UERROR("Cannot find method \"detect(...)\" in %s", path_.c_str());
					UERROR("%s", getPythonTraceback().c_str());
					if(pFunc_)
					{
						Py_DECREF(pFunc_);
						pFunc_ = 0;
					}
					return keypoints;
				}
			}
			else
			{
				UERROR("Cannot call method \"init(...)\" in %s", path_.c_str());
				UERROR("%s", getPythonTraceback().c_str());
				return keypoints;
			}
			Py_DECREF(pFunc);
		}
		else
		{
			UERROR("Cannot find method \"init(...)\"");
			UERROR("%s", getPythonTraceback().c_str());
			return keypoints;
		}
	}
	timer.ticks("init");

	if(pFunc_)
	{
		npy_intp dims[2] = {imgRoi.rows, imgRoi.cols};
		PyObject* pImageBuffer = PyArray_SimpleNewFromData(2, dims, NPY_UBYTE, (void*)imgRoi.data);
		assert(pImageBuffer);

		timer.ticks("Preparing data");

		PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc_, pImageBuffer, NULL);
		if(pReturn == NULL)
		{
			UERROR("Failed to call match() function!");
			UERROR("%s", getPythonTraceback().c_str());
		}
		else
		{
			timer.ticks("Python detector");

			if (PyTuple_Check(pReturn) && PyTuple_GET_SIZE(pReturn) == 2)
			{
				PyObject *kptsPtr = PyTuple_GET_ITEM(pReturn, 0);
				PyObject *descPtr = PyTuple_GET_ITEM(pReturn, 1);
				if(PyArray_Check(kptsPtr) && PyArray_Check(descPtr))
				{
					PyArrayObject *arrayPtr = reinterpret_cast<PyArrayObject*>(kptsPtr);
					int nKpts = PyArray_SHAPE(arrayPtr)[0];
					int kptSize = PyArray_SHAPE(arrayPtr)[1];
					int type = PyArray_TYPE(arrayPtr);
					UDEBUG("Kpts array %dx%d (type=%d)", nKpts, kptSize, type);
					UASSERT(kptSize == 3);
					UASSERT_MSG(type == NPY_FLOAT, UtilsBase::uFormat("Returned matches should type FLOAT=11, received type=%d", type).c_str());

					float* c_out = reinterpret_cast<float*>(PyArray_DATA(arrayPtr));
					keypoints.reserve(nKpts);
					for (int i = 0; i < nKpts*kptSize; i+=kptSize)
					{
						cv::KeyPoint kpt(c_out[i], c_out[i+1], 8, -1, c_out[i+2]);
						keypoints.push_back(kpt);
					}

					arrayPtr = reinterpret_cast<PyArrayObject*>(descPtr);
					int nDesc = PyArray_SHAPE(arrayPtr)[0];
					UASSERT(nDesc = nKpts);
					int dim = PyArray_SHAPE(arrayPtr)[1];
					type = PyArray_TYPE(arrayPtr);
					UDEBUG("Desc array %dx%d (type=%d)", nDesc, dim, type);
					UASSERT_MSG(type == NPY_FLOAT, UtilsBase::uFormat("Returned matches should type FLOAT=11, received type=%d", type).c_str());

					c_out = reinterpret_cast<float*>(PyArray_DATA(arrayPtr));
					for (int i = 0; i < nDesc*dim; i+=dim)
					{
						cv::Mat descriptor = cv::Mat(1, dim, CV_32FC1, &c_out[i]).clone();
						descriptors_.push_back(descriptor);
					}
				}
			}
			else
			{
				UWARN("Expected tuple (Kpts 3 x N, Descriptors dim x N), returning empty features.");
			}
			Py_DECREF(pReturn);
		}
		Py_DECREF(pImageBuffer);
	}

	return keypoints;
}

cv::Mat PyDetector::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
{
	UASSERT((int)keypoints.size() == descriptors_.rows);
	return descriptors_;
}

}
