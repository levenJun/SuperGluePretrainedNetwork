#include <Interpreter/Python/PyDetector.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <strings.h>
#include <iostream>

#include <memory>

std::shared_ptr<rtabmap::PythonInterface> pythonPtr; //构造函数中启动python解释器，析构函数释放python解释器

int main() {
    
    std::string imgPath = "../../assets/image_pairs/1403715273262142976.png";
    std::string spScriptPath = "../../rtabmap_superpoint.py";
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "无法打开图像文件！" << std::endl;
        return -1;
    }    

    pythonPtr = std::make_shared<rtabmap::PythonInterface>();//在主线程中启动Python环境，对象析构时自动卸载python环境

    rtabmap::PyDetector mPyDetector(spScriptPath);

    // 获取图像的宽度和高度
    int width = img.cols;
    int height = img.rows;

    // 使用图像的宽高创建一个 cv::Rect
    cv::Rect rect(0, 0, width, height);
    std::cout << "try generateKeypointsImpl" << std::endl;
    std::vector<cv::KeyPoint> spKpts = mPyDetector.generateKeypointsImpl(img, rect);
    std::cout << "spKpts size=," << spKpts.size() << std::endl;

    // 在图像上绘制特征点
    cv::Mat img_with_keypoints;
    cv::drawKeypoints(img, spKpts, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);    

    // 显示结果
    cv::imshow("SP Feature Points", img_with_keypoints);
    cv::waitKey(0); // 等待按键以关闭窗口
    cv::destroyAllWindows();

    return 0;    
}