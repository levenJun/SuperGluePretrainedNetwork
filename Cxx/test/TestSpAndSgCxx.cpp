#include <Interpreter/Python/PyDetector.h>
#include <Interpreter/Python/PyMatcher.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <strings.h>
#include <iostream>

#include <memory>

std::shared_ptr<rtabmap::PythonInterface> pythonPtr; //构造函数中启动python解释器，析构函数释放python解释器

int main() {
    
    std::string imgPath[2] = {
        "../../assets/image_pairs/1403715273262142976.png",
        "../../assets/image_pairs/1403715310562142976.png"
    };
    std::string spScriptPath = "../../rtabmap_superpoint.py";
    std::string sgScriptPath = "../../rtabmap_superglue.py";
    cv::Mat img[2];
    for (int idx = 0; idx < 2; idx++)
    {
        img[idx] = cv::imread(imgPath[idx], cv::IMREAD_GRAYSCALE);
        if (img[idx].empty()) {
            std::cerr << "idx=," << idx << ",无法打开图像文件！" << std::endl;
            return -1;
        }
    }

    pythonPtr = std::make_shared<rtabmap::PythonInterface>();//在主线程中启动Python环境，对象析构时自动卸载python环境

    rtabmap::PyDetector mPyDetector(spScriptPath);

    rtabmap::PyMatcher mPyMatcher(sgScriptPath);

    std::vector<cv::KeyPoint> spKpts[2];
    cv::Mat spDescs[2];
    for (int idx = 0; idx < 2; idx++)
    {
        // 获取图像的宽度和高度
        int width = img[idx].cols;
        int height = img[idx].rows;

        // 使用图像的宽高创建一个 cv::Rect
        cv::Rect rect(0, 0, width, height);
        std::cout << "idx=," << idx << ",try generateKeypointsImpl" << std::endl;
        spKpts[idx] = mPyDetector.generateKeypointsImpl(img[idx], rect);
        spDescs[idx] = mPyDetector.generateDescriptorsImpl(img[idx], spKpts[idx]);
        std::cout << "idx=," << idx << ",spKpts size=," << spKpts[idx].size() << std::endl;

        // 在图像上绘制特征点
        cv::Mat img_with_keypoints;
        cv::cvtColor(img[idx], img_with_keypoints, cv::COLOR_GRAY2BGR);
        for (int pdx = 0; pdx < spKpts[idx].size(); pdx++)
        {
            cv::circle(img_with_keypoints, spKpts[idx][pdx].pt, 1, cv::Scalar(0, 255, 0), 1);
        }
        // cv::drawKeypoints(img[idx], spKpts, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // 显示结果
        std::string showTag = "SP Feature Points:" + std::to_string(idx);
        cv::namedWindow(showTag, cv::WINDOW_NORMAL);
        cv::imshow(showTag, img_with_keypoints);
        cv::waitKey(0); // 等待按键以关闭窗口
    }

    std::cout << "try match" << std::endl;

    cv::Size imgSize = cv::Size(img[0].cols, img[0].rows);
    std::vector<cv::DMatch> matchRes = mPyMatcher.match(spDescs[0], spDescs[1], spKpts[0], spKpts[1], imgSize);

    std::cout << "matchRes size=," << matchRes.size() << std::endl;

    std::vector<cv::KeyPoint> matchedKpts[2];
    std::vector<cv::DMatch> matchResPure;
    matchResPure.reserve(matchRes.size());
    for (size_t i = 0; i < matchRes.size(); ++i) {
        const auto& match = matchRes[i];
        matchedKpts[0].push_back(spKpts[0][match.queryIdx]);
        matchedKpts[1].push_back(spKpts[1][match.trainIdx]);
        // 更新匹配索引以适应新的特征点向量
        cv::DMatch newMatch(i, i, match.distance);
        matchResPure.push_back(newMatch);        
    }
    cv::Mat imgMatches;    
    cv::drawMatches(img[0], matchedKpts[0], img[1], matchedKpts[1], matchResPure, imgMatches);
    std::string showTag = "SG Feature Matchs";
    cv::namedWindow(showTag, cv::WINDOW_NORMAL);
    cv::imshow(showTag, imgMatches);
    cv::waitKey(0); // 等待按键以关闭窗口

    cv::destroyAllWindows();

    pythonPtr = nullptr;

    return 0;    
}