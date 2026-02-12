#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

string positive_dir = "F:/Test/example2/elec_watch/positive";
string negative_dir = "F:/Test/example2/elec_watch/negative";


void get_hog_descriptor(const Mat &image, vector<float> &desc);
void generate_dataset(Mat &trainData, Mat &labels);
void svm_train(Mat &trainData, Mat &labels);

int main(int argc, char** argv)
{
    // read data and generate dataset
	//Mat trainData = Mat::zeros(Size(3780, 26), CV_32FC1);
	//Mat labels = Mat::zeros(Size(1, 26), CV_32SC1);
	//generate_dataset(trainData, labels);

	// SVM train, and save model
	// svm_train(trainData, labels);

	// load model 

    Ptr<SVM> svm = SVM::load("F:/Test/example2/elec_watch/hog_elec.xml");

    Mat test = imread("F:/Test/example2/elec_watch/test/scene_02.jpg");
	resize(test, test, Size(0, 0), 0.2, 0.2);
	//imshow("input", test);
    //waitKey(0);
    Rect winRect;
	winRect.width = 64;
	winRect.height = 128;
	int sum_x = 0;
	int sum_y = 0;
	int count = 0;
    for(int row = 64; row < test.rows-64; row += 4)
    {
        for(int col = 32; col < test.cols-32; col += 4)
        {
            winRect.x = col - 32;
            winRect.y = row - 64;
            vector<float> fv;
            get_hog_descriptor(test(winRect), fv);
            Mat one_row = Mat::zeros(Size(fv.size(), 1), CV_32FC1);
			for (int i = 0; i < fv.size(); i++) {
				one_row.at<float>(0, i) = fv[i];
			}
			float result = svm->predict(one_row);
			if (result > 0) {
				// rectangle(test, winRect, Scalar(0, 0, 255), 1, 8, 0);
				count += 1;
				sum_x += winRect.x;
				sum_y += winRect.y;
			}
        }
    }

    winRect.x = sum_x / count;
	winRect.y = sum_y / count;
	rectangle(test, winRect, Scalar(255, 0, 0), 2, 8, 0);
	imshow("object detection result", test);
	imwrite("F:/Test/example2/case02.png", test);
	waitKey(0);
	return 0;
}


void get_hog_descriptor(const Mat &image, vector<float> &desc) {
    int h = image.rows;
	int w = image.cols;
	float rate = 64.0 / w;
    Mat img, gray;
    resize(image, img, Size(64, int(rate*h)));
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat result = Mat::zeros(Size(64,128), CV_8UC1);
    result = Scalar(127);
    Rect roi;                       // OpenCV的矩形类，用于标记图像的感兴趣区域（ROI）
    roi.x = 0;                      // ROI的左上角x坐标：0（水平方向靠左，因为宽已经是64，和画布宽一致）
    roi.width = 64;                 // ROI的宽度：64（和画布宽、缩放后图片宽一致）
    roi.y = (128 - gray.rows) / 2;  // ROI的左上角y坐标：垂直方向居中的核心计算
    roi.height = gray.rows;         // ROI的高度：和缩放后灰度图的高度一致
    gray.copyTo(result(roi));

    HOGDescriptor hog;
    hog.compute(result, desc, Size(8, 8), Size(0, 0));
}


void generate_dataset(Mat &trainData, Mat &labels)
{
    vector<string> temp_imgs;
    glob(positive_dir, temp_imgs);  //得到的正样本图片路径向量，每个元素是一张图片的完整路径
    if(temp_imgs.empty())
    {
        cerr << "The directory for positive samples is empty or the path is incorrect." << endl;
        return;
    }
    Mat temp_img = imread(temp_imgs[0 ].c_str()); //取向量中第一个元素，即正样本目录下的第一张图片的路径

    vector<float> temp_fv;
    get_hog_descriptor(temp_img, temp_fv);
    int feat_dim = temp_fv.size(); // HOG特征维度
    if(feat_dim == 0){
        cerr << "Error: HOG feature extraction failed, feature dimension is 0!" << endl;
        return;
    }

    // 统计正、负样本总数
    int pos_num = temp_imgs.size();
    temp_imgs.clear();
    glob(negative_dir, temp_imgs);
    int neg_num = temp_imgs.size();
    int total_num = pos_num + neg_num; // 训练集总样本数
    if (neg_num == 0) {
        cerr << "Warning: The negative sample directory is empty or the path is incorrect!" << endl;
        return;
    }

    // 初始化训练矩阵和标签矩阵（核心！必须提前分配内存+指定类型）
    trainData = Mat::zeros(total_num, feat_dim, CV_32F); // float型，总样本行×特征列
    labels = Mat::zeros(total_num, 1, CV_32S);           // int型，总样本行×1列（CV_32S对应int）
    temp_imgs.clear(); temp_img.release(); temp_fv.clear(); // 释放临时内存

    // 原有逻辑：处理正样本（标签=1）+ 新增异常校验
    vector<string> images;
    glob(positive_dir, images);
    for(size_t i=0; i<images.size(); i++)
    {
        Mat image = imread(images[i].c_str());
        if(image.empty()){
            cout << "error!" << endl;
            continue;
        }
        vector<float> fv;
        get_hog_descriptor(image, fv);
        if(fv.size() != feat_dim)
        {
            cerr << "Skip: The feature dimensions of the positive samples do not match → " << images[i] << endl;
            image.release();
            continue;
        }
        for(size_t j=0; j<fv.size(); j++)
        {
            trainData.at<float>(i,j) = fv[j];
        }
        // 正样本标签设为1
        labels.at<int>(i, 0) = 1;
        // 新增：释放临时图片内存，避免大数据集内存溢出
        image.release();
    }

    images.clear();
    glob(negative_dir, images);
    for (int i = 0; i < images.size(); i++) {
        Mat image = imread(images[i].c_str());
        // 新增：校验图片是否读取成功
        if (image.empty()) {
            cerr << "Skip: Failed to read negative sample image → " << images[i] << endl;
            continue;
        }
        vector<float> fv;
        get_hog_descriptor(image, fv);
        // 新增：校验特征是否提取成功（维度是否正确）
        if (fv.size() != feat_dim) {
            cerr << "Skip: Dimension mismatch of negative sample features → " << images[i] << endl;
            image.release();
            continue;
        }
        int row_idx = pos_num + i; // 负样本行索引（接在正样本后）
        // 特征值写入trainData
        for (int j = 0; j < fv.size(); j++) {
            trainData.at<float>(row_idx, j) = fv[j];
        }
        // 负样本标签设为-1
        labels.at<int>(row_idx, 0) = -1;
        // 新增：释放临时图片内存
        image.release();
    }

    // 新增：打印数据集信息，方便调试
    cout << "make dateset ok" << endl;
    cout << "Total sample size:" << total_num << "(Positive sample:" << pos_num << ",Negative sample:" << neg_num << ")" << endl;
    cout << "HOG feature dimension" << feat_dim << endl;
}


void svm_train(Mat &trainData, Mat &labels) {
	printf("\n start SVM training... \n");
	Ptr<SVM> svm = SVM::create();
	svm->setC(2.67);
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setGamma(5.383);
	svm->train(trainData, ROW_SAMPLE, labels);
	clog << "....[Done]" << endl;
	printf("end train...\n");

	// save xml
	svm->save("D:/images/train_data/elec_watch/hog_elec.xml");
}



//& "C:/Program Files/mingw64/bin/g++.exe" -g svm_test.cpp -o svm_test.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

//svm_test