#ifndef QUICKDEMO_H_ 
#define QUICKDEMO_H_ 

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;


class QuickDemo{
    public:
        void colorSpace_Demo(Mat &image);
        void mat_creation_demo();
        void pixel_visit_demo(Mat &image);
        void tracking_bar_demo(Mat &image);
        void color_style_demo(Mat &image);
        void bitwise_demo(Mat &image);
        void channels_demo(Mat &image);
        void inrange_demo(Mat &image);
        void pixel_statistic_demo(Mat &image);
        void drawing_demo(Mat &image);
        void random_drawing();
        void polyline_drawing_demo();
        void mouse_drawing_demo(Mat &image);
        void norm_demo(Mat &image);
        void resize_demo(Mat &image);
        void flip_demo(Mat &image);
        void rotate_demo(Mat &image);
        void video_demo(Mat &image);
        void histogram_demo(Mat &image);
		void histogram_2d_demo(Mat &image);
		void histogram_eq_demo(Mat &image);
		void blur_demo(Mat &image);
		void gaussian_blur_demo(Mat &image);
		void bifilter_demo(Mat &image);
		void face_detection_demo();

        void image_binaryzation(Mat &src);
        void image_binaryzation2(Mat &src);
        void image_connect_detect(Mat &src);
        void ccl_stats_demo(Mat &image);
        void Profile_demo(Mat &image);
        void nebula_demo(Mat &image);
        void histogram_demo2(Mat &image1, Mat &image2);
        void ORBBFMatcher(const cv::Mat& img1, const cv::Mat& img2, 
                      std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
                      cv::Mat& des1, cv::Mat& des2, 
                      std::vector<cv::DMatch>& goodMatches);




};

#endif
