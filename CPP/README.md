这个目录中的各个测试demo 主要围绕 OpenCV展开：基础图像处理、噪声与滤波、视频处理、特征点检测匹配、目标定位和全景拼接。

quickdemo.cpp
这是一个 OpenCV 功能示例集合，里面实现了 QuickDemo 类的很多演示函数。包括颜色空间转换、像素访问、亮度/对比度滑条、伪彩色、通道分离、阈值分割、绘图、鼠标框选 ROI、归一化、缩放旋转、视频读取写入、直方图、滤波、人脸检测、二值化、连通域、轮廓检测、HSV 反向投影、ORB 匹配等。这个是在做测试的时候，写的一些公共工具demo 库，用来被其他文件调用。

test.cpp
主要演示图像加噪和高斯滤波。读取 home.jpg，转成灰度图，然后添加高斯噪声，再分别用 OpenCV 的 GaussianBlur 和手动构造的二维高斯核 filter2D 做滤波，对比系统高斯模糊和手写卷积核的效果。里面也有盐椒噪声函数，但主流程没用到。


target_detect.cpp
做基于 ORB 特征匹配的目标检测。读取 book.jpg 和 book_on_desk.jpg，在两张图中提取 ORB 特征，暴力匹配后保留前 15% 的优质匹配，再通过 findHomography 计算书本到场景图的映射关系，最后在 book_on_desk 中用红色四边形框出书的位置。

test_feature_detect.cpp
做 ORB 特征匹配的综合测试。读取 book.jpg 和 book_on_desk.jpg，提取 ORB 特征，用 BFMatcher(NORM_HAMMING) 匹配，然后按最小/最大距离筛选好匹配，还尝试用 RANSAC 过滤匹配点，最后又把 ORB 描述子转成 CV_32F 用 FlannBasedMatcher 做匹配对比。这个文件偏“比较 BF 和 FLANN 匹配效果”。

test_feature_orb.cpp
专门演示 ORB 特征检测与匹配。读取两张书本图片，画出 ORB 关键点，然后调用 QuickDemo::ORBBFMatcher 做 ORB 暴力匹配，最后用 drawMatches 显示匹配结果。比 test_feature_detect.cpp 更简洁，重点是 ORB + BFMatcher。

test_feature_sift.cpp
专门演示 SIFT 特征检测与匹配。使用 SIFT::create 提取关键点和描述子，用 BFMatcher(NORM_L2) 做 KNN 匹配，并通过 Lowe Ratio Test 筛选优质匹配。需要 opencv_xfeatures2d / nonfree 相关支持。

test_feature_surf.cpp
专门演示 SURF 特征检测与匹配。使用 SURF::create 提取特征，用 BFMatcher(NORM_L2) + KNN + Ratio Test 筛选匹配点，并显示 SURF 关键点和匹配结果。和 SIFT 文件结构很像，只是算法换成 SURF，同样依赖 xfeatures2d 和 nonfree 编译支持。

picture_joint.cpp
做两张图片的全景拼接。读取 q11.jpg 和 q22.jpg，用 AKAZE 提取特征点，暴力匹配加 Ratio Test 筛选匹配点，再用 findHomography + RANSAC 求单应矩阵，把右图透视变换到左图坐标系。后面还做了重叠区域的渐变权重融合，减少拼接缝。

test_video.cpp
做视频读取和颜色阈值分割实验。读取 panda.mp4，逐帧显示原始视频，把每帧转换到 HLS 色彩空间，然后用 inRange 按指定颜色范围生成 mask 并显示。作用类似视频中的颜色区域检测/分割 demo。

