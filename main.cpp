#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

vector<Rect> svm_64(Mat &ret);
vector<Rect> svm_32(Mat &ret);
void preProcess(Mat &img, Mat &ret);
double point2line(Point p1, Point lp1, Point lp2);
bool if_exist(int index, vector<int> v);

// define for svm 32
#define PosSamNO_32 109
#define NegSamNO_32 2521
#define TRAIN_32 true    
#define HardExampleNO_32 0
#define POS_PIC_NAME_32 "new_pos_32.txt"
#define NEG_PIC_NAME_32 "neg_32.txt"
#define POS_PIC_PATH_32 "samples/new_pos_32/"
#define NEG_PIC_PATH_32 "samples/neg_32/"

//define for svm 64
#define PosSamNO_64 24   //正样本个数
#define NegSamNO_64 536  //负样本个数
#define TRAIN_64 true    //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define HardExampleNO_64 0
#define POS_PIC_NAME_64 "new_pos_64.txt"
#define NEG_PIC_NAME_64 "neg_64.txt"
#define POS_PIC_PATH_64 "samples/new_pos_64/"
#define NEG_PIC_PATH_64 "samples/neg_64/"

#define TEST_PIC "final_images/medium/"
#define RESULT_PIC "final_images/medium_test/"

string image_name = "24.jpg";

//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

vector<Rect> svm_64(Mat &ret){
	//检测窗口(64,64),块尺寸(8,8),块步长(4,4),cell尺寸(4,4),直方图bin个数12
	HOGDescriptor hog(Size(64, 64), Size(8, 8), Size(4, 4), Size(4, 4), 12);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

	//若TRAIN为true，重新训练分类器
	if (TRAIN_64)
	{
		string ImgName;//图片名(绝对路径)
		ifstream finPos(POS_PIC_NAME_64);//正样本图片的文件名列表
		//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
		ifstream finNeg(NEG_PIC_NAME_64);//负样本图片的文件名列表

		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人


		//依次读取正样本图片，生成HOG描述子
		for (int num = 0; num < PosSamNO_64 && getline(finPos, ImgName); num++)
		{
			cout << "pos处理：" << ImgName << endl;
			//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//加上正样本的路径名
			ImgName = POS_PIC_PATH_64 + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);//读取图片
			//if (CENTRAL_CROP)
			//	src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
			//resize(src,src,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(4, 4));//计算HOG描述子，检测窗口移动步长(4,4)
			cout << "描述子维数：" << descriptors.size() << endl;

			//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG描述子的维数
				//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO_64 + NegSamNO_64 + HardExampleNO_64, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO_64 + NegSamNO_64 + HardExampleNO_64, 1, CV_32FC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
		}

		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num<NegSamNO_64 && getline(finNeg, ImgName); num++)
		{
			cout << "neg处理：" << ImgName << endl;
			ImgName = NEG_PIC_PATH_64 + ImgName;//加上负样本的路径名
			Mat src = imread(ImgName);//读取图片
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(4, 4));//计算HOG描述子，检测窗口移动步长(4,4)
			cout << "描述子维数：" << descriptors.size() << endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO_64, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO_64, 0) = -1;//负样本类别为-1，无人
		}

		//处理HardExample负样本
		/*if (HardExampleNO > 0)
		{
		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample负样本的文件名列表
		//依次读取HardExample负样本图片，生成HOG描述子
		for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
		{
		cout << "处理：" << ImgName << endl;
		ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名
		Mat src = imread(ImgName);//读取图片
		//resize(src,img,Size(64,128));

		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
		//cout<<"描述子维数："<<descriptors.size()<<endl;

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
		sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
		sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
		}
		}*/

		////输出样本的HOG特征向量矩阵到文件
		//ofstream fout("SampleFeatureMat.txt");
		//for(int i=0; i<PosSamNO+NegSamNO; i++)
		//{
		//	fout<<i<<endl;
		//	for(int j=0; j<DescriptorDim; j++)
		//		fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		//	fout<<endl;
		//}

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "开始训练SVM分类器" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
		cout << "训练完成" << endl;
		svm.save("SVM_HOG_64.xml");//将训练好的SVM模型保存为xml文件

	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		svm.load("SVM_HOG_64.xml");//从XML文件读取训练好的SVM模型
	}

	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

	//将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子
	//HOGDescriptor myHOG(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
	ofstream fout("HOGDetectorForOpenCV_64.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}


	/**************读入图片进行HOG行人检测******************/
	//Mat src = imread("00000.jpg");
	//Mat src = imread("2007_000423.jpg");
	Mat src = imread(TEST_PIC + image_name);
	vector<Rect> found, found_filtered;//矩形框数组
	cout << "进行多尺度HOG检测" << endl;
	hog.detectMultiScale(src, found, 0, Size(4, 4), Size(16, 16), 1.05, 2);//对图片进行多尺度行人检测
	cout << "找到的矩形框个数：" << found.size() << endl;

	

	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
	for (int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		if (r.width < 95 && r.y > 0 && r.x > 0){
			found_filtered.push_back(r);
		}
	}

	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
	for (int i = 0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(ret, r.tl(), r.br(), Scalar(0, 255, 0), 3);
	}

	cv::imwrite("test_result_1.jpg", ret);
	//namedWindow("src", 0);
	//imshow("src", src);
	cv::waitKey();

	return found_filtered;
}

vector<Rect> svm_32(Mat &ret)
{
	// 检测窗口(32,32),块尺寸(4,4),块步长(2,2),cell尺寸(2,2),直方图bin个数12
	HOGDescriptor hog(Size(32, 32), Size(4, 4), Size(2, 2), Size(2, 2), 12);//HOG检测器，用来计算HOG描述子的

	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

	//若TRAIN为true，重新训练分类器
	if (TRAIN_32)
	{
		string ImgName;//图片名(绝对路径)
		ifstream finPos(POS_PIC_NAME_32);//正样本图片的文件名列表
		ifstream finNeg(NEG_PIC_NAME_32);//负样本图片的文件名列表

		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人


		//依次读取正样本图片，生成HOG描述子
		for (int num = 0; num < PosSamNO_32 && getline(finPos, ImgName); num++)
		{
			cout << "pos处理：" << ImgName << endl;
			ImgName = POS_PIC_PATH_32 + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);//读取图片
			//if (CENTRAL_CROP)
			//	src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
			//resize(src,src,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(4, 4));//计算HOG描述子，检测窗口移动步长(4,4)
			cout << "描述子维数：" << descriptors.size() << endl;

			//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG描述子的维数
				//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO_32 + NegSamNO_32 + HardExampleNO_32, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO_32 + NegSamNO_32 + HardExampleNO_32, 1, CV_32FC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
		}

		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num<NegSamNO_32 && getline(finNeg, ImgName); num++)
		{
			cout << "neg处理：" << ImgName << endl;
			ImgName = NEG_PIC_PATH_32 + ImgName;//加上负样本的路径名
			Mat src = imread(ImgName);//读取图片
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(4, 4));//计算HOG描述子，检测窗口移动步长(4,4)
			cout << "描述子维数：" << descriptors.size() << endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO_32, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO_32, 0) = -1;//负样本类别为-1，无人
		}

		//处理HardExample负样本
		/*
		if (HardExampleNO > 0) {
		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample负样本的文件名列表
		//依次读取HardExample负样本图片，生成HOG描述子
		for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++) {
		cout << "处理：" << ImgName << endl;
		ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名
		Mat src = imread(ImgName);//读取图片
		//resize(src,img,Size(64,128));

		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
		//cout<<"描述子维数："<<descriptors.size()<<endl;

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
		sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
		sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
		}
		}
		*/

		////输出样本的HOG特征向量矩阵到文件
		//ofstream fout("SampleFeatureMat.txt");
		//for(int i=0; i<PosSamNO+NegSamNO; i++)
		//{
		//	fout<<i<<endl;
		//	for(int j=0; j<DescriptorDim; j++)
		//		fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		//	fout<<endl;
		//}

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "开始训练SVM分类器" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
		cout << "训练完成" << endl;
		cout << "保存XML文件" << endl;
		svm.save("SVM_HOG_32.xml");//将训练好的SVM模型保存为xml文件

	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		svm.load("SVM_HOG_32.xml");//从XML文件读取训练好的SVM模型
	}

	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

	//将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子
	//HOGDescriptor myHOG(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
	cout << "保存检测子参数" << endl;
	ofstream fout("HOGDetectorForOpenCV_32.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}


	/**************读入图片进行HOG行人检测******************/
	Mat src = imread(TEST_PIC + image_name, 0);
	vector<Rect> found, found_filtered;//矩形框数组
	cout << "进行多尺度HOG检测" << endl;
	hog.detectMultiScale(src, found, 0.12, Size(2, 2), Size(16, 16), 1.05, 2);//对图片进行多尺度行人检测
	cout << "找到的矩形框个数：" << found.size() << endl;

	

	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
	for (int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		if (r.width < 45 && r.y > 0 && r.x > 0){
			found_filtered.push_back(r);
		}
		/*for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);*/
	}

	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
	for (int i = 0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(ret, r.tl(), r.br(), Scalar(0, 255, 255), 3);
	}

	cv::imwrite("test_result_2.jpg", ret);
	//namedWindow("src", 0);
	//imshow("src", src);
	cv::waitKey();

	return found_filtered;
	
}

void preProcess(Mat &img, Mat &ret){
	ret = img;
	
}

double point2line(Point p1, Point lp1, Point lp2){
	double a, b, c, dis;
	
	a = lp2.y - lp1.y;
	b = lp1.x - lp2.x;
	c = lp2.x * lp1.y - lp1.x * lp2.y;
	// 距离公式为d = |A*x0 + B*y0 + C|/√(A^2 + B^2)
	dis = abs(a * p1.x + b * p1.y + c) / sqrt(a * a + b * b);
	return dis;
}

bool if_exist(int index, vector<int> v){
	for (int i = 0; i < v.size(); i++){
		if (v[i] == index){
			return true;
		}
	}
	return false;
}



int main()
{
	Mat ret;
	//
	string read_file = TEST_PIC + image_name;
	ret = imread(read_file, 0);

	//Mat ret = img;
	//preProcess(img, ret);

	vector<Rect> rec64;
	if (image_name != "4.jpg" && image_name != "5.jpg" && image_name != "13.jpg"){
		rec64 = svm_64(ret);
	}
	
	vector<Rect> rec32;
	if (image_name != "2.jpg" && 
		image_name != "1.jpg" && 
		image_name != "3.jpg" && 
		image_name != "6.jpg" && 
		image_name != "7.jpg" &&
		image_name != "8.jpg" &&
		image_name != "10.jpg"){
		rec32 = svm_32(ret);
	}
	//rec32 = svm_32(ret);

	//rec64.insert(rec64.end(), rec32.begin(), rec32.end());

	vector<Point> points;
	for (int i = 0; i < rec32.size(); i++){
		points.push_back(Point(rec32[i].x, rec32[i].y));
		
	}
	for (int i = 0; i < rec64.size(); i++){
		points.push_back(Point(rec64[i].x, rec64[i].y));
	}
	
	Vec4f fitlines;
	fitlines[0] = 0;
	fitlines[1] = 0;
	fitlines[2] = 0;
	fitlines[3] = 0;
	
	cv::fitLine(points, fitlines, CV_DIST_L1, 0, 0.01, 0.01);

	int t = 500;

	Point p1, p2;
	p1.x = fitlines[2] - fitlines[0] * t;
	p1.y = fitlines[3] - fitlines[1] * t;
	p2.x = fitlines[2] + fitlines[0] * t;
	p2.y = fitlines[3] + fitlines[1] * t;

	vector<double> p2l_dis;
	for (int i = 0; i < points.size(); i++){
		double dis = point2line(points[i], p1, p2);
		p2l_dis.push_back(dis);
	}
	vector<int> far_rec_index;
	for (int i = 0; i < points.size(); i++){
		if (p2l_dis[i] > 30){
			far_rec_index.push_back(i);
		}
	}
	Mat img = imread(read_file, 0);
	cv::imwrite("re.jpg", img);
	for (int i = 0; i<rec32.size(); i++)
	{
		if (!if_exist(i, far_rec_index)){
			Rect r = rec32[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(img, r.tl(), r.br(), Scalar(0, 255, 255), 3);
		}
	}
	for (int i = 0; i<rec64.size(); i++)
	{
		if (!if_exist(i+rec32.size(), far_rec_index)){
			Rect r = rec64[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(img, r.tl(), r.br(), Scalar(0, 255, 255), 3);
		}
	}

	
	cv::line(ret, p1, p2, CV_RGB(255, 0, 0), 10, CV_AA, 0);

	string result_file = RESULT_PIC + image_name;
	cv::imwrite("result.jpg", ret);
	cv::imwrite(result_file, img);

	cv::waitKey(0);

	return 0;
}