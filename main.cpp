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
#define PosSamNO_64 24   //����������
#define NegSamNO_64 536  //����������
#define TRAIN_64 true    //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
#define HardExampleNO_64 0
#define POS_PIC_NAME_64 "new_pos_64.txt"
#define NEG_PIC_NAME_64 "neg_64.txt"
#define POS_PIC_PATH_64 "samples/new_pos_64/"
#define NEG_PIC_PATH_64 "samples/neg_64/"

#define TEST_PIC "final_images/medium/"
#define RESULT_PIC "final_images/medium_test/"

string image_name = "24.jpg";

//�̳���CvSVM���࣬��Ϊ����setSVMDetector()���õ��ļ���Ӳ���ʱ����Ҫ�õ�ѵ���õ�SVM��decision_func������
//��ͨ���鿴CvSVMԴ���֪decision_func������protected���ͱ������޷�ֱ�ӷ��ʵ���ֻ�ܼ̳�֮��ͨ����������
class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

vector<Rect> svm_64(Mat &ret){
	//��ⴰ��(64,64),��ߴ�(8,8),�鲽��(4,4),cell�ߴ�(4,4),ֱ��ͼbin����12
	HOGDescriptor hog(Size(64, 64), Size(8, 8), Size(4, 4), Size(4, 4), 12);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������

	//��TRAINΪtrue������ѵ��������
	if (TRAIN_64)
	{
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos(POS_PIC_NAME_64);//������ͼƬ���ļ����б�
		//ifstream finPos("PersonFromVOC2012List.txt");//������ͼƬ���ļ����б�
		ifstream finNeg(NEG_PIC_NAME_64);//������ͼƬ���ļ����б�

		Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
		Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����


		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num < PosSamNO_64 && getline(finPos, ImgName); num++)
		{
			cout << "pos����" << ImgName << endl;
			//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//������������·����
			ImgName = POS_PIC_PATH_64 + ImgName;//������������·����
			Mat src = imread(ImgName);//��ȡͼƬ
			//if (CENTRAL_CROP)
			//	src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
			//resize(src,src,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(4, 4));//����HOG�����ӣ���ⴰ���ƶ�����(4,4)
			cout << "������ά����" << descriptors.size() << endl;

			//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
				//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO_64 + NegSamNO_64 + HardExampleNO_64, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
				sampleLabelMat = Mat::zeros(PosSamNO_64 + NegSamNO_64 + HardExampleNO_64, 1, CV_32FC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<NegSamNO_64 && getline(finNeg, ImgName); num++)
		{
			cout << "neg����" << ImgName << endl;
			ImgName = NEG_PIC_PATH_64 + ImgName;//���ϸ�������·����
			Mat src = imread(ImgName);//��ȡͼƬ
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(4, 4));//����HOG�����ӣ���ⴰ���ƶ�����(4,4)
			cout << "������ά����" << descriptors.size() << endl;

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO_64, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num + PosSamNO_64, 0) = -1;//���������Ϊ-1������
		}

		//����HardExample������
		/*if (HardExampleNO > 0)
		{
		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample���������ļ����б�
		//���ζ�ȡHardExample������ͼƬ������HOG������
		for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
		{
		cout << "����" << ImgName << endl;
		ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//����HardExample��������·����
		Mat src = imread(ImgName);//��ȡͼƬ
		//resize(src,img,Size(64,128));

		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		//cout<<"������ά����"<<descriptors.size()<<endl;

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
		sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
		sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������
		}
		}*/

		////���������HOG�������������ļ�
		//ofstream fout("SampleFeatureMat.txt");
		//for(int i=0; i<PosSamNO+NegSamNO; i++)
		//{
		//	fout<<i<<endl;
		//	for(int j=0; j<DescriptorDim; j++)
		//		fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		//	fout<<endl;
		//}

		//ѵ��SVM������
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "��ʼѵ��SVM������" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������
		cout << "ѵ�����" << endl;
		svm.save("SVM_HOG_64.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�

	}
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	{
		svm.load("SVM_HOG_64.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
	}

	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

	//��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
	//HOGDescriptor myHOG(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	ofstream fout("HOGDetectorForOpenCV_64.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}


	/**************����ͼƬ����HOG���˼��******************/
	//Mat src = imread("00000.jpg");
	//Mat src = imread("2007_000423.jpg");
	Mat src = imread(TEST_PIC + image_name);
	vector<Rect> found, found_filtered;//���ο�����
	cout << "���ж�߶�HOG���" << endl;
	hog.detectMultiScale(src, found, 0, Size(4, 4), Size(16, 16), 1.05, 2);//��ͼƬ���ж�߶����˼��
	cout << "�ҵ��ľ��ο������" << found.size() << endl;

	

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
	for (int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		if (r.width < 95 && r.y > 0 && r.x > 0){
			found_filtered.push_back(r);
		}
	}

	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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
	// ��ⴰ��(32,32),��ߴ�(4,4),�鲽��(2,2),cell�ߴ�(2,2),ֱ��ͼbin����12
	HOGDescriptor hog(Size(32, 32), Size(4, 4), Size(2, 2), Size(2, 2), 12);//HOG���������������HOG�����ӵ�

	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������

	//��TRAINΪtrue������ѵ��������
	if (TRAIN_32)
	{
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos(POS_PIC_NAME_32);//������ͼƬ���ļ����б�
		ifstream finNeg(NEG_PIC_NAME_32);//������ͼƬ���ļ����б�

		Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
		Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����


		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num < PosSamNO_32 && getline(finPos, ImgName); num++)
		{
			cout << "pos����" << ImgName << endl;
			ImgName = POS_PIC_PATH_32 + ImgName;//������������·����
			Mat src = imread(ImgName);//��ȡͼƬ
			//if (CENTRAL_CROP)
			//	src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
			//resize(src,src,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(4, 4));//����HOG�����ӣ���ⴰ���ƶ�����(4,4)
			cout << "������ά����" << descriptors.size() << endl;

			//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
				//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO_32 + NegSamNO_32 + HardExampleNO_32, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
				sampleLabelMat = Mat::zeros(PosSamNO_32 + NegSamNO_32 + HardExampleNO_32, 1, CV_32FC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<NegSamNO_32 && getline(finNeg, ImgName); num++)
		{
			cout << "neg����" << ImgName << endl;
			ImgName = NEG_PIC_PATH_32 + ImgName;//���ϸ�������·����
			Mat src = imread(ImgName);//��ȡͼƬ
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(4, 4));//����HOG�����ӣ���ⴰ���ƶ�����(4,4)
			cout << "������ά����" << descriptors.size() << endl;

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO_32, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num + PosSamNO_32, 0) = -1;//���������Ϊ-1������
		}

		//����HardExample������
		/*
		if (HardExampleNO > 0) {
		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample���������ļ����б�
		//���ζ�ȡHardExample������ͼƬ������HOG������
		for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++) {
		cout << "����" << ImgName << endl;
		ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//����HardExample��������·����
		Mat src = imread(ImgName);//��ȡͼƬ
		//resize(src,img,Size(64,128));

		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		//cout<<"������ά����"<<descriptors.size()<<endl;

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
		sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
		sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������
		}
		}
		*/

		////���������HOG�������������ļ�
		//ofstream fout("SampleFeatureMat.txt");
		//for(int i=0; i<PosSamNO+NegSamNO; i++)
		//{
		//	fout<<i<<endl;
		//	for(int j=0; j<DescriptorDim; j++)
		//		fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		//	fout<<endl;
		//}

		//ѵ��SVM������
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "��ʼѵ��SVM������" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������
		cout << "ѵ�����" << endl;
		cout << "����XML�ļ�" << endl;
		svm.save("SVM_HOG_32.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�

	}
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	{
		svm.load("SVM_HOG_32.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
	}

	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

	//��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
	//HOGDescriptor myHOG(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	cout << "�������Ӳ���" << endl;
	ofstream fout("HOGDetectorForOpenCV_32.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}


	/**************����ͼƬ����HOG���˼��******************/
	Mat src = imread(TEST_PIC + image_name, 0);
	vector<Rect> found, found_filtered;//���ο�����
	cout << "���ж�߶�HOG���" << endl;
	hog.detectMultiScale(src, found, 0.12, Size(2, 2), Size(16, 16), 1.05, 2);//��ͼƬ���ж�߶����˼��
	cout << "�ҵ��ľ��ο������" << found.size() << endl;

	

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
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

	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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
	// ���빫ʽΪd = |A*x0 + B*y0 + C|/��(A^2 + B^2)
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