
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <cstdlib>
#include <ctime>

#define NUM_OF_PROPERTIES 17

enum props {
    m00,
    m10,
    m01,
    m11,
    u00,
    u11,
    u20,
    u02,
    circf,
    mX,
    mY,
    Area,
    max,
    min,
    F1,
    F2,
    oClass // 0 - square, 1 - star, 2 - rectangle
};


cv::Mat colorIndexedImage(cv::Mat &src) {
    cv::Mat res = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    uchar index;
//    cv::Vec3b channels;
    //int random;
    for (int y = 0; y < src.rows; y++) {
//        random = rand() % 10;
        for (int x = 0; x < src.cols; x++) {
            index = src.at<uchar>(y,x);
//            channels = res.at<cv::Vec3b>(y,x)[0];
            
            res.at<cv::Vec3b>(y,x)[0] = (index * 30) % 255;
            res.at<cv::Vec3b>(y,x)[1] = (index * 10) % 255;
            res.at<cv::Vec3b>(y,x)[2] = (index * 5) % 255;
//            printf("(%u, %u, %u)", channels[0], channels[1], channels[2]);
        }
    }
    return res;
}

void recFloodFill(cv::Mat &src, cv::Mat &res, uchar &value, int y, int x) {
    if(src.at<uchar>(y, x) == 255 && res.at<uchar>(y, x) == 0 && y < src.rows && x < src.cols && y >= 0 && x>= 0 ) {
        res.at<uchar>(y, x) = value;
        recFloodFill(src, res, value, y, x+1);
        recFloodFill(src, res, value, y, x-1);
        recFloodFill(src, res, value, y+1, x);
        recFloodFill(src, res, value, y-1, x);
    }
    return;
}

// the result matrix mustn't be initialized when passed;
uchar floodFillBinaryImage(cv::Mat &src, cv::Mat &resultMatrix) {
    // initialize result matrix
    resultMatrix = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    uchar currIndx = 1;
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
//            printf("%u\t", src.at<uchar>(y, x));
            if(src.at<uchar>(y, x) == 255 && resultMatrix.at<uchar>(y, x) == 0) {
                recFloodFill(src, resultMatrix, currIndx, y, x);
                currIndx += 1;
            }
//            printf("%u,", resultMatrix.at<uchar>(y, x));
//
        }
//        printf("\n");

    }
    return currIndx - 1;
}


void thresholdImage(cv::Mat &src, cv::Mat &dst, double threshold, double max = 255) {
    if(src.type() != CV_8UC1)
        throw "Wrong type of matrix";
    
    dst.create(src.rows, src.cols, CV_8UC1);
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if(src.at<uchar>(y, x)  > threshold)
                dst.at<uchar>(y, x) = max;
            else
                dst.at<uchar>(y, x) = 0;
        }
    }
}




double moment(cv::Mat &src, int p, int q, int index) {
    if(src.type() != CV_8UC1)
        throw "Wrong type of matrix";
    
    double sum = 0;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if(src.at<uchar>(y, x) == index) {
                // x^p + y^q
                sum +=  pow(y, p) * pow(x, q);
            }
        }
    }
    return sum;
}

// moment related to the center of the mass
// so no matter where the object is positioned in the picture
// it won't affect results

double massMoment(cv::Mat &src, int p, int q, double mX, double mY, int index) {
    if(src.type() != CV_8UC1)
        throw "Wrong type of matrix";
    
    double sum = 0;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if(src.at<uchar>(y, x) == index) {

                sum +=  pow(y - mY, p) * pow(x - mX, q);
            }
        }
    }
    return sum;
}

int circumference(cv::Mat &src, int index) {
    if(src.type() != CV_8UC1)
        throw "Wrong type of matrix";
    
    uchar curr;
    int counter = 0;
    
    for (int y = 0; y <  src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            curr = src.at<uchar>(y, x);
            
            if (index == src.at<uchar>(y, x)) {
                // Compare with neighbors
                if (y > 0 && curr != src.at<uchar>(y - 1, x)) { // up
                    counter++;
                    continue;
                }
                if (y < src.rows - 1 && curr != src.at<uchar>(y + 1, x)) { // down
                    counter++;
                    continue;
                }
                if (x > 0 && curr != src.at<uchar>(y, x - 1 )) { // left
                    counter++;
                    continue;
                }
                if (x < src.cols - 1 && curr != src.at<uchar>(y, x + 1)) { // rigth
                    counter++;
                    continue;
                }
            }
        }
    }
    return counter;
}

double uMinMax(double *objects, bool max) {
    double q11 = objects[u11];
    double q20 = objects[u20];
    double q02 = objects[u02];
    if (max)
        return (1 / 2.) * (q20 + q02) + ((1 / 2.) * sqrt((4 * pow(q11, 2)) + pow((q20 - q02), 2 )));
    return (1 / 2.) * (q20 + q02) - ((1 / 2.) * sqrt((4 * pow(q11, 2)) + pow((q20 - q02), 2 )));
}


void calcProps(double **objects, int numOfObjects, cv::Mat &src) {
    double mx,my;
    for (int i = 0; i < numOfObjects; i++) {
        // moments
        objects[i][m00] = moment(src, 0, 0, i + 1);
        objects[i][m10] = moment(src, 1, 0, i + 1);
        objects[i][m01] = moment(src, 0, 1, i + 1);
        objects[i][m11] = moment(src, 1, 1, i + 1);
        
        // centres of masses
        objects[i][mY] = objects[i][m10] / objects[i][m00];
        objects[i][mX] = objects[i][m01] / objects[i][m00];
        
        // mass moments
        mx = objects[i][mX];
        my = objects[i][mY];
        objects[i][u00] = massMoment(src, 0, 0, mx, my, i + 1);
        objects[i][u11] = massMoment(src, 1, 1, mx, my, i + 1);
        objects[i][u02] = massMoment(src, 0, 2, mx, my, i + 1);
        objects[i][u20] = massMoment(src, 2, 0, mx, my, i + 1);
        
        // areas
        objects[i][Area] = objects[i][u00];
        
        // circumferences
        objects[i][circf] = circumference(src, i + 1);
        
        // max and min
        objects[i][max] = uMinMax(objects[i], true);
        objects[i][min] = uMinMax(objects[i], false);
        
        // F1
        objects[i][F1] = pow(objects[i][circf], 2) / (100 * objects[i][Area]);
        // F2
        objects[i][F2] = objects[i][min] / objects[i][max];
    }

}

cv::Point findObject(cv::Mat &src, int index) {
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if(src.at<uchar>(y, x) == index) {
                return cv::Point(x-7,y+10);
            }
        }
    }
    return cv::Point(0,0);
}

void printProps(double **objects, int numOfObjects, cv::Mat &image, cv::Mat &src) {
    for (int i = 0; i < numOfObjects; i++) {
        std::cout << "Object[" << i + 1 << "]" << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "M00 = " << objects[i][m00] << std::endl;
        std::cout << "M11 = " << objects[i][m11] << std::endl;
        std::cout << "M10 = " << objects[i][m10] << std::endl;
        std::cout << "M01 = " << objects[i][m01] << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "U00 = " << objects[i][u00] << std::endl;
        std::cout << "U11 = " << objects[i][u11] << std::endl;
        std::cout << "U02 = " << objects[i][u02] << std::endl;
        std::cout << "U20 = " << objects[i][u20] << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "XT = " << objects[i][mX] << std::endl;
        std::cout << "YT = " << objects[i][mY] << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "Area = " << objects[i][Area] << std::endl;
        std::cout << "Circumference = " << objects[i][circf] << std::endl;
        std::cout << "Min = " << objects[i][min] << std::endl;
        std::cout << "Max = " << objects[i][max] << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "F1 = " << objects[i][F1] << std::endl;
        std::cout << "F2 = " << objects[i][F2] << std::endl;
        std::cout << "point = (" << objects[i][F1] <<", " << objects[i][F2] << ")"<< std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "Class = " << objects[i][oClass] << std::endl;

        std::cout << std::endl;
        cv::Point org = findObject(src, i + 1);
        cv::putText(image, "Obj: " + std::to_string(i + 1), org, cv::FONT_HERSHEY_SIMPLEX, 0.3,cv::Scalar(0, 0, 255), 0.05);

    }
}



void assignClassTrainImage(double **objects, int numOfObjects) {
    // function only works for train.png
    // to manually assign classes
    
    // squares
    for (int i = 0 ; i < 4; i++) {
        objects[i][oClass] = 0;
    }
    // stars
    for (int i = 4 ; i < 8; i++) {
        objects[i][oClass] = 1;
    }
    // rectangle
    for (int i = 8 ; i < numOfObjects; i++) {
        objects[i][oClass] = 2;
    }
}
double calculateDistance(const cv::Point2d& p1, const cv::Point2d& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return sqrt(dx * dx + dy * dy);
}

cv::Point2d* computeEtalons(double **objects, int numOfObjects, int numOfClasses){
    double f1Sum = 0;
    double f2Sum = 0;
    int classCounter = 0;
    cv::Point2d *etalons = new cv::Point2d[numOfClasses];
    // squares
    //classes should be ordered from 0...n
    for (int classC = 0 ; classC < numOfClasses; classC++) {
        f1Sum = 0;
        f2Sum = 0;
        classCounter = 0;
        for (int i = 0; i < numOfObjects; i++) {
            if(objects[i][oClass] == (int) classC) {
                f1Sum += objects[i][F1];
                f2Sum += objects[i][F2];
                classCounter += 1;
            }
        }
        etalons[classC].x = f1Sum / (double) classCounter;
        etalons[classC].y = f2Sum / (double) classCounter;
        std::cout << "Class " << classC << " etalon x = (" << etalons[classC].x << ", " << etalons[classC].y << ")" << std::endl;

        std::cout << "\n--------------------\n";
    }
    return etalons;
}

void assignClassToUnknownObj(double **objects, int numOfObjects, int numOfClasses, cv::Point2d *etalons) {
    double d = DBL_MAX;
    int min_class = -1;
    double res;
    cv::Point2d a;
    cv::Point2d b;
    for (int i = 0; i < numOfObjects; i++) {
        d = DBL_MAX;
        // find min distance
        for(int j = 0; j < numOfClasses; j++) {
            a = cv::Point2d(objects[i][F1],objects[i][F2]);
            b = etalons[j];
            res = calculateDistance(a,b);
            if(res < d) {
                d = res;
                min_class = j;
            }
        }
        objects[i][oClass] = min_class;
        std::cout << "unknown Object[" << i + 1 << "] has class " << min_class << std::endl;
    }
}

void initializeCentroids(double **objects, int numOfObjects,cv::Point2d *centroids, int k) {
    int index;
    for (int i = 0; i < k; i++) {
        index = rand() % (numOfObjects);
        centroids[i] = cv::Point2d(objects[index][F1],objects[index][F2]);
    }
}

cv::Point2d* kMeansClustering(double **objects, int numOfObjects, int k){
    double f1Sum;
    double f2Sum;
    double cCounter;
    double newX;
    double newY;
    bool keepRunning = true;
    cv::Point2d a;
    cv::Point2d b;
    double dis;
    double min_dis = DBL_MAX;
    int classToBeAssigned = -1;
    double threshold = 1e-3;
    double delta;
    
    cv::Point2d *centroids = new cv::Point2d[k];
    initializeCentroids(objects, numOfObjects, centroids, k);
    
    while(keepRunning){
        keepRunning = false;
        //assign class to objects
        for (int i = 0 ; i < numOfObjects; i++) {
            min_dis = DBL_MAX;
            classToBeAssigned = -1;
            a = cv::Point2d(objects[i][F1],objects[i][F2]);
            for (int j = 0; j < k; j++) {
                b = centroids[j];
                dis = calculateDistance(a,b);
                if(dis < min_dis) {
                    min_dis = dis;
                    classToBeAssigned = j;
                }
            }
            objects[i][oClass] = classToBeAssigned;
        }
        // update centroids
        for (int cl = 0 ; cl < k; cl++) {
            f1Sum = 0;
            f2Sum = 0;
            cCounter = 0;
            for (int i = 0; i < numOfObjects; i++) {
                if(objects[i][oClass] == cl) {
                    f1Sum += objects[i][F1];
                    f2Sum += objects[i][F2];
                    cCounter += 1;
                }
            }
            newX = f1Sum / cCounter;
            newY = f2Sum / cCounter;
            delta = calculateDistance(cv::Point2d(newX,newY), centroids[cl]);
            if(delta > threshold)
                keepRunning = true;
            centroids[cl].x = f1Sum / cCounter;
            centroids[cl].y = f2Sum / cCounter;
        }

    }
    // print centroids
    std::cout << "K-means clustering" << std::endl;
    for (int i = 0; i < k; i++) {
        std::cout << "Class " << i << " etalon x = (" << centroids[i].x << ", " << centroids[i].y << ")" << std::endl;

        std::cout << "\n--------------------\n";
    }
    return centroids;
}

void printFormated(double **objects, int N) {
    std::cout << "{";
    for (int i = 0; i < N; i++) {
        std::cout << "{" << objects[i][F1] << "," <<objects[i][F2] << ",";
        switch ((int)objects[i][oClass]) {
            case 0:
                std::cout << 1 << "," << 0 << "," << 0 << "}," << std::endl;
                break;
            case 1:
                std::cout << 0 << "," << 1 << "," << 0 << "}," << std::endl;
                break;
            case 2:
                std::cout << 0 << "," << 0 << "," << 1 << "}," << std::endl;
                break;

            default:
                break;
        }
    }
    std::cout << "}";

}

int main()
{
    srand(time(0));
    cv::Mat src_8uc1_img = cv::imread( "../images/train.png", cv::IMREAD_GRAYSCALE );

    cv::Mat src_8uc1_img_2 = cv::imread( "../images/test02.png", cv::IMREAD_GRAYSCALE );

    if (src_8uc1_img.empty()) {
        printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
        return -1;
    }

    cv::Mat binary_8uc1_img;
    cv::Mat binary_8uc1_img_2;
    cv::Mat indexed_8uc1_img;
    cv::Mat indexed_8uc1_img_2;

    double threshold = 50;
    double max = 255;
    
    thresholdImage(src_8uc1_img, binary_8uc1_img, threshold, max); // to binary
    thresholdImage(src_8uc1_img_2, binary_8uc1_img_2, threshold, max); // to binary


    int numOfObjects = (int) floodFillBinaryImage(binary_8uc1_img, indexed_8uc1_img);
    int numOfObjects_2 = (int) floodFillBinaryImage(binary_8uc1_img_2, indexed_8uc1_img_2);

    // 2d array that will hold values neededed to calc features for every object
    // m00, m10, m01, m11, u00, u11, u20, u02 cricumference, centerOfMass, Area, F1, F2...
    
    // train image
    double **objects = new double*[numOfObjects];
    for (int i = 0; i < numOfObjects; i++) {
        objects[i] = new double[NUM_OF_PROPERTIES];
    }
    
    
    // test image
    double **objects_2 = new double*[numOfObjects_2];
    for (int i = 0; i < numOfObjects_2; i++) {
        objects_2[i] = new double[NUM_OF_PROPERTIES];
    }
    
    cv::Mat colored_8uc3_img = colorIndexedImage(indexed_8uc1_img);
    cv::Mat colored_8uc3_img_2 = colorIndexedImage(indexed_8uc1_img_2);


    //props for train image
    //get centroids for each class using kmeans from train image
    calcProps(objects, numOfObjects, indexed_8uc1_img);
    cv::Point2d* centroids = kMeansClustering(objects, numOfObjects, 3);
    printProps(objects, numOfObjects, colored_8uc3_img, indexed_8uc1_img);

    // props for test image
    calcProps(objects_2, numOfObjects_2, indexed_8uc1_img_2);
    // assign class to test image objects based on centroids computed from train image
    assignClassToUnknownObj(objects_2, numOfObjects_2, 3, centroids);
    printProps(objects_2, numOfObjects_2, colored_8uc3_img_2, indexed_8uc1_img_2);
    
    cv::imshow("Colored", colored_8uc3_img);
    cv::imshow("Colored 2", colored_8uc3_img_2);

    // printFormated(objects, numOfObjects);
    // std::cout << "\nObject 2\n";
    // printFormated(objects_2, numOfObjects_2);

    cv::waitKey( 0 ); // wait until keypressed
    
    for(int i = 0; i < numOfObjects; ++i) {
      delete[] objects[i];
    }
    for(int i = 0; i < numOfObjects_2; ++i) {
      delete[] objects_2[i];
    }
    delete[] objects;
    delete[] objects_2;
    delete centroids;
//    delete etalons;

    return 0;
}

