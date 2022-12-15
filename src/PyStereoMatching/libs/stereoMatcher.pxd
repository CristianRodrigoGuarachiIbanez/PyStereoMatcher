# distutils: language = c++
from openCVModuls cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "../../methods/sad.hpp" namespace "sad":
    cdef cppclass SAD:
        #public:
        SAD() except +
        SAD(int _radius, int _disp) except +
        Mat do_match(Mat &leftImage, Mat &rightImage);

        #private:
        int radius #kernel size
        int disp #search window size


cdef extern from "../../methods/bp.hpp" namespace "bp":
    cdef cppclass BP:
        #private:
        int height;
        int width;
        int disp;
        int iter;
        Mat leftImg;
        Mat rightImg;
        Mat smoothCostMat;
        vector[vector[Mat]] msg;
        vector[vector[Mat]] obs;

        #public:
        BP(Mat &leftImg, Mat &rightImg, const int disp, const float lmbd, const float sp, int iter) except +
        float calculateDataCost(Mat&leftPaddingImg, Mat&rightPaddingImg, const int h, const int w, const int d);
        void beliefPropagate(bool visualize);
        Mat maxProduct(vector[vector[Mat]] &msgCopy, int h, int w, int dir);
        Mat getDispMap();
        Mat do_match();

cdef extern from "../../methods/mbp.hpp" namespace "mbp":
    cdef cppclass MBP:
        #private:
        # not safe
        int height;
        int width;
        int disp;
        int iter;
        Mat leftPaddingImg;
        Mat rightPaddingImg;
        float costLambda;
        Mat smoothCostMat;
        vector[vector[Mat]] msg;
        vector[vector[Mat]] obs;

        #public:
        MBP(Mat &leftImg, Mat &rightImg, const int disp, const float lmbd, const float sp, int iter) except +
        float calculateDataCost(const int h, const int w, const int d);
        void calculateDataCostThread(int sh, int eh);
        void beliefPropagate(bool visualize);
        void beliefPropagateThread(vector[vector[Mat]] &msgCopy, int sh, int eh);
        Mat maxProduct(vector[vector[Mat]] &msgCopy, int h, int w, int dir);
        Mat getDispMap();
        Mat do_match();