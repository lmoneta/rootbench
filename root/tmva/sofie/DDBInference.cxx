
#include <iostream>
#include <thread>
#include <chrono>
#include <utility>
#include <vector>
#include <memory>
#include <functional>
#include <random>

#include "DDB_B1GPU_FromONNX.hxx"   // CMS onnx model
#include "TMath.h"


using namespace std;
bool verbose = false;
bool testOutput = true;

template <class S>
void BM_SOFIE_Inference_3(size_t bsize, size_t inputSize1, size_t inputSize2, size_t inputSize3)
{
   size_t nevts = 64;
   size_t nrep = nevts / bsize;

   size_t eventSize = inputSize1 + inputSize2+inputSize3;

   vector<float> input1(inputSize1*nevts);
   vector<float> input2(inputSize2*nevts);
   vector<float> input3(inputSize3*nevts);

   if (!testOutput) {
   static std::uniform_real_distribution<float> distribution(-1, 1);
   static std::default_random_engine generator;
   std::generate(input1.begin(), input1.end(), []() { return distribution(generator); });
   std::generate(input2.begin(), input2.end(), []() { return distribution(generator); });
   std::generate(input3.begin(), input3.end(), []() { return distribution(generator); });
   }
   else {
      // generate fixed data
      input1 = vector<float>(input1.size(),1.);
      input2 = vector<float>(input2.size(),2.);
      input3 = vector<float>(input3.size(),3.);
   }

   S s("");

   //std::cout << "init done - do benchmark \n";

   double totDuration = 0;
   int ntimes = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nevts; i += bsize) {
        float * p1 = input1.data()+ inputSize1*i;
        float * p2 = input2.data()+ inputSize2*i;
        float * p3 = input3.data()+ inputSize3*i;
        auto y = s.infer(std::vector<float>(p1, p1 + inputSize1), std::vector<float>(p2, p2 + inputSize2), std::vector<float>(p3, p3 + inputSize3));
      
      auto t2 = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      totDuration += duration / 1.E3;  // in milliseconds
      ntimes++;
   }

}

int main() {

    BM_SOFIE_Inference_3<TMVA_SOFIE_DDB_B1::Session>(1, 1*27, 60*80, 5*2);

    return 0;

}