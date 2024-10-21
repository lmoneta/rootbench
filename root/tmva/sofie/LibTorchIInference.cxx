// Automatically configured by CMake
// Author: Federico Sossai (fsossai), 2021

#include <benchmark/benchmark.h>
// #include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>


#include <torch/script.h> // One-stop header.

#include <check_mem.h>


using namespace std;

bool testOutput = true;

// benchmark::State& state, ) {
//   auto args_tuple = std::make_tuple(std::move(args)...);

template <class... Args>

static void BM_Libtorch_Inference(benchmark::State &state, std::string model_path,
                                  std::vector<std::vector<int64_t>> shapes) //, vector<vector<int64_t>> shapes)
{

  double mem0 = check_mem();
  double mem1 = 0;

   // fix negative shapes
   for (int i = 0; i < shapes.size(); i++) {
      for (int j = 0; j < shapes[i].size(); j++) {
         if (shapes[i][j] < 0)
            shapes[i][j] = -shapes[i][j];
         // cout << shapes[i][j] << " ";
      }
      // shapes[i].insert(shapes[i].begin(), nevts);
      // cout << endl;
   }


   //size_t nevts = 64;
   size_t bsize = shapes[0][0];
   size_t nevts = (bsize == 1) ? 100 : 4*bsize;
   size_t nrep = nevts / bsize;
   //std::cout << "using batch size " << bsize << std::endl;

   auto t00 = std::chrono::high_resolution_clock::now();
   torch::jit::script::Module module;
   module = torch::jit::load(model_path);
   auto t01 = std::chrono::high_resolution_clock::now();
   auto initDuration = std::chrono::duration_cast<std::chrono::microseconds>(t01 - t00).count();
   state.counters["init time(ms)"] = initDuration/1.E6;


   std::vector<std::vector<torch::jit::IValue>> allInputs(nrep);
   for ( auto & inputs : allInputs) {
      inputs.clear();
      for (int i = 0; i < shapes.size(); i++) {
         inputs.push_back(torch::rand(shapes[i]));
      }
   }
   mem0 = std::min(mem0, check_mem());

   //at::Tensor output = module.forward(inputs).toTensor();

   double totDuration = 0;
   int ntimes = 0;

   for (auto _ : state) {
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < nrep; i++) {
         auto & inputs = allInputs[i];
         //for (int i = 0; i < shapes.size(); i++) {
         //   inputs.push_back(torch::rand(shapes[i]));
         //}
         at::Tensor output = module.forward(inputs).toTensor();
      }

      auto t2 = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

      totDuration += duration / 1.E3; // in milliseconds
      ntimes++;
      mem1 = std::max(check_mem(),mem1);

   }


   //std::cout << "ntimes is " << ntimes << std::endl;
   state.counters["time/evt(ms)"] = totDuration / double(ntimes * nevts);
   state.counters["memory"] = mem1-mem0;
}

//vector<vector<int64_t>> shapes = {{10}};
//string model_path = "input_models/model.pt";

//, "input_models/model.pt")
// BENCHMARK_CAPTURE(BM_Libtorch_Inference, Conv2DTranspose_Relu_Sigmoid2, "input_models/model.pt", {{64, 10}})
//    ->Unit(benchmark::kMillisecond);

// conv models

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Conv2d_d100_L14_B1, "input_models/Conv2d_d100_L14_B1.pt", {{1, 1, 100, 100}})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Conv2d_d100_L14_B32, "input_models/Conv2d_d100_L14_B32.pt", {{32, 1, 100, 100}})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Conv2d_d100_L1_B1, "input_models/Conv2d_d100_L1_B1.pt", {{1, 1, 100, 100}})
   ->Unit(benchmark::kMillisecond);


BENCHMARK_CAPTURE(BM_Libtorch_Inference, Conv3d_d32_L4_B1, "input_models/Conv3d_d32_L4_B1.pt", {{1, 1, 32, 32, 32}})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, ConvTrans2d_Model_B1, "input_models/ConvTrans2d_Model_B1.pt", {{1, 1, 4, 4}})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Generator_B1, "input_models/Generator_B1.pt", {{1, 14}})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Generator_B64, "input_models/Generator_B64.pt", {{64, 14}})
   ->Unit(benchmark::kMillisecond);


BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_16, "input_models/Linear_16.pt", {{16, 100}})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_32, "input_models/Linear_32.pt", {{32, 100}})
   ->Unit(benchmark::kMillisecond);


BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_64, "input_models/Linear_64.pt", {{64, 100}})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_100_100_B1, "input_models/Linear_model_100_100_B1.pt", {{1, 100}})->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_100_1000_B1, "input_models/Linear_model_100_1000_B1.pt", {{1, 100}})->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_100_10000_B1, "input_models/Linear_model_100_10000_B1.pt", {{1, 100}})->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_100_100000_B1, "input_models/Linear_model_100_100000_B1.pt", {{1, 100}})->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_1000_100000_B1, "input_models/Linear_model_1000_100000_B1.pt", {{1, 1000}})->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_100_1000_B16, "input_models/Linear_model_100_1000_B16.pt", {{16, 100}})->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_100_1000_B64, "input_models/Linear_model_100_1000_B64.pt", {{64, 100}})->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_100_1000_B256, "input_models/Linear_model_100_1000_B256.pt", {{256, 100}})->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_100_1000_B1024, "input_models/Linear_model_100_1000_B1024.pt", {{1024, 100}})->Unit(benchmark::kMillisecond);




BENCHMARK_CAPTURE(BM_Libtorch_Inference, Linear_event, "input_models/Linear_event.pt", {{1, 100}})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_Libtorch_Inference, higgs_model_dense, "input_models/higgs_model_dense.pt", {{1, 7}})
   ->Unit(benchmark::kMillisecond);



BENCHMARK_MAIN();
