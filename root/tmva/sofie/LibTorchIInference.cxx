

#include <benchmark/benchmark.h>
// #include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>

//#include <torch/script.h>

using namespace std;

bool testOutput = true;

// benchmark::State& state, ) {
//   auto args_tuple = std::make_tuple(std::move(args)...);

//template <class... Args>

vector<vector<int64_t>> shapes = {{1, 10}};
string model_path = "input_models/model.pt";

//static void BM_Libtorch_Inference(benchmark::State &state, std::string model_path, std::vector<std::vector<int64_t>> shapes) //, vector<vector<int64_t>> shapes)
static void BM_Libtorch_Inference(benchmark::State &state)
{
   //cout << "hello" << endl;
   //auto args_tuple = std::make_tuple(std::move(args)...);

   //string model_path = get<0>(args_tuple);
   std::cout << "path : " << model_path << std::endl;
   //vector<vector<int64_t>> shapes = get<1>(args_tuple);

   // fix negative shapes
   for (int i = 0; i < shapes.size(); i++) {
      for (int j = 0; j < shapes[i].size(); j++) {
         // if (shapes[i][j] < 0)
         // shapes[i][j] = -shapes[i][j];
         cout << shapes[i][j] << " ";
      }
      cout << endl;
   }

   // print shapes
#if 0
   torch::jit::script::Module module;
   module = torch::jit::load(model_path);

   std::vector<torch::jit::IValue> inputs;

   for (int i = 0; i < shapes.size(); i++) {
       inputs.push_back(torch::rand(shapes[i]));
    }

   at::Tensor output = module.forward(inputs).toTensor();
#endif

#if 0
   double totDuration = 0;
   int ntimes = 0;
   int nevts = 1000;
   int bsize = 1;
   for (auto _ : state) {
       auto t1 = std::chrono::high_resolution_clock::now();
       for (int i = 0; i < nevts; i += bsize) {
          float *p1 = input.data() + inputSize1 * i;
          //float *p2 = input2.data() + inputSize2 * i;
          //float *p3 = input3.data() + inputSize3 * i;
         auto y = s.infer(p1);
        //  auto y = s.infer(p1, p2, p3);
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      totDuration += duration / 1.E3; // in milliseconds
      ntimes++;
   }
#endif
   //state.counters["time/evt(ms)"] = 1; // totDuration / double(ntimes * nevts);
}



//, "input_models/model.pt")
//BENCHMARK_CAPTURE(BM_Libtorch_Inference, Conv2DTranspose_Relu_Sigmoid2, model_path, shapes);
BENCHMARK(BM_Libtorch_Inference);
//->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Conv3d_d32_L4_B1, "input_models/Conv3d_d32_L4_B1.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, ConvTModel_G4, "input_models/ConvTModel_G4.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, ConvTrans2dModel_B1, "input_models/ConvTrans2dModel_B1.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Conv_d100_L14_B1, "input_models/Conv_d100_L14_B1.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Conv_d100_L14_B32, "input_models/Conv_d100_L14_B32.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Conv_d100_L1_B1, "input_models/Conv_d100_L1_B1.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, DDB_B1, "input_models/DDB_B1.onnx")->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, GRU_d10_L20_h8_B1, "input_models/GRU_d10_L20_h8_B1.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Generator_B1, "input_models/Generator_B1.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Generator_B64, "input_models/Generator_B64.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, LSTM_d10_L20_h8_B1, "input_models/LSTM_d10_L20_h8_B1.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Linear_16, "input_models/Linear_16.onnx")->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Linear_32, "input_models/Linear_32.onnx")->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Linear_64, "input_models/Linear_64.onnx")->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, Linear_event, "input_models/Linear_event.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, RNN_d10_L20_h8_B1, "input_models/RNN_d10_L20_h8_B1.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, SimpleNN_Alice, "input_models/SimpleNN_Alice.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, higgs_model_dense, "input_models/higgs_model_dense.onnx")
//    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(BM_ONNXRuntime_Inference, resnet18v1,
// "input_models/resnet18v1.onnx")->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

// // define main to pass some convenient command line parameters
// int main(int argc, char **argv)
// {

//    // Parse command line arguments
//    for (int i = 1; i < argc; i++) {
//       std::string arg = argv[i];
//       if (arg == "-v") {
//          // std::cout << "---running in verbose mode" << std::endl;
//          // verbose = true;
//       } else if ((arg == "-d" || arg == "--dir") && argc > i + 1) {
//          std::string pathDir = argv[i + 1];
//          std::filesystem::path path(pathDir);
//          std::filesystem::current_path(path);
//          i++;
//       }
//    }

//    std::cout << "running benchmark from current directory " << std::filesystem::current_path() << std::endl;

//    ::benchmark::Initialize(&argc, argv);
//    ::benchmark::RunSpecifiedBenchmarks();

//    return 0;
// }
