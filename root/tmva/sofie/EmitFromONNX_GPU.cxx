// Author: Federico Sossai
// Last modified: 2021/07/30
// Description:
//    SOFIE command line compiler.
//    This program is automatically run when the corresponding test target is built.
//    Usage example: $./EmitFromONNX indir/mymodel.onnx outdir/myname.hxx

#include <iostream>

#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

int main(int argc, char *argv[]){
   if (argc < 2) {
      std::cerr << "ERROR: missing input file\n";
      return -1;
   }

   std::string outname = argv[1];
   // assuming the file name is input_models/*.onnx
   outname.erase(outname.end()-5, outname.end());
   std::cout << outname << std::endl;
   
   outname.erase(outname.begin(), outname.begin()+13);
   std::cout << outname << std::endl;
   RModelParser_ONNX parser;
   std::cout << "Parsing file " << argv[1] << std::endl;
   RModel model = parser.Parse(argv[1]);
   model.GenerateGPU();
   model.OutputGeneratedGPU(outname + "GPU_FromONNX.hxx");

   return 0;
}
