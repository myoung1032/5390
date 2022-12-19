Dataset file: CIFAR-10 validation set: "test_batch", a binary file
Main files for different benchmark tests:
  1. CPU benckmark test for different BLAS: run "main_nets.py"
  2. CPU benckmark test for different depth of VGG: run "main_nets_vgg_depth.py"
  3. To get the plots of the benckmark results: run "getBlasPlot.py", "getBlasPlotVggDepth.py" and "getCuDNNplot.py"
Run the code: 
  step1: change the file path of the .py file intended to be run
  step2: cd to the file folder
  step3: activate conda environment with pytorch and the specific BLAS installed by "conda activate <your env name>"
  step4: run the file by command "python3 main_nets.py"
