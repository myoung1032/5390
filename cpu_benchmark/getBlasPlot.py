import numpy as np
import matplotlib.pyplot as plt

PROJECT_FOLDER_PATH = "/Users/dennice/Desktop/ese539_project/"


if __name__ == '__main__':  
    # gather results from terminal
    # intelMKL
    mkl_accumulated_runtimes_vgg16 =  [0, 5.053309281667073, 13.298949321111042, 19.057863235473633, 24.33093198140462, 29.876496871312458, 35.15975133577982, 39.99678325653076, 45.230734984079994, 50.40185236930847, 55.50572737058004]
    mkl_accumulated_runtimes_resnet18 =  [0, 7.378676732381185, 14.652623494466145, 22.034536600112915, 29.836192051569622, 37.227570454279586, 44.99072106679281, 54.127244631449386, 62.097943067550666, 70.02563937505087, 77.50988936424257]
    mkl_accumulated_runtimes_alexnet =  [0, 4.6099222501118975, 9.375885645548502, 14.136274576187134, 18.849567969640095, 23.466859579086304, 27.921010573705036, 32.824009497960404, 37.649938821792595, 42.2095870176951, 46.83620309829711]

    # Eigen
    eigen_accumulated_runtimes_vgg16 =  [0, 8.845110416412354, 17.814742724100746, 26.971078316370644, 36.04378962516785, 45.20081067085266, 54.23926258087158, 63.34352699915568, 73.84662556648254, 82.99063126246135, 92.15194201469421]
    eigen_accumulated_runtimes_resnet18 =  [0, 13.341625054677328, 26.97143316268921, 40.746382077534996, 54.32487988471985, 68.17685023943584, 81.72911826769511, 95.27664796511333, 108.76746662457785, 122.24155632654826, 136.009751478831]
    eigen_accumulated_runtimes_alexnet =  [0, 7.072631359100342, 14.145347356796265, 21.531800508499146, 28.650781393051147, 35.601744651794434, 42.67157133420309, 49.84142629305522, 56.844382286071784, 63.806945959726974, 70.95917479197185]

    # OpenBlAS
    openblas_accumulated_runtimes_vgg16 =  [0, 6.663691600163777, 13.340088208516438, 20.39374264081319, 26.96438654263814, 33.653284311294556, 40.05916142463684, 46.83469533920288, 53.307268142700195, 59.831080516179405, 66.64956482251486]
    openblas_accumulated_runtimes_resnet18 =  [0, 8.449357271194458, 15.649805148442585, 22.56061275800069, 29.52746240297953, 36.89913328488667, 44.119978666305535, 51.19614100456237, 58.43299094835916, 65.65496269861856, 72.98847897847493]
    openblas_accumulated_runtimes_alexnet =  [0, 5.948598146438599, 11.781911452611286, 17.46285653114319, 23.5251358350118, 29.2996084690094, 36.322557051976524, 42.384570837020874, 48.08326745033264, 53.92362117767334, 59.76902310053507]

    n = 10000          # total number of samples to inference
    plot_unit = 1000    # plot_unit is the step by which our plots are used



    nets = ["different_blas_on_vgg16", "different_blas_on_resnet18", "different_blas_on_alexnet"]

    x = np.arange(0, n+plot_unit, plot_unit)/1000

    # plot different_blas_on_vgg16
    plt.plot(x, mkl_accumulated_runtimes_vgg16, marker='o',label='intelMKL')
    plt.plot(x, eigen_accumulated_runtimes_vgg16, marker='^',label='Eigen')
    plt.plot(x, openblas_accumulated_runtimes_vgg16, marker='D',label='OpenBLAS')

    # Add labels and title
    plt.title("Comparison of vgg16 inference runtime between BLAS on Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30Hz (IntelMKL vs Eigen vs OpenBLAS")
    plt.xlabel("number of inference samples (k)")
    plt.ylabel("accumulated runtime (second)")

    plt.legend()
    plt.savefig(PROJECT_FOLDER_PATH + "graphs/"+nets[0]+".png", bbox_inches ="tight", pad_inches = 1)
    plt.clf()


    # plot different_blas_on_resnet18
    plt.plot(x, mkl_accumulated_runtimes_resnet18, marker='o',label='intelMKL')
    plt.plot(x, eigen_accumulated_runtimes_resnet18, marker='^',label='Eigen')
    plt.plot(x, openblas_accumulated_runtimes_resnet18, marker='D',label='OpenBLAS')

    # Add labels and title
    plt.title("Comparison of resnet18 inference runtime between BLAS on Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30Hz (IntelMKL vs Eigen vs OpenBLAS")
    plt.xlabel("number of inference samples (k)")
    plt.ylabel("accumulated runtime (second)")

    plt.legend()
    plt.savefig(PROJECT_FOLDER_PATH + "graphs/"+nets[1]+".png", bbox_inches ="tight", pad_inches = 1)
    plt.clf()


    # plot different_blas_on_alexnet
    plt.plot(x, mkl_accumulated_runtimes_alexnet, marker='o',label='intelMKL')
    plt.plot(x, eigen_accumulated_runtimes_alexnet, marker='^',label='Eigen')
    plt.plot(x, openblas_accumulated_runtimes_alexnet, marker='D',label='OpenBLAS')

    # Add labels and title
    plt.title("Comparison of alexnet inference runtime between BLAS on Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30Hz (IntelMKL vs Eigen vs OpenBLAS")
    plt.xlabel("number of inference samples (k)")
    plt.ylabel("accumulated runtime (second)")

    plt.legend()
    plt.savefig(PROJECT_FOLDER_PATH + "graphs/"+nets[2]+".png", bbox_inches ="tight", pad_inches = 1)
    plt.clf()
    # plt.show()
    # print("Task: inference {n} epochs\n average run time for {n_trial} times = {time}s".format(n = 50, n_trial=trials, time=runtime))
    print("end")