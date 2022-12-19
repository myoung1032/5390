import numpy as np
import matplotlib.pyplot as plt

PROJECT_FOLDER_PATH = "/Users/dennice/Desktop/ese539_project/"


if __name__ == '__main__':  
    # gather results from terminal
    # intelMKL
    mkl_accumulated_runtimes_vgg16 =  [0, 5.199206113815308, 9.990572532018025, 14.724559307098389, 20.170207182566326, 25.34984381993612, 30.026220798492435, 35.09671115875244, 39.85350354512533, 44.72512944539388, 49.63954448699951]
    mkl_accumulated_runtimes_vgg13 =  [0, 3.993264357248942, 7.893426179885864, 11.738737185796102, 15.812665780385336, 19.82254163424174, 23.74586296081543, 27.722416162490845, 31.734636704126995, 35.8162005742391, 39.79699770609538]
    mkl_accumulated_runtimes_vgg19 =   [0, 5.644344409306844, 11.27334705988566, 17.208319107691445, 22.787654161453244, 28.486349105834957, 34.23181080818176, 40.03750236829122, 45.576524337132774, 51.265472650527954, 56.89262914657593]

    # Eigen
    eigen_accumulated_runtimes_vgg16 =   [0, 8.821001688639322, 17.644165674845375, 26.46629444758097, 35.12368869781494, 43.83033108711243, 52.47998030980428, 61.19740263621013, 69.91312917073569, 78.64336840311687, 87.38298996289572]
    eigen_accumulated_runtimes_vgg13 =  [0, 6.813329140345256, 13.62054713567098, 20.44076148668925, 27.247116804122925, 34.09068012237549, 40.90605243047079, 47.78326098124187, 54.62879045804342, 61.43207049369812, 68.24987451235454]
    eigen_accumulated_runtimes_vgg19 =  [0, 10.57789691289266, 21.13245495160421, 31.827746629714966, 42.43997518221537, 53.035576820373535, 63.61068320274353, 74.21344415346782, 84.81261189778647, 95.40271838506064, 106.01195780436198]

    # OpenBlAS
    openblas_accumulated_runtimes_vgg16 =  [0, 6.892654021581014, 13.025930643081665, 19.373157103856403, 25.665801286697388, 31.946534474690754, 38.42763511339823, 44.90132411321004, 51.32465553283691, 57.67728996276855, 64.11292171478271]
    openblas_accumulated_runtimes_vgg13 =   [0, 5.524138847986857, 10.732083082199097, 16.011467695236206, 21.291414340337116, 26.53609299659729, 31.857831875483196, 37.35439968109131, 42.726849714914955, 48.060693105061844, 53.35010838508605]
    openblas_accumulated_runtimes_vgg19 =  [0, 7.768385012944539, 15.333263397216797, 23.248083353042603, 30.83296052614848, 38.693677822748825, 46.47208650906881, 54.031857649485275, 62.11539403597514, 70.28375736872356, 78.06106201807658]

    n = 10000          # total number of samples to inference
    plot_unit = 1000    # plot_unit is the step by which our plots are used



    nets = ["different_blas_on_vgg16_depth", "different_blas_on_vgg13_depth", "different_blas_on_vgg19_depth"]

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


    # plot different_blas_on_vgg13
    plt.plot(x, mkl_accumulated_runtimes_vgg13, marker='o',label='intelMKL')
    plt.plot(x, eigen_accumulated_runtimes_vgg13, marker='^',label='Eigen')
    plt.plot(x, openblas_accumulated_runtimes_vgg13, marker='D',label='OpenBLAS')

    # Add labels and title
    plt.title("Comparison of vgg13 inference runtime between BLAS on Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30Hz (IntelMKL vs Eigen vs OpenBLAS")
    plt.xlabel("number of inference samples (k)")
    plt.ylabel("accumulated runtime (second)")

    plt.legend()
    plt.savefig(PROJECT_FOLDER_PATH + "graphs/"+nets[1]+".png", bbox_inches ="tight", pad_inches = 1)
    plt.clf()


    # plot different_blas_on_vgg19
    plt.plot(x, mkl_accumulated_runtimes_vgg19, marker='o',label='intelMKL')
    plt.plot(x, eigen_accumulated_runtimes_vgg19, marker='^',label='Eigen')
    plt.plot(x, openblas_accumulated_runtimes_vgg19, marker='D',label='OpenBLAS')

    # Add labels and title
    plt.title("Comparison of vgg19 inference runtime between BLAS on Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30Hz (IntelMKL vs Eigen vs OpenBLAS")
    plt.xlabel("number of inference samples (k)")
    plt.ylabel("accumulated runtime (second)")

    plt.legend()
    plt.savefig(PROJECT_FOLDER_PATH + "graphs/"+nets[2]+".png", bbox_inches ="tight", pad_inches = 1)
    plt.clf()
    # plt.show()
    # print("Task: inference {n} epochs\n average run time for {n_trial} times = {time}s".format(n = 50, n_trial=trials, time=runtime))
    print("end")