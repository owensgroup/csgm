#include <boost/program_options.hpp>

namespace po = boost::program_options;

void parseArgsSGM(int argc, char**argv, po::variables_map& vm) {
  po::options_description desc("SGM options");
  desc.add_options()
    ("help",
        "produce help message")
    ("ta", po::value<int>()->default_value(32),
        "threads per A row")
    ("tb", po::value<int>()->default_value(32),
        "B slab width")
    ("mode", po::value<std::string>()->default_value("fixedrow"),
        "row or column")
    ("split", po::value<bool>()->default_value(false),
        "True means split spgemm computation")

    // General params
    ("source", po::value<int>()->default_value(0),
        "Source node traversal is launched from")
    ("niter", po::value<int>()->default_value(10),
        "Number of iterations to run after warmup")
    ("directed", po::value<int>()->default_value(0),
        "0: follow mtx, 1: force undirected graph to be directed, 2: force directed graph to be undirected")
    ("mxvmode", po::value<int>()->default_value(1),
        "0: push-pull, 1: push only, 2: pull only")
    ("timing", po::value<int>()->default_value(1),
        "0: final timing, 1: per niter timing, 2: per graphblas algorithm timing")
    ("memusage", po::value<float>()->default_value(1.0),
        "Multiple of edge used to store temporary neighbor list during push phase")
    ("switchpoint", po::value<float>()->default_value(0.01),
        "Percentage of nnz needed in order to switch from sparse to dense when mxvmode=push-pull")
    ("transpose", po::value<bool>()->default_value(false),
        "True means use transpose graph")
    ("mtxinfo", po::value<bool>()->default_value(true),
        "True means show matrix MTX info")
    ("dirinfo", po::value<bool>()->default_value(false),
        "True means show mxvmode direction info, and when switches happen")
    ("verbose", po::value<bool>()->default_value(true),
        "0: timing output only, 1: correctness indicator")
    ("struconly", po::value<bool>()->default_value(false),
        "True means use implied nonzeroes, False means key-value operations")
    ("earlyexit", po::value<bool>()->default_value(true),
        "True means use early exit, False means do not use it")
    ("earlyexitbench", po::value<bool>()->default_value(false),
        "True means do early exit benchmarking (will automatically turn earlyexit on as well), False means do not use it")
    ("opreuse", po::value<bool>()->default_value(false),
        "True means use operand reuse, False means do not use it")
    ("endbit", po::value<bool>()->default_value(true),
        "True means do not do radix sort on full 32 bits, False means do it on full 32 bits")
    ("reduce", po::value<bool>()->default_value(true),
        "True means do the reduce, False means do not do it")
    ("prealloc", po::value<bool>()->default_value(true),
        "True means do the prealloc, False means do not do it")
    ("sort", po::value<bool>()->default_value(true),
        "True means sort, False means do not sort. (Option is only valid if struconly is true)")
    ("mask", po::value<bool>()->default_value(true),
        "True means use fused mask in pull direction, False means do not do it")

    // GPU params
    ("nthread", po::value<int>()->default_value(128),
        "Number of threads per block")
    ("ndevice", po::value<int>()->default_value(0),
        "GPU device number to use")
    ("debug", po::value<bool>()->default_value(false),
        "True means show debug messages")
    ("memory", po::value<bool>()->default_value(false),
        "True means show memory info")

    // SGM params
    ("num-seeds", po::value<int>(),                                                   "number of seeds"      )
    ("num-iters", po::value<int>()->default_value(20),                                "number of iterations" )
    ("tolerance", po::value<float>()->default_value(1.0),                             "convergence tolerance")
    ("A",         po::value<std::string>()->default_value((std::string)"data/A.mtx"), "path to A.mtx"        )
    ("B",         po::value<std::string>()->default_value((std::string)"data/B.mtx"), "path to B.mtx"        )
    ("sgm-debug", po::value<bool>()->default_value(false),                            "print SGM debug info" );

  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(vm.count("help")) {
    std::cout << desc << "\n";
    exit(1);
  }
}
