#include <iostream>
#include <vector>
#include <ctime>
#include <limits>
#include <time.h>
#include "/home/b212/10boy/nanoflann-test/include/nanoflann.hpp"
#include "/home/b212/10boy/nanoflann-test/include/utils.h"
using namespace std;
using namespace nanoflann;
clock_t start_build, end_build;
clock_t start_quary, end_quary;

int main(int argc, char const *argv[])
{
    srand(static_cast<unsigned int>(time(nullptr)));

    Tree<double> tree_;

    generateRandomTree(tree_, 100000);

    start_build = clock();
    double min_distance = std::numeric_limits<double>::max();

    for (const auto &node : tree_.pts)
    {
        double distance = std::sqrt(std::pow(node.x, 2) + std::pow(node.y, 2));
        if (distance < min_distance)
        {
            min_distance = distance;
        }
    }
    std::cout << min_distance << std::endl;

    end_build = clock();
    cout << "build_time = " << double(end_build - start_build) / CLOCKS_PER_SEC << "s" << endl; // 输出时间（单位：ｓ）
    // for (auto i = tree_.pts.begin(); i != tree_.pts.end(); ++i)
    // {
    //     std::cout << i->x << " "
    //               << i->y << " "
    //               << i->z << " "
    //               << std::endl;
    // }
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, Tree<double>>,
        Tree<double>, 2 /* dim */
        >;

    // 定义一个查询点的结构体

    start_quary = clock();
    my_kd_tree_t index(2 /*dim*/, tree_, {1000 /* max leaf */});

    struct QueryPoint
    {
        double x, y;
    };

    double query[3] = {0, 0};
    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr);
    index.findNeighbors(
        resultSet, &query[0], nanoflann::SearchParameters(10));

    end_quary = clock();
    cout << "quary_time = " << double(end_quary - start_quary) / CLOCKS_PER_SEC << "s" << endl; // 输出时间（单位：ｓ）
    std::cout << "knnSearch(nn=" << num_results << "): \n";
    std::cout << "ret_index=" << ret_index
              << " out_dist_sqr=" << out_dist_sqr << std::endl;

    return 0;
}
