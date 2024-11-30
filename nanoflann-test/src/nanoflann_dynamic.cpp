#include <iostream>
#include <vector>
#include <ctime>
#include <limits>
#include <time.h>
#include "/home/b212/10boy/nanoflann-test/include/nanoflann.hpp"
#include "/home/b212/10boy/nanoflann-test/include/utils.h"
using namespace std;
using namespace nanoflann;

using my_kd_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Tree<double>>,
    Tree<double>, 2 /* dim */
    >;

clock_t start_quary, end_quary;

void direct_quary(Tree<double> &tree_)
{
    for (int i = 0; i < 100000; i = i + 1)
    {
        Tree<double>::tree_node node;
        node.x = 10 * (rand() % 1000) / double(1000);
        node.y = 10 * (rand() % 1000) / double(1000);
        node.z = 0;
        tree_.pts.emplace_back(node);
        double min_distance = std::numeric_limits<double>::max();
        for (const auto &node : tree_.pts)
        {
            double distance = std::sqrt(std::pow(node.x - 10, 2) + std::pow(node.y - 20, 2));
            if (distance < min_distance)
            {
                min_distance = distance;
            }
        }
    }
}

void kdtree_quary(Tree<double> &tree_)
{
    my_kd_tree_t index(2 /*dim*/, tree_, {100 /* max leaf */});

    const size_t num_results = 1;

    size_t ret_index;
    double query[2] = {10, 10};
    double out_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr);
    for (int i = 0; i < 100000; i = i + 1)
    {
        Tree<double>::tree_node node;
        node.x = 10 * (rand() % 1000) / double(1000);
        node.y = 10 * (rand() % 1000) / double(1000);
        node.z = 0;

        tree_.pts.emplace_back(node);
        index.addPoints(i, i + 1);
        index.findNeighbors(
            resultSet, query, {10});
        std::cout << "knnSearch(nn=" << num_results << "): \n";
        std::cout << "ret_index=" << ret_index
                  << " out_dist_sqr=" << out_dist_sqr << std::endl;
        std::cout << "point: ("
                  << "point: (" << tree_.pts[ret_index].x << ", "
                  << tree_.pts[ret_index].y << ", "
                  << ")" << std::endl;
        std::cout << "tree_:" << tree_.pts.size() << std::endl;
    }

    // std::cout << "knnSearch(nn=" << num_results << "): \n";
    // std::cout << "ret_index=" << ret_index
    //           << " out_dist_sqr=" << out_dist_sqr << std::endl;

    // std::cout << std::endl;
}
int main(int argc, char const *argv[])
{
    srand(static_cast<unsigned int>(time(nullptr)));

    Tree<double> tree_;

    start_quary = clock();
    kdtree_quary(tree_);
    // direct_quary(tree_);
    end_quary = clock();
    std::cout << "quary_time = " << double(end_quary - start_quary) / CLOCKS_PER_SEC << "s" << endl; // 输出时间（单位：ｓ）
    return 0;
}
