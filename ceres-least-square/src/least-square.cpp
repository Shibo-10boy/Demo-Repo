#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>
#include <ceres/cost_function.h>
#include <matplotlibcpp.h>
#include <cmath>
#include "least_square.hpp"
using namespace std;
using namespace ceres;
using namespace matplotlibcpp;

int main(int argc, char **argv)
{
    double ar = 3, br = 4, cr = 5; // 真实参数值
    double ae = 0, be = 0, ce = 0; // 估计参数值/
    int N = 100;                   // 数据点个数
    double abc[3] = {ae, be, ce};

    default_random_engine e(time(0)); // 引擎生成随机序列，设置种子

    uniform_real_distribution<double> u(-100, 100);

    vector<double> x_data; // 加入噪声之后的 x,y
    vector<double> y_data;
    vector<double> xr_data; // 拟合之后的x,y
    vector<double> yr_data;

    for (int i = 0; i < N; ++i)
    {
        double y = ar * i * i + br * i + cr; // y =ax^2 + bx + c

        x_data.emplace_back(i);
        y_data.emplace_back(y + u(e));
    }

    ceres::Problem problem;

    for (int i = 0; i < N; i++)
    {

        // TODo：自动求导（方式一)
        // 添加误差项。使用自动求导，模板参数：误差类型、输出维度、输入维度、维数要与前面struct中一致
        // problem.AddResidualBlock(new ceres::AutoDiffCostFunction<cost, 1, 3>(
        //                              new cost(x_data[i], y_data[i])),
        //                          nullptr, abc);
        //  nullptr为核函数不使用为空，abc为待估计参数
        // TODO：自动求导（方式二)
        // CostFunction *costfunction = new AutoDiffCostFunction<cost, 1, 3>(
        //     new cost(x_data[i], y_data[i]));
        // problem.AddResidualBlock(costfunction, nullptr, abc);
        // TODO：数值求导(最慢)
        // CostFunction *costfunction =
        //     new NumericDiffCostFunction<cost, ceres::CENTRAL, 1, 3>(
        //         new cost(x_data[i], y_data[i]));
        // problem.AddResidualBlock(costfunction, nullptr, abc);
        // TODO:解析求导（最快）
        CostFunction *costfuncion = new QuadraticCostFunction(x_data[i], y_data[i]);
        problem.AddResidualBlock(costfuncion, nullptr, abc);
    }

    ceres::Solver::Options options; // 定义配置项

    options.linear_solver_type =
        ceres::DENSE_NORMAL_CHOLESKY; // 配置增量方程的解法

    options.minimizer_progress_to_stdout = true; // 输出到cout

    ceres::Solver::Summary summary; // 定义优化信息

    chrono::steady_clock::time_point t1 =
        chrono::steady_clock::now(); // 计时：求解开始时间

    ceres::Solve(options, &problem, &summary); // 开始优化求解！

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    cout << "solve time cost = " << time_used.count() << "s."
         << endl; // 输出求解耗时

    cout << summary.BriefReport() << endl; // 输出简要优化信息

    cout << "estimated a, b, c = ";

    for (auto a : abc) // 输出优化变量
        cout << a << " ";
    cout << endl;

    // 显示拟合之后的曲线
    for (int i = 0; i < N; ++i)
    {
        double y = abc[0] * i * i + abc[1] * i + abc[2];
        xr_data.emplace_back(i);
        yr_data.emplace_back(y);
    }

    plot(x_data, y_data, "r.");
    plot(xr_data, yr_data, "b+");
    title("least-square");
    legend();
    show();

    return 0;
}