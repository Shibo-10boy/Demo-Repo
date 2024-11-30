#include <iostream>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>

using namespace Eigen;
using namespace std;

int main()
{
    // init
    SparseMatrix<double> Hessian(2, 2);
    Vector2d gradient(2);

    SparseMatrix<double> A(3, 2);
    Vector3d a(3);
    Vector3d b(3);

    Hessian.insert(0, 0) = 1.0;
    Hessian.insert(0, 1) = -1.0;
    Hessian.insert(1, 0) = -1.0;
    Hessian.insert(1, 1) = 2.0;

    gradient << -2.0, -6.0;

    A.insert(0, 0) = 1.0;
    A.insert(0, 1) = 1.0;
    A.insert(1, 0) = -1.0;
    A.insert(1, 1) = 2.0;
    A.insert(2, 0) = 2.0;
    A.insert(2, 1) = 1.0;

    a << -OsqpEigen::INFTY,
        -OsqpEigen::INFTY,
        -OsqpEigen::INFTY;

    b << 2.0, 2.0, 3.0;

    OsqpEigen::Solver solver;

    solver.data()->setNumberOfConstraints(3);
    solver.data()->setNumberOfVariables(2);
    solver.data()->setHessianMatrix(Hessian);
    solver.data()->setGradient(gradient);
    solver.data()->setLinearConstraintsMatrix(A);
    solver.data()->setLowerBound(a);
    solver.data()->setUpperBound(b);

    // setting
    solver.settings()->setVerbosity(true);
    solver.initSolver();

    solver.solveProblem();

    auto QPSoQPSolutionlution = solver.getSolution();
    std::cout << "QPSoQPSolutionlution" << std::endl
              << QPSoQPSolutionlution << std::endl;
    auto x1 = QPSoQPSolutionlution[0];
    auto x2 = QPSoQPSolutionlution[1];
    auto f = 0.5 * pow(x1, 2) + pow(x2, 2) - x1 * x2 - 2 * x1 - 6 * x2;
    std::cout << "f(x):" << f << std::endl;
    return 0;
}
