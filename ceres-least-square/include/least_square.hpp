#pragma once

#include <ceres/ceres.h>

class cost
{
public:
  double x_;
  double y_;
  cost(const double &x, const double &y) : x_(x), y_(y)
  {
  }
  template <typename T>
  bool operator()(T const *parameters, T *residuals) const
  {
    residuals[0] = T(y_) - (parameters[0] * T(x_) * T(x_) + parameters[1] * T(x_) + parameters[2]);
    return true;
  }
};

class QuadraticCostFunction : public ceres::SizedCostFunction<1, 3>
{
public:
  double x_;
  double y_;
  QuadraticCostFunction(const double &x, const double &y) : x_(x), y_(y)
  {
  }
  bool Evaluate(double const *const *parameters,
                double *residuals,
                double **jacobians) const
  {
    double a = parameters[0][0];
    double b = parameters[0][1];
    double c = parameters[0][2];

    double y = a * x_ * x_ + b * x_ + c;

    residuals[0] = y - y_;

    if (jacobians != nullptr && jacobians[0] != nullptr)
    {
      jacobians[0][0] = x_ * x_;
      jacobians[0][1] = x_;
      jacobians[0][2] = 1;
    }

    return true;
  }
};
