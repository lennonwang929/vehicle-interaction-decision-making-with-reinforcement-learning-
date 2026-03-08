#pragma once
#ifndef __DOUBLE_LANE_ENV_HPP
#define __DOUBLE_LANE_ENV_HPP

#include <Eigen/Core>
#include <vector>

class EnvCrossroads {
private:
public:
  double map_size;
  double lanewidth;

  std::vector<std::vector<std::vector<double>>> rect;
  std::vector<std::vector<std::vector<double>>> laneline;

  std::vector<Eigen::MatrixXd> rect_mat;
  std::vector<Eigen::MatrixXd> laneline_mat;

  EnvCrossroads(double size = 25.0, double width = 4.0);
  ~EnvCrossroads() {}
  void draw_env(void);
};

#endif
