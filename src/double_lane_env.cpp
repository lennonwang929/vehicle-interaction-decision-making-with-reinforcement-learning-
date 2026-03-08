#include <matplotlib-cpp/matplotlibcpp.h>

// #include "env.hpp"
#include "double_lane_env.hpp"

namespace plt = matplotlibcpp;

EnvCrossroads::EnvCrossroads(double size, double width)
    : map_size(size), lanewidth(width) {
  rect = {{{-size, -size, -lanewidth, -lanewidth, -size},
           {-size, size, size, -size, -size}},
          {{size, size, lanewidth, lanewidth, size},
           {-size, size, size, -size, -size}}};

  laneline = {{{0, 0}, {-size, size}}};

  for (const std::vector<std::vector<double>> &r : rect) {
    Eigen::MatrixXd mat(r.size(), r[0].size());
    for (int i = 0; i < r.size(); ++i) {
      for (int j = 0; j < r[0].size(); ++j) {
        mat(i, j) = r[i][j];
      }
    }
    rect_mat.emplace_back(mat);
  }

  for (const std::vector<std::vector<double>> &l : laneline) {
    Eigen::MatrixXd mat(l.size(), l[0].size());
    for (int i = 0; i < l.size(); ++i) {
      for (int j = 0; j < l[0].size(); ++j) {
        mat(i, j) = l[i][j];
      }
    }
    laneline_mat.emplace_back(mat);
  }
}

void EnvCrossroads::draw_env(void) {
  plt::fill(rect[0][0], rect[0][1], {{"color", "#BFBFBF"}});
  plt::fill(rect[1][0], rect[1][1], {{"color", "#BFBFBF"}});
  plt::plot(rect[0][0], rect[0][1], {{"color", "k"}, {"linewidth", "2"}});
  plt::plot(rect[1][0], rect[1][1], {{"color", "k"}, {"linewidth", "2"}});

  plt::plot(laneline[0][0], laneline[0][1],
            {{"color", "orange"}, {"linewidth", "2"}, {"linestyle", "--"}});
}
