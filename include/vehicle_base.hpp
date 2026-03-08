#pragma once
#ifndef __VEHICLE_BASE_HPP
#define __VEHICLE_BASE_HPP

#include <cmath>
#include <memory>
#include <string>

#include <Eigen/Core>

#include "double_lane_env.hpp"
// #include "env.hpp"

#include "tracked_object.hpp"
#include "utils.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

class VehicleBase {
private:
  /* data */
public:
  static double length;
  static double width;
  static double tail_length;
  static double tail_width;
  static double tractor_length;
  static double tractor_width;
  static double safe_length;
  static double safe_width;
  static std::shared_ptr<EnvCrossroads> env;

  std::string name;
  State state;
  State target;
  int level;
  bool have_got_target;
  std::vector<TrackedObject> tracked_objects;

  VehicleBase(std::string _name)
      : name(_name), level(0), have_got_target(false) {
    state = State(0, 0, 0, 0, 0, 0, 0, 0);
    target = State(0, 0, 0, 0, 0, 0, 0, 0);
  }
  virtual ~VehicleBase(){};
  void set_target(State tar);
  void set_level(int l);
  bool is_get_target(void) const;

  static void initialize(std::shared_ptr<EnvCrossroads> _env, double _len,
                         double _width, double _safe_len, double _safe_width) {
    VehicleBase::length = _len;
    VehicleBase::width = _width;
    VehicleBase::safe_length = _safe_len;
    VehicleBase::safe_width = _safe_width;
    VehicleBase::env = _env;
  }

  static Eigen::Matrix<double, 2, 5> get_box2d(const State &tar_offset) {
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle;
    vehicle << -VehicleBase::length / 2, VehicleBase::length / 2,
        VehicleBase::length / 2, -VehicleBase::length / 2,
        -VehicleBase::length / 2, VehicleBase::width / 2,
        VehicleBase::width / 2, -VehicleBase::width / 2,
        -VehicleBase::width / 2, VehicleBase::width / 2;
    Eigen::Matrix2d rot;
    rot << cos(tar_offset.yaw), -sin(tar_offset.yaw), sin(tar_offset.yaw),
        cos(tar_offset.yaw);

    vehicle = rot * vehicle;
    vehicle += Eigen::Vector2d(tar_offset.x, tar_offset.y).replicate(1, 5);

    return vehicle;
  }

  static Eigen::Matrix<double, 2, 5> get_safezone(const State &tar_offset) {
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> safezone;
    safezone << -VehicleBase::safe_length / 2, VehicleBase::safe_length / 2,
        VehicleBase::safe_length / 2, -VehicleBase::safe_length / 2,
        -VehicleBase::safe_length / 2, VehicleBase::safe_width / 2,
        VehicleBase::safe_width / 2, -VehicleBase::safe_width / 2,
        -VehicleBase::safe_width / 2, VehicleBase::safe_width / 2;
    Eigen::Matrix2d rot;
    rot << cos(tar_offset.yaw), -sin(tar_offset.yaw), sin(tar_offset.yaw),
        cos(tar_offset.yaw);

    safezone = rot * safezone;
    safezone += Eigen::Vector2d(tar_offset.x, tar_offset.y).replicate(1, 5);

    return safezone;
  }
  static Eigen::Matrix<double, 2, 5>
  get_tractor_box2d(const State &tar_offset) {
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle;
    vehicle << -VehicleBase::tractor_length / 2 - 2,
        VehicleBase::tractor_length / 2 - 2,
        VehicleBase::tractor_length / 2 - 2,
        -VehicleBase::tractor_length / 2 - 2,
        -VehicleBase::tractor_length / 2 - 2, VehicleBase::tractor_width / 2,
        VehicleBase::tractor_width / 2, -VehicleBase::tractor_width / 2,
        -VehicleBase::tractor_width / 2, VehicleBase::tractor_width / 2;
    Eigen::Matrix2d rot;
    rot << cos(tar_offset.yaw), -sin(tar_offset.yaw), sin(tar_offset.yaw),
        cos(tar_offset.yaw);

    vehicle = rot * vehicle;
    vehicle += Eigen::Vector2d(tar_offset.x, tar_offset.y).replicate(1, 5);
    // spdlog::info(fmt::format("tractor was activated"));
    return vehicle;
  }

  static Eigen::Matrix<double, 2, 5>
  get_tractor_safe_box2d(const State &tar_offset) {
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle;
    vehicle << -(VehicleBase::tractor_length + 1) / 2 - 2,
        (VehicleBase::tractor_length + 1) / 2 - 2,
        (VehicleBase::tractor_length + 1) / 2 - 2,
        -(VehicleBase::tractor_length + 1) / 2 - 2,
        -(VehicleBase::tractor_length + 1) / 2 - 2,
        (VehicleBase::tractor_width + 0.4) / 2,
        (VehicleBase::tractor_width + 0.4) / 2,
        -(VehicleBase::tractor_width + 0.4) / 2,
        -(VehicleBase::tractor_width + 0.4) / 2,
        (VehicleBase::tractor_width + 0.4) / 2;
    Eigen::Matrix2d rot;
    rot << cos(tar_offset.yaw), -sin(tar_offset.yaw), sin(tar_offset.yaw),
        cos(tar_offset.yaw);

    vehicle = rot * vehicle;
    vehicle += Eigen::Vector2d(tar_offset.x, tar_offset.y).replicate(1, 5);
    // spdlog::info(fmt::format("tractor was activated"));
    return vehicle;
  }

  static Eigen::Matrix<double, 2, 5> get_tail_box2d(const State &tar_offset) {
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle;
    vehicle << -VehicleBase::tail_length / 2, VehicleBase::tail_length / 2,
        VehicleBase::tail_length / 2, -VehicleBase::tail_length / 2,
        -VehicleBase::tail_length / 2, VehicleBase::tail_width / 2,
        VehicleBase::tail_width / 2, -VehicleBase::tail_width / 2,
        -VehicleBase::tail_width / 2, VehicleBase::tail_width / 2;
    Eigen::Matrix2d rot;
    rot << cos(tar_offset.tail_yaw), -sin(tar_offset.tail_yaw),
        sin(tar_offset.tail_yaw), cos(tar_offset.tail_yaw);

    vehicle = rot * vehicle;
    vehicle +=
        Eigen::Vector2d(tar_offset.tail_x, tar_offset.tail_y).replicate(1, 5);
    // spdlog::info(fmt::format("tail was activated"));
    return vehicle;
  }
  static Eigen::Matrix<double, 2, 5>
  get_tail_safe_box2d(const State &tar_offset) {
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle;
    vehicle << -(VehicleBase::tail_length + 1) / 2,
        (VehicleBase::tail_length + 1) / 2, (VehicleBase::tail_length + 1) / 2,
        -(VehicleBase::tail_length + 1) / 2,
        -(VehicleBase::tail_length + 1) / 2, (VehicleBase::tail_width + 1) / 2,
        (VehicleBase::tail_width + 1) / 2, -(VehicleBase::tail_width + 1) / 2,
        -(VehicleBase::tail_width + 1) / 2, (VehicleBase::tail_width + 1) / 2;
    Eigen::Matrix2d rot;
    rot << cos(tar_offset.tail_yaw), -sin(tar_offset.tail_yaw),
        sin(tar_offset.tail_yaw), cos(tar_offset.tail_yaw);

    vehicle = rot * vehicle;
    vehicle +=
        Eigen::Vector2d(tar_offset.tail_x, tar_offset.tail_y).replicate(1, 5);
    // spdlog::info(fmt::format("tail was activated"));
    return vehicle;
  }
};

#endif
