#include "isaaclab/utils/fk_utils.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void expect_near(
    const std::string& name,
    const Eigen::Vector3f& actual,
    const Eigen::Vector3f& expected,
    float tolerance = 1.0e-5f)
{
    const float error = (actual - expected).norm();
    if (error > tolerance) {
        throw std::runtime_error(
            name + " error " + std::to_string(error) + " exceeds tolerance");
    }
}

void check_pose(
    const std::array<float, 12>& q,
    const std::array<Eigen::Vector3f, 4>& expected)
{
    const auto left_params = isaaclab::fk::kLeftLegParams();
    const auto right_params = isaaclab::fk::kRightLegParams();
    const auto left = isaaclab::fk::legFootLandmarks(q.data(), left_params.data());
    const auto right =
        isaaclab::fk::legFootLandmarks(q.data() + 6, right_params.data());

    expect_near("left_toe", left.toe, expected[0]);
    expect_near("left_heel", left.heel, expected[1]);
    expect_near("right_toe", right.toe, expected[2]);
    expect_near("right_heel", right.heel, expected[3]);
}

}  // namespace

int main()
{
    // Golden positions were generated with MuJoCo 3.8.1 from the training
    // asset src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml.
    check_pose(
        {-0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
         -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f},
        {
            Eigen::Vector3f(0.093998399f, 0.118506455f, -0.779202182f),
            Eigen::Vector3f(-0.076001601f, 0.118506455f, -0.779202182f),
            Eigen::Vector3f(0.093998399f, -0.118506455f, -0.779202182f),
            Eigen::Vector3f(-0.076001601f, -0.118506455f, -0.779202182f),
        });

    check_pose(
        {0.009457300f, 0.360370960f, -0.284672310f, 0.358919560f,
         -0.150534840f, -0.061338840f, 0.262162080f, -0.072640690f,
         0.039674950f, -0.377952710f, 0.202810490f, 0.030514650f},
        {
            Eigen::Vector3f(-0.007922840f, 0.381485485f, -0.735967195f),
            Eigen::Vector3f(-0.169545176f, 0.400502573f, -0.686808565f),
            Eigen::Vector3f(0.062931253f, -0.160196642f, -0.782107754f),
            Eigen::Vector3f(-0.106249563f, -0.166924821f, -0.766857071f),
        });

    std::cout << "G1 FK contract test passed.\n";
    return 0;
}
