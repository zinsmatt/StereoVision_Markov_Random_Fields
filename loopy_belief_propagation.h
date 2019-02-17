#ifndef LOOPY_BELIEF_PROPAGATION_H
#define LOOPY_BELIEF_PROPAGATION_H

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

enum MessageDirection { RIGHT, DOWN, LEFT, UP, DATA};

const int LABELS = 16;
const int BLOCK_SIZE = 5;

struct Pixel
{
  std::array<std::array<int, LABELS>, 5> msg;
  int best_assignment;
};

struct Markov_random_field
{
  std::vector<Pixel> grid;
  int width, height;
};

int data_cost(cv::Mat const& left, cv::Mat const& right, int x, int y, int label, int block_size);

int smoothness_cost(int a, int b);

void initialize_data_cost(cv::Mat const& left, cv::Mat const& right, Markov_random_field& mrf);

void send_message(Markov_random_field& mrf, int x, int y, MessageDirection direction);

void propagate_belief(Markov_random_field& mrf, MessageDirection direction);

cv::Mat stereo_belief_propagation(cv::Mat const& left, cv::Mat const& right, int nb_iter);

#endif // LOOPY_BELIEF_PROPAGATION_H
