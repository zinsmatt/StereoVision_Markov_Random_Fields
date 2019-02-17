#include "loopy_belief_propagation.h"


int data_cost(cv::Mat const& left, cv::Mat const& right, int x, int y, int label, int block_size)
{
  int block_half_size = block_size / 2;
  cv::Mat block_left = left(cv::Range(y - block_half_size, y + block_half_size + 1),
                            cv::Range(x - block_half_size, x + block_half_size + 1));

  cv::Mat block_right = right(cv::Range(y - block_half_size, y + block_half_size + 1),
                              cv::Range(x - block_half_size - label, x + block_half_size + 1 - label));

  return cv::sum(cv::abs(block_left - block_right))[0] / (block_size * block_size);
}

int smoothness_cost(int a, int b)
{
  int d = a - b;
  return 1.0 * std::min(std::abs(d), 10);
}

void initialize_data_cost(const cv::Mat &left, const cv::Mat &right, Markov_random_field &mrf)
{
  mrf.width = left.cols;
  mrf.height = left.rows;
  int total = left.rows * left.cols;
  mrf.grid.resize(total);

  for (int i = 0; i < total; ++i)
  {
    for (int k = 0; k < 5; ++k)
    {
      for (int j = 0; j < LABELS; ++j)
      {
        mrf.grid[i].msg[k][j] = 0;
      }
    }
  }

  int border = LABELS + BLOCK_SIZE / 2;
  for (int i = border; i < left.rows - border; ++i)
  {
    for (int j = border; j < left.cols - border; ++j)
    {
      for (int l = 0; l < LABELS; ++l)
      {
        mrf.grid[i * left.cols + j].msg[MessageDirection::DATA][l] = data_cost(left, right,
                                                                               j, i, l, BLOCK_SIZE);
      }
    }
  }

}

void send_message(Markov_random_field &mrf, int x, int y, MessageDirection direction)
{
  std::array<int, LABELS> new_msg;
  for (int l = 0; l < LABELS; ++l)
  {
    int min_cost = std::numeric_limits<int>::max();
    for (int l2 = 0; l2 < LABELS; ++l2)
    {
      int cost = 0;
      cost += smoothness_cost(l, l2);
      cost += mrf.grid[y * mrf.width + x].msg[MessageDirection::DATA][l2];

      if (direction != MessageDirection::LEFT)
        cost += mrf.grid[y * mrf.width + x].msg[MessageDirection::LEFT][l2];
      if (direction != MessageDirection::RIGHT)
        cost += mrf.grid[y * mrf.width + x].msg[MessageDirection::RIGHT][l2];
      if (direction != MessageDirection::UP)
        cost += mrf.grid[y * mrf.width + x].msg[MessageDirection::UP][l2];
      if (direction != MessageDirection::DOWN)
        cost += mrf.grid[y * mrf.width + x].msg[MessageDirection::DOWN][l2];

      min_cost = std::min(min_cost, cost);
    }
    new_msg[l] = min_cost;
  }

  switch (direction)
  {
    case MessageDirection::LEFT:
      mrf.grid[y * mrf.width + x - 1].msg[MessageDirection::RIGHT] = new_msg;
      break;
    case MessageDirection::RIGHT:
      mrf.grid[y * mrf.width + x + 1].msg[MessageDirection::LEFT] = new_msg;
      break;
    case MessageDirection::UP:
      mrf.grid[(y-1) * mrf.width + x].msg[MessageDirection::DOWN] = new_msg;
      break;
    case MessageDirection::DOWN:
      mrf.grid[(y+1) * mrf.width + x].msg[MessageDirection::UP] = new_msg;
  }

}

void propagate_belief(Markov_random_field &mrf, MessageDirection direction)
{
  int w = mrf.width;
  int h = mrf.height;
  switch (direction)
  {
    case MessageDirection::RIGHT:
      for (int y = 0; y < h; ++y)
      {
        for (int x = 0; x < w - 1; ++x)
        {
          send_message(mrf, x, y, direction);
        }
      }
      break;

    case MessageDirection::LEFT:
      for (int y = 0; y < h; ++y)
      {
        for (int x = w - 1; x > 0; --x)
        {
          send_message(mrf, x, y, direction);
        }
      }
      break;

    case MessageDirection::DOWN:
      for (int x = 0; x < w; ++x)
      {
        for (int y = 0; y < h - 1; ++y)
        {
          send_message(mrf, x, y, direction);
        }
      }
      break;

    case MessageDirection::UP:
      for (int x = 0; x > w; +x)
      {
        for (int y = h - 1; y > 0; --y)
        {
          send_message(mrf, x, y, direction);
        }
      }
      break;
  }
}

int maximum_a_posteriori(Markov_random_field &mrf)
{
  for (int i = 0; i < mrf.grid.size(); ++i)
  {
    int min_cost = std::numeric_limits<int>::max();
    for (int l = 0; l < LABELS; ++l)
    {
      int cost = 0;
      for (int k = 0; k < 5; ++k)
      {
        cost += mrf.grid[i].msg[k][l];
      }

      if (cost < min_cost)
      {
        min_cost = cost;
        mrf.grid[i].best_assignment = l;
      }
    }
  }

  int w = mrf.width;
  int h = mrf.height;
  int energy = 0;
  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      int cur_label = mrf.grid[i * w + j].best_assignment;
      // add data cost
      energy += mrf.grid[i * w + j].msg[MessageDirection::DATA][cur_label];

      // add smooth cost
      if (i > 0)
        energy += smoothness_cost(mrf.grid[(i-1) * w + j].best_assignment, cur_label);
      if (j > 0)
        energy += smoothness_cost(mrf.grid[i * w + j - 1].best_assignment, cur_label);
      if (i < h - 1)
        energy += smoothness_cost(mrf.grid[(i+1) * w + j].best_assignment, cur_label);
      if (j < w - 1)
        energy += smoothness_cost(mrf.grid[i * w + j + 1].best_assignment, cur_label);
    }
  }
  return energy;
}


cv::Mat stereo_belief_propagation(const cv::Mat &left, const cv::Mat &right, int nb_iter)
{
  Markov_random_field mrf;
  initialize_data_cost(left, right, mrf);

  for (int i = 0; i < nb_iter; ++i)
  {
    propagate_belief(mrf, MessageDirection::RIGHT);
    propagate_belief(mrf, MessageDirection::LEFT);
    propagate_belief(mrf, MessageDirection::UP);
    propagate_belief(mrf, MessageDirection::DOWN);
  }

  int energy = maximum_a_posteriori(mrf);
  cv::Mat disparity(left.rows, left.cols, CV_8U, static_cast<int>(0));
  int block_half_size = BLOCK_SIZE / 2;
  for (int i = LABELS + block_half_size ; i < disparity.rows - LABELS - block_half_size; ++i)
  {
    for (int j = LABELS + block_half_size; j < disparity.cols - LABELS - block_half_size; ++j)
    {
      disparity.at<uchar>(i, j) = mrf.grid[i * mrf.width + j].best_assignment;
    }
  }
  return disparity;
}
