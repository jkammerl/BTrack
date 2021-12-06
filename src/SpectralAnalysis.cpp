#include <cassert>
#include <ccomplex>
#include <vector>

#include "SpectralAnalysis.h"
#include "kiss_fft.h"

class SpectralAnalysisImpl {
 public:
  SpectralAnalysisImpl() {}
  ~SpectralAnalysisImpl();
  void Init(int fft_size, WindowType window_type);

  void ComputeAcf(const std::vector<double>& input,
                  std::vector<double>* output);

  void ComputeForwardFft(const std::vector<double>& input,
                         std::vector<std::complex<float>>* output);

 private:
  //=======================================================================
  /** Calculate a Rectangular window */
  void calculateRectangularWindow();

  /** Calculate a Hanning window */
  void calculateHanningWindow();

  /** Calculate a Hamming window */
  void calclulateHammingWindow();

  /** Calculate a Blackman window */
  void calculateBlackmanWindow();

  /** Calculate a Tukey window */
  void calculateTukeyWindow();

  int fft_size_ = 0;
  ;
  std::vector<kiss_fft_cpx> fftIn;  /**< FFT input samples, in complex form */
  std::vector<kiss_fft_cpx> fftOut; /**< FFT output samples, in complex form */
  kiss_fft_cfg cfgForwards = nullptr;  /**< Kiss FFT configuration */
  kiss_fft_cfg cfgBackwards = nullptr; /**< Kiss FFT configuration */
  std::vector<double> window_;
};

SpectralAnalysisImpl::~SpectralAnalysisImpl() {
  if (cfgForwards != nullptr) {
    free(cfgForwards);
  }
  if (cfgBackwards != nullptr) {
    free(cfgBackwards);
  }
}

void SpectralAnalysisImpl::Init(int fft_size, WindowType window_type) {
  fft_size_ = fft_size;
  cfgForwards = kiss_fft_alloc(fft_size, 0, 0, 0);
  cfgBackwards = kiss_fft_alloc(fft_size, 1, 0, 0);
  fftIn.resize(fft_size);
  fftOut.resize(fft_size);
  window_.resize(fft_size);

  // set the window to the specified type
  switch (window_type) {
    case RectangularWindow:
      calculateRectangularWindow();  // Rectangular window
      break;
    case HanningWindow:
      calculateHanningWindow();  // Hanning Window
      break;
    case HammingWindow:
      calclulateHammingWindow();  // Hamming Window
      break;
    case BlackmanWindow:
      calculateBlackmanWindow();  // Blackman Window
      break;
    case TukeyWindow:
      calculateTukeyWindow();  // Tukey Window
      break;
    default:
      calculateHanningWindow();  // DEFAULT: Hanning Window
  }
}

void SpectralAnalysisImpl::ComputeAcf(const std::vector<double>& input,
                                      std::vector<double>* output) {
  // copy into complex array and zero pad
  for (int i = 0; i < fftIn.size(); i++) {
    if (i < input.size()) {
      fftIn[i].r = input[i];
      fftIn[i].i = 0.0;
    } else {
      fftIn[i].r = 0.0;
      fftIn[i].i = 0.0;
    }
  }

  // execute kiss fft
  kiss_fft(cfgForwards, &fftIn[0], &fftOut[0]);

  // multiply by complex conjugate
  for (int i = 0; i < fftOut.size(); i++) {
    fftOut[i].r = fftOut[i].r * fftOut[i].r + fftOut[i].i * fftOut[i].i;
    fftOut[i].i = 0.0;
  }

  // perform the ifft
  kiss_fft(cfgBackwards, &fftOut[0], &fftIn[0]);

  // calculate absolute value of result
  for (int i = 0; i < output->size(); i++) {
    (*output)[i] = sqrt(fftIn[i].r * fftIn[i].r + fftIn[i].i * fftIn[i].i);
  }
}

void SpectralAnalysisImpl::ComputeForwardFft(
    const std::vector<double>& input,
    std::vector<std::complex<float>>* output) {
  for (int i = 0; i < fftIn.size(); i++) {
    fftIn[i].r = input[i] * window_[i];
    fftIn[i].i = 0.0;
  }

  // execute kiss fft
  kiss_fft(cfgForwards, &fftIn[0], &fftOut[0]);

  // calculate absolute value of result
  output->resize(fftOut.size());
  for (int i = 0; i < fftOut.size(); i++) {
    (*output)[i] = {fftOut[i].r, fftOut[i].i};
  }
}

//=======================================================================
void SpectralAnalysisImpl::calculateHanningWindow() {
  double N;  // variable to store fft_size_ minus 1

  N = (double)(fft_size_ - 1);  // fft_size_ minus 1

  // Hanning window calculation
  for (int n = 0; n < fft_size_; n++) {
    assert(n < window_.size());

    window_[n] = 0.5 * (1 - cos(2 * M_PI * (n / N)));
  }
}

//=======================================================================
void SpectralAnalysisImpl::calclulateHammingWindow() {
  double N;      // variable to store fft_size_ minus 1
  double n_val;  // double version of index 'n'

  N = (double)(fft_size_ - 1);  // fft_size_ minus 1
  n_val = 0;

  // Hamming window calculation
  for (int n = 0; n < fft_size_; n++) {
    assert(n < window_.size());

    window_[n] = 0.54 - (0.46 * cos(2 * M_PI * (n_val / N)));
    n_val = n_val + 1;
  }
}

//=======================================================================
void SpectralAnalysisImpl::calculateBlackmanWindow() {
  double N;      // variable to store fft_size_ minus 1
  double n_val;  // double version of index 'n'

  N = (double)(fft_size_ - 1);  // fft_size_ minus 1
  n_val = 0;

  // Blackman window calculation
  for (int n = 0; n < fft_size_; n++) {
    window_[n] = 0.42 - (0.5 * cos(2 * M_PI * (n_val / N))) +
                 (0.08 * cos(4 * M_PI * (n_val / N)));
    n_val = n_val + 1;
  }
}

//=======================================================================
void SpectralAnalysisImpl::calculateTukeyWindow() {
  double N;      // variable to store fft_size_ minus 1
  double n_val;  // double version of index 'n'
  double alpha;  // alpha [default value = 0.5];

  alpha = 0.5;

  N = (double)(fft_size_ - 1);  // fft_size_ minus 1

  // Tukey window calculation

  n_val = (double)(-1 * ((fft_size_ / 2))) + 1;

  for (int n = 0; n < fft_size_; n++)  // left taper
  {
    assert(n < window_.size());

    if ((n_val >= 0) && (n_val <= (alpha * (N / 2)))) {
      window_[n] = 1.0;
    } else if ((n_val <= 0) && (n_val >= (-1 * alpha * (N / 2)))) {
      window_[n] = 1.0;
    } else {
      window_[n] = 0.5 * (1 + cos(M_PI * (((2 * n_val) / (alpha * N)) - 1)));
    }

    n_val = n_val + 1;
  }
}

//=======================================================================
void SpectralAnalysisImpl::calculateRectangularWindow() {
  // Rectangular window calculation
  for (int n = 0; n < fft_size_; n++) {
    assert(n < window_.size());
    window_[n] = 1.0;
  }
}

SpectralAnalysis::SpectralAnalysis() : impl_(new SpectralAnalysisImpl()) {}
SpectralAnalysis::~SpectralAnalysis() {}

void SpectralAnalysis::Init(int fft_size, WindowType window_type) {
  impl_->Init(fft_size, window_type);
}

void SpectralAnalysis::ComputeAcf(const std::vector<double>& input,
                                  std::vector<double>* output) {
  impl_->ComputeAcf(input, output);
}

void SpectralAnalysis::ComputeForwardFft(
    const std::vector<double>& input,
    std::vector<std::complex<float>>* output) {
  impl_->ComputeForwardFft(input, output);
}
