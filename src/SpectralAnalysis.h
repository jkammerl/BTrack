#ifndef __SpectralAnalysis__H
#define __SpectralAnalysis__H

#include <ccomplex>
#include <memory>
#include <vector>

//=======================================================================
/** The type of window to use when calculating onset detection function samples
 */
enum WindowType {
  RectangularWindow,
  HanningWindow,
  HammingWindow,
  BlackmanWindow,
  TukeyWindow
};

class SpectralAnalysisImpl;

class SpectralAnalysis {
 public:
  SpectralAnalysis();
  ~SpectralAnalysis();

  void Init(int fft_size, WindowType window_type);

  void ComputeAcf(const std::vector<double>& input,
                  std::vector<double>* output);

  void ComputeForwardFft(const std::vector<double>& input,
                         std::vector<std::complex<float>>* output);

 private:
  std::unique_ptr<SpectralAnalysisImpl> impl_;
};

#endif