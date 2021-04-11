//
// Created by brucknem on 12.01.21.
//

#ifndef CAMERASTABILIZATION_FEATUREDETECTION_HPP
#define CAMERASTABILIZATION_FEATUREDETECTION_HPP

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>

#include "TimeMeasurable.hpp"

namespace providentia {
	namespace features {

		/**
		 * Base class for all feature detectors.
		 */
		class FeatureDetectorBase : public providentia::utils::TimeMeasurable {
		protected:

			/**
			 * Flag if the latest mask should be used.
			 */
			bool useLatestMaskFlag = false;

			/**
			 * The current processed GPU frame.
			 */
			cv::cuda::GpuMat processedFrame;

			/**
			 * The current processed CPU frame.
			 */
			cv::Mat frameCPU;

			/**
			 * The current original frame.
			 */
			cv::cuda::GpuMat originalFrame;

			/**
			 * The masks used during processing.
			 */
			cv::cuda::GpuMat latestMask, noMask, currentMask;

			/**
			 * The frame with the features drawn into.
			 */
			cv::Mat drawFrame;

			/**
			 * The CPU feature keypoints.
			 */
			std::vector<cv::KeyPoint> keypointsCPU;

			/**
			 * The CPU feature descriptors.
			 */
			cv::Mat descriptorsCPU;

			/**
			 * The GPU feature descriptors.
			 */
			cv::cuda::GpuMat descriptorsGPU;

			/**
			 * The GPU feature keypoints.
			 */
			cv::cuda::GpuMat keypointsGPU;

			/**
			 * Specific detection implementation.
			 */
			virtual void specificDetect() = 0;

			/**
			 * Sets the current mask from the latest or an empty mask based on the useLatestMask flag.
			 * @see FeatureDetectorBase#useLatestMask
			 */
			void setCurrentMask(cv::Size size = cv::Size());

			/**
			 * @constructor
			 */
			FeatureDetectorBase();

		public:
			/**
			 * @destructor
			 */
			~FeatureDetectorBase() override = default;

			/**
			 * @get
			 */
			const std::vector<cv::KeyPoint> &getKeypoints() const;

			/**
			 * @get
			 */
			const cv::Mat &getDescriptorsCPU() const;

			/**
			 * @get
			 */
			const cv::cuda::GpuMat &getDescriptorsGPU() const;

			/**
			 * @get Gets the current mask from the latest or an empty mask based on the useLatestMask flag.
			 */
			const cv::cuda::GpuMat &getCurrentMask(cv::Size size = cv::Size());

			/**
			 * @get
			 */
			const cv::cuda::GpuMat &getOriginalFrame() const;

			/**
			 * Flag if there is a frame and corresponding keypoints and descriptors present.
			 *
			 * @return true if a detection has been performed previously, false else.
			 */
			bool isEmpty();

			/**
			 * Grayscales the given frame.
			 * Detects keypoints and features using the subclass specific implementations.
			 *
			 * @param frame The frame used for detection.
			 */
			void detect(const cv::cuda::GpuMat &frame);

			/**
			 * Detects keypoints and features using the latest mask or without a mask.
			 *
			 * @param frame The frame used for detection.
			 * @param useLatestMask Flag whether or not to use the latest mask.
			 *
			 * @see FeatureDetectorBase#useLatestMask
			 * @see FeatureDetectorBase#detect(cv::cuda::GpuMat)
			 */
			void detect(const cv::cuda::GpuMat &frame, bool useLatestMask);

			/**
			 * Detects keypoints and features using the given mask.
			 *
			 * @param frame The frame used for detection.
			 * @param mask The mask used for detection.
			 *
			 * @see FeatureDetectorBase#detect(cv::cuda::GpuMat)
			 */
			void detect(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &mask);

			/**
			 * Draws the detected features.
			 *
			 * @return The original image with the features drawn.
			 */
			cv::Mat draw();
		};

		/**
		 * Wrapper for the CUDA SURF feature detector.
		 */
		class SIFTFeatureDetector : public FeatureDetectorBase {
		private:
			/**
			 * The CPU SIFT detector used to detect keypoints and descriptors.
			 */
			cv::Ptr<cv::SIFT> detector;

		protected:
			/**
			 * @copydoc
			 */
			void specificDetect() override;

		public:

			/**
			 * @constructor
			 *
			 * @ref opencv2/xfeatures2d/cuda.hpp -> cv::cuda::SURF_CUDA::create
			 */
			explicit SIFTFeatureDetector(int nfeatures = 0, int nOctaveLayers = 3,
										 double contrastThreshold = 0.04, double edgeThreshold = 10,
										 double sigma = 1.6);

			/**
			 * @destructor
			 */
			~SIFTFeatureDetector() override = default;
		};

		/**
		 * Wrapper for the CUDA SURF feature detector.
		 */
		class SURFFeatureDetector : public FeatureDetectorBase {
		private:
			/**
			 * The CUDA SURF detector used to detect keypoints and descriptors.
			 */
			cv::Ptr<cv::cuda::SURF_CUDA> detector;

		protected:
			/**
			 * @copydoc
			 */
			void specificDetect() override;

		public:

			/**
			 * @constructor
			 *
			 * @ref opencv2/xfeatures2d/cuda.hpp -> cv::cuda::SURF_CUDA::create
			 */
			explicit SURFFeatureDetector(double hessianThreshold = 500, int nOctaves = 4,
										 int nOctaveLayers = 2, bool extended = false, float keypointsRatio = 0.01f,
										 bool upright = false);

			/**
			 * @destructor
			 */
			~SURFFeatureDetector() override = default;

		};

		/**
		 * Wrapper for the CUDA SURF feature detector.
		 */
		class ORBFeatureDetector : public FeatureDetectorBase {
		private:
			/**
			 * The CUDA ORB detector used to detect keypoints and descriptors.
			 */
			cv::Ptr<cv::cuda::ORB> detector;

		protected:
			/**
			 * @copydoc
			 */
			void specificDetect() override;

		public:

			/**
			 * @constructor
			 *
			 * @ref opencv2/cudafeatures2d.hpp -> cv::cuda::ORB::create
			 */
			explicit ORBFeatureDetector(int nfeatures = 1e4,
										float scaleFactor = 1.2f,
										int nlevels = 8,
										int edgeThreshold = 31,
										int firstLevel = 0,
										int wtaK = 2,
										int scoreType = cv::ORB::FAST_SCORE,
										int patchSize = 31,
										int fastThreshold = 20,
										bool blurForDescriptor = false);

			/**
			 * @destructor
			 */
			~ORBFeatureDetector() override = default;
		};

		/**
		 * Wrapper for the CUDA SURF feature detector.
		 */
		class FastFREAKFeatureDetector : public FeatureDetectorBase {
		private:
			/**
			 * The CUDA FastFeature detector used to detect keypoints and descriptors.
			 */
			cv::Ptr<cv::cuda::FastFeatureDetector> detector;
			cv::Ptr<cv::xfeatures2d::FREAK> descriptor;

		protected:
			/**
			 * @copydoc
			 */
			void specificDetect() override;

		public:

			/**
			 * @constructor
			 *
			 * @ref opencv2/cudafeatures2d.hpp -> cv::cuda::FastFeatures::create
			 */
			explicit FastFREAKFeatureDetector(int threshold = 40,
											  bool nonmaxSuppression = true,
											  cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16,
											  int maxNPoints = 500000,
											  bool orientationNormalized = true,
											  bool scaleNormalized = true,
											  float patternScale = 22.0f,
											  int nOctaves = 4,
											  const std::vector<int> &selectedPairs = std::vector<int>());

			/**
			 * @destructor
			 */
			~FastFREAKFeatureDetector() override = default;
		};

		/**
		 * Wrapper for the CUDA SURF feature detector.
		 */
		class StarBRIEFFeatureDetector : public FeatureDetectorBase {
		private:
			/**
			 * The CUDA FastFeature detector used to detect keypoints and descriptors.
			 */
			cv::Ptr<cv::xfeatures2d::StarDetector> detector;
			cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> descriptor;

		protected:
			/**
			 * @copydoc
			 */
			void specificDetect() override;

		public:

			/**
			 * @constructor
			 *
			 * @ref opencv2/cudafeatures2d.hpp -> cv::cuda::FastFeatures::create
			 */
			explicit StarBRIEFFeatureDetector(int maxSize = 45, int responseThreshold = 30,
											  int lineThresholdProjected = 10,
											  int lineThresholdBinarized = 8,
											  int suppressNonmaxSize = 5,
											  int bytes = 64, bool useOrientation = false);

			/**
			 * @destructor
			 */
			~StarBRIEFFeatureDetector() override = default;

		};
	}
}

#endif //CAMERASTABILIZATION_FEATUREDETECTION_HPP
